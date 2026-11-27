"""Lokalny, 100% offline tagger ról dialogowych oparty na polskim modelu Bielik-4.5B-v3.0-Instruct (llama.cpp CUDA).

Uruchamia lokalny proces llama-server.exe na wolnym porcie localhost, analizuje rozdziały
z dialogami w oknach z kontekstem i przypisuje każdemu fragmentowi rolę:
  - 'narrator' (N) — narracja i didaskalia
  - 'male'     (M) — kwestia wypowiadana przez mężczyznę
  - 'female'   (K) — kwestia wypowiadana przez kobietę

Po zakończeniu pracy zamyka proces llama-server.exe, gwarantując natychmiastowe zwolnienie 100% VRAM.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from f5_audiobook.config import BIELIK_MODEL_DIR, LLAMA_CPP_DIR
from f5_audiobook.pipeline.project_db import ChunkRow

# Szybka polska heurystyka gramatyczna (fallback oraz podpowiedź przy imporcie przed Bielikiem)
_MALE_TAG_RE = re.compile(
    r"\b(?:powiedział|odparł|zapytał|spytał|dodał|rzucił|mruknął|szepnął|krzyknął|wykrzyknął|"
    r"odrzekł|zawołał|westchnął|uśmiechnął|pokiwał|skinął|wyjaśnił|stwierdził|przerwał|"
    r"kontynuował|ciągnął|zauważył|przyznał|zapewnił|ostrzegł|poradził|bąknął|warknął|"
    r"prychnął|syknął|parsknął|zaśmiał|wyszeptał|rzekł|przytaknął)\b",
    re.IGNORECASE,
)
_FEMALE_TAG_RE = re.compile(
    r"\b(?:powiedziała|odparła|zapytała|spytała|dodała|rzuciła|mruknęła|szepnęła|krzyknęła|"
    r"wykrzyknęła|odrzekła|zawołała|westchnęła|uśmiechnęła|pokiwała|skinęła|wyjaśniła|"
    r"stwierdziła|przerwała|kontynuowała|ciągnęła|zauważyła|przyznała|zapewniła|ostrzegła|"
    r"poradziła|bąknęła|warknęła|prychnęła|syknęła|parsknęła|zaśmiała|wyszeptała|rzekła|"
    r"przytaknęła)\b",
    re.IGNORECASE,
)
_MALE_SELF_RE = re.compile(r"\b[a-ząćęłńóśźż]{2,}(?:ałem|iłem|yłem|ełem|ąłem|łem)\b", re.IGNORECASE)
_FEMALE_SELF_RE = re.compile(r"\b[a-ząćęłńóśźż]{2,}(?:ałam|iłam|yłam|ełam|ęłam|łam)\b", re.IGNORECASE)


def heuristic_guess_dialogue_gender(dialogue_text: str, next_narrator_text: str = "", prev_narrator_text: str = "") -> str | None:
    """Szybka polska heurystyka gramatyczna ('male' | 'female' | None) po końcówkach czasowników."""
    for ctx in (next_narrator_text, prev_narrator_text):
        if not ctx:
            continue
        m = _MALE_TAG_RE.search(ctx)
        f = _FEMALE_TAG_RE.search(ctx)
        if m and not f:
            return "male"
        if f and not m:
            return "female"
    # Czasowniki w 1. osobie wewnątrz samej kwestii (np. "Nie wiedziałem" vs "Nie wiedziałam")
    ms = _MALE_SELF_RE.search(dialogue_text)
    fs = _FEMALE_SELF_RE.search(dialogue_text)
    if ms and not fs:
        return "male"
    if fs and not ms:
        return "female"
    return None


def find_bielik_gguf(model_dir: Path = BIELIK_MODEL_DIR) -> Path | None:
    if not model_dir.exists():
        return None
    files = sorted(model_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    return files[0] if files else None


def find_llama_server_exe(llama_dir: Path = LLAMA_CPP_DIR) -> Path | None:
    if not llama_dir.exists():
        return None
    direct = llama_dir / "llama-server.exe"
    if direct.exists():
        return direct
    found = list(llama_dir.rglob("llama-server.exe"))
    return found[0] if found else None


def is_bielik_available() -> tuple[bool, str]:
    gguf = find_bielik_gguf()
    exe = find_llama_server_exe()
    if exe is None:
        return False, f"Brak llama-server.exe w {LLAMA_CPP_DIR}"
    if gguf is None:
        return False, f"Brak modelu Bielik .gguf w {BIELIK_MODEL_DIR}"
    return True, f"{gguf.name} ({gguf.stat().st_size / 1e9:.2f} GB)"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


SYSTEM_PROMPT_PL = (
    "Jesteś reżyserem obsady polskiego audiobooka. Określasz, KTO wypowiada każdą linię dialogową [D<id>] "
    "oraz płeć tej osoby: \"M\" (mężczyzna), \"K\" (kobieta) albo \"N\" (jeśli to jednak narracja/tytuł).\n\n"
    "Kluczowe zasady języka polskiego:\n"
    "1. Każda nowa linia [D<id>] w rozmowie dwóch osób zazwyczaj oznacza ZMIANĘ rozmówcy (naprzemiennie: A -> B -> A -> B), "
    "chyba że narracja [N] wyraźnie wskazuje, że ta sama osoba mówi dalej.\n"
    "2. Zwracanie się do kogoś w wołaczu oznacza ODBIORCĘ, a nie mówcę! Np. jeśli mężczyzna mówi „Pani dyrektor, mamy problem”, "
    "to odbiorcą jest kobieta, ale MÓWI mężczyzna (\"M\"). Jeśli kobieta pyta „Co się stało, Janie?”, mówi kobieta (\"K\").\n"
    "3. Zwracaj WYŁĄCZNIE poprawny JSON w formacie:\n"
    "{\"wynik\": [{\"id\": 2, \"mowi\": \"Imię\", \"g\": \"M\"}, ...]}\n"
    "Uwzględnij wszystkie [D<id>] z BIEŻĄCEGO FRAGMENTU w kolejności występowania."
)


class BielikTagger:
    """Zarządza lokalnym serwerem Bielik-4.5B-v3.0-Instruct (llama-server.exe) i taguje dialogi."""

    def __init__(
        self,
        model_path: Path | None = None,
        server_exe: Path | None = None,
        n_gpu_layers: int = 99,
        ctx_size: int = 4096,
    ) -> None:
        self.model_path = model_path or find_bielik_gguf()
        self.server_exe = server_exe or find_llama_server_exe()
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self._proc: subprocess.Popen | None = None
        self._port: int = 0
        # Zawsze omijaj ewentualne firmowe proxy Windows dla połączeń lokalnych 127.0.0.1
        self._http_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __enter__(self) -> "BielikTagger":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self, log_cb: Callable[[str], None] | None = None) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self.server_exe is None or not self.server_exe.exists():
            raise FileNotFoundError(f"Brak llama-server.exe w {LLAMA_CPP_DIR}. Uruchom scripts/setup_bielik.py.")
        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(f"Brak pliku Bielik .gguf w {BIELIK_MODEL_DIR}. Uruchom scripts/setup_bielik.py.")

        self._port = _pick_free_port()
        cmd = [
            str(self.server_exe),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.ctx_size),
            "--threads",
            "4",
            "--log-disable",
        ]
        if log_cb:
            log_cb(f"Uruchamiam lokalnego Bielika ({self.model_path.name}) na GPU...")

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.server_exe.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        # Czekaj aż serwer odpowie na /health
        health_url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server.exe zakończył się niespodziewanie (kod {self._proc.returncode}).")
            try:
                with self._http_opener.open(health_url, timeout=2.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("status") == "ok":
                            if log_cb:
                                log_cb("Bielik-4.5B gotowy na GPU.")
                            return
            except Exception:
                pass
            time.sleep(0.4)

        self.stop()
        raise TimeoutError("Bielik (llama-server.exe) nie uruchomił się w ciągu 60 sekund.")

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _chat_json(self, user_prompt: str, max_tokens: int = 1024) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Serwer Bielika nie jest uruchomiony.")
        url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
        payload = {
            "model": "bielik",
            "temperature": 0.05,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_PL},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._http_opener.open(req, timeout=120.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        # Bezpieczne wyciągnięcie bloku JSON (także gdyby model otoczył go ```json ... ```)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            # Awaryjne odzyskanie pojedynczych obiektów {"id": ..., "g": "..."} przy ewentualnym ucięciu
            recovered = []
            for om in re.finditer(r'\{\s*"id"\s*:\s*"?(\d+)"?[^{}]*?"g"\s*:\s*"([MKNmkn])"\s*\}', content):
                recovered.append({"id": int(om.group(1)), "g": om.group(2).upper()})
            return {"wynik": recovered} if recovered else {}

    def tag_chapter_chunks(
        self,
        chunks: list[ChunkRow],
        window_size: int = 28,
        context_overlap: int = 6,
    ) -> dict[int, str]:
        """Taguje wszystkie dialogowe chunki rozdziału (voice_role != 'narrator' lub z '—').

        Zwraca słownik {chunk_id: 'narrator' | 'male' | 'female'}.
        """
        # Wybieramy chunki oznaczone jako dialog (lub już wcześniej przydzielone do male/female)
        def _is_dialogue_candidate(c: ChunkRow) -> bool:
            if c.voice_role in ("dialogue", "male", "female"):
                return True
            raw = (c.raw_text or "").lstrip()
            return raw.startswith(("—", "–", "- ", "„", '"'))

        if not any(_is_dialogue_candidate(c) for c in chunks):
            return {}

        assignments: dict[int, str] = {}
        n = len(chunks)
        start = 0
        last_known_role = "male"

        while start < n:
            end = min(n, start + window_size)
            ctx_start = max(0, start - context_overlap)

            # Zbierz dialogi w oknie [start, end)
            window_dialogues: list[ChunkRow] = [c for c in chunks[start:end] if _is_dialogue_candidate(c)]
            if not window_dialogues:
                start = end
                continue

            lines: list[str] = []
            if ctx_start < start:
                lines.append("KONTEKST POPRZEDNI (nie taguj, tylko dla zrozumienia sceny):")
                for c in chunks[ctx_start:start]:
                    role_tag = assignments.get(c.id, c.voice_role)
                    short_r = {"male": "M", "female": "K"}.get(role_tag, "N")
                    lines.append(f"  ({short_r}) {c.raw_text or c.normalized_text}")
                lines.append("")

            lines.append("BIEŻĄCY FRAGMENT (określ M / K / N dla każdego [D<id>]):")
            for c in chunks[start:end]:
                txt = (c.raw_text or c.normalized_text).strip()
                if _is_dialogue_candidate(c):
                    lines.append(f"[D{c.id}] {txt}")
                else:
                    lines.append(f"[N] {txt}")

            prompt = "\n".join(lines)
            parsed_roles: dict[str, str] = {}
            try:
                res = self._chat_json(prompt, max_tokens=max(768, len(window_dialogues) * 64))
                if isinstance(res, dict):
                    # Obsługa formatu listowego {"wynik": [{"id": 2, "mowi": "...", "g": "M"}, ...]}
                    list_val = None
                    for v in res.values():
                        if isinstance(v, list):
                            list_val = v
                            break
                    if list_val is not None:
                        for item in list_val:
                            if isinstance(item, dict):
                                cid_raw = item.get("id")
                                g_raw = item.get("g") or item.get("plec") or item.get("role") or ""
                                if cid_raw is not None and g_raw:
                                    cid_clean = re.sub(r"[^0-9]", "", str(cid_raw))
                                    if cid_clean:
                                        parsed_roles[cid_clean] = str(g_raw).strip().upper()
                    else:
                        # Fallback dla płaskiego słownika lub słownika zagnieżdżonego pod dowolnym kluczem
                        dict_val = res
                        if len(res) == 1 and isinstance(next(iter(res.values())), dict):
                            dict_val = next(iter(res.values()))
                        for k, v in dict_val.items():
                            cid_clean = re.sub(r"[^0-9]", "", str(k))
                            if cid_clean and isinstance(v, str):
                                parsed_roles[cid_clean] = v.strip().upper()
            except Exception:
                parsed_roles = {}

            for idx_in_ch in range(start, end):
                c = chunks[idx_in_ch]
                if not _is_dialogue_candidate(c):
                    continue
                code = parsed_roles.get(str(c.id), "")
                if code.startswith("M"):
                    role = "male"
                elif code.startswith("K") or code.startswith("F"):
                    role = "female"
                elif code.startswith("N"):
                    role = "narrator"
                else:
                    # Fallback do polskiej heurystyki gramatycznej z sąsiednich chunków
                    prev_txt = chunks[idx_in_ch - 1].raw_text if idx_in_ch > 0 else ""
                    next_txt = chunks[idx_in_ch + 1].raw_text if idx_in_ch + 1 < n else ""
                    role = heuristic_guess_dialogue_gender(c.raw_text or c.normalized_text, next_txt, prev_txt) or last_known_role
                assignments[c.id] = role
                if role in ("male", "female"):
                    last_known_role = role

            start = end

        return assignments
