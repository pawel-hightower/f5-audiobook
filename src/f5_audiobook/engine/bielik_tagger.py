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

# Szybka polska heurystyka gramatyczna + twarde kotwice z didaskaliów
_MALE_TAG_RE = re.compile(
    r"\b(?:powiedział|odparł|zapytał|spytał|dodał|rzucił|mruknął|szepnął|krzyknął|wykrzyknął|"
    r"odrzekł|zawołał|westchnął|uśmiechnął|pokiwał|skinął|wyjaśnił|stwierdził|przerwał|"
    r"kontynuował|ciągnął|zauważył|przyznał|zapewnił|ostrzegł|poradził|bąknął|warknął|"
    r"prychnął|syknął|parsknął|zaśmiał|wyszeptał|rzekł|przytaknął|poprosił|wymamrotał|"
    r"wypalił|jęknął|stęknął|wysapał|wycedził|skwitował|burknął|fuknął|parsknął|dopytywał)\b",
    re.IGNORECASE,
)
_FEMALE_TAG_RE = re.compile(
    r"\b(?:powiedziała|odparła|zapytała|spytała|dodała|rzuciła|mruknęła|szepnęła|krzyknęła|"
    r"wykrzyknęła|odrzekła|zawołała|westchnęła|uśmiechnęła|pokiwała|skinęła|wyjaśniła|"
    r"stwierdziła|przerwała|kontynuowała|ciągnęła|zauważyła|przyznała|zapewniła|ostrzegła|"
    r"poradziła|bąknęła|warknęła|prychnęła|syknęła|parsknęła|zaśmiała|wyszeptała|rzekła|"
    r"przytaknęła|poprosiła|wymamrotała|wypaliła|jęknęła|stęknęła|wysapała|wycedziła|"
    r"skwitowała|burknęła|fuknęła|dopytywała)\b",
    re.IGNORECASE,
)
_MALE_SELF_RE = re.compile(r"\b[a-ząćęłńóśźż]{2,}(?:ałem|iłem|yłem|ełem|ąłem|łem)\b", re.IGNORECASE)
_FEMALE_SELF_RE = re.compile(r"\b[a-ząćęłńóśźż]{2,}(?:ałam|iłam|yłam|ełam|ęłam|łam)\b", re.IGNORECASE)

# Ogólny detektor polskiej końcówki czasownika przeszłego w 3. os. l.p. na początku didaskaliów
_WORD_PL_RE = re.compile(r"[a-ząćęłńóśźż]+", re.IGNORECASE)
_FEM_PAST_VERB_RE = re.compile(
    r"^[a-ząćęłńóśźż]{3,}(?:ała|iła|yła|eła|ęła|rła|sła|zła|dła|tła|kła|gła|bła|pła|wła|chła)$"
)
_MALE_PAST_VERB_RE = re.compile(
    r"^[a-ząćęłńóśźż]{3,}(?:ał|ił|ył|eł|ął|rł|sł|zł|dł|tł|kł|gł|bł|pł|wł|chł)$"
)
# Rzeczowniki, przymiotniki oraz czasowniki bezosobowe/dźwiękowe, które nie określają płci mówcy
_NON_SPEAKER_FEM = {
    "siła", "chwała", "skała", "strzała", "bryła", "mgła", "igła", "mała", "cała", "biała",
    "piła", "sala", "fala", "rozległa", "dobiegła", "padła", "brzmiała", "zabrzmiała",
    "wydobyła", "zapadła", "minęła", "upłynęła", "trwała", "została", "była",
}
_NON_SPEAKER_MALE = {
    "diabeł", "orzeł", "anioł", "poseł", "węzeł", "kocioł", "osioł", "stół", "dół", "wół", "pół",
    "rozległ", "dobiegł", "padł", "brzmiał", "zabrzmiał", "wydobył", "zapadł", "minął",
    "upłynął", "powstał", "nastał", "trwał", "został", "był", "miał", "głos",
}
_LEADING_ADVERBS = {
    "cicho", "głośno", "nagle", "szybko", "powoli", "znowu", "znów", "wreszcie", "zaraz",
    "potem", "wtedy", "ledwo", "ledwie", "prawie", "tylko", "ponownie", "natychmiast",
    "gniewnie", "spokojnie", "delikatnie", "nerwowo", "chłodno", "sucho", "oschle",
    "ciepło", "smutno", "wesoło", "ostro", "łagodnie", "niepewnie", "nieśmiało",
    "lekko", "mocno", "ostrożnie", "cichutko", "półgłosem", "szeptem", "krótko",
}


def extract_didaskalia_gender(narrator_text: str) -> str | None:
    """Deterministycznie określa płeć mówcy ('male' | 'female' | None) z początku polskich didaskaliów.

    Działa dla DOWOLNEGO polskiego czasownika przeszłego w 3. os. l.p. (-ł vs -ła), np.:
    'poprosiła Dżuliet.', 'wymamrotał mężczyzna.', 'cicho wyszeptała.', 'pstryknęła palcem...'.
    """
    txt = (narrator_text or "").lstrip(" —–-\t\n")
    if not txt:
        return None
    words = _WORD_PL_RE.findall(txt)
    if not words:
        return None

    # Sprawdzamy pierwsze słowo, a jeśli to przysłówek sposobu (np. "cicho wyszeptała") — także drugie
    candidates = [words[0].lower()]
    if len(words) >= 2 and candidates[0] in _LEADING_ADVERBS:
        candidates.append(words[1].lower())

    for w in candidates:
        if w in _NON_SPEAKER_FEM or w in _NON_SPEAKER_MALE:
            return None
        if _FEM_PAST_VERB_RE.match(w):
            return "female"
        if _MALE_PAST_VERB_RE.match(w):
            return "male"
    return None


def heuristic_guess_dialogue_gender(dialogue_text: str, next_narrator_text: str = "", prev_narrator_text: str = "") -> str | None:
    """Polska heurystyka gramatyczna ('male' | 'female' | None) po didaskaliach i końcówkach czasowników."""
    if next_narrator_text:
        direct = extract_didaskalia_gender(next_narrator_text)
        if direct:
            return direct
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


def _is_dialogue_chunk(c: ChunkRow) -> bool:
    if c.voice_role in ("dialogue", "male", "female"):
        return True
    raw = (c.raw_text or "").lstrip()
    return raw.startswith(("—", "–", "- ", "„", '"'))


def compute_hard_grammar_anchors(
    chunks: list[ChunkRow],
) -> tuple[dict[int, str], list[tuple[int, int]]]:
    """Wyznacza 100%-pewne kotwice gramatyczne płci oraz powiązane połówki przerwanego akapitu dialogowego.

    Zwraca:
      - anchors: {chunk_id: 'male' | 'female'} — twarde przypisania z jawnych didaskaliów (Bielik nie może ich nadpisać!)
      - linked_pairs: [(first_dialogue_id, continuation_dialogue_id), ...] — połówki tej samej wypowiedzi
        rozdzielone didaskaliami (pauza 220 ms), które ZAWSZE muszą mieć tego samego mówcę.
    """
    anchors: dict[int, str] = {}
    linked_pairs: list[tuple[int, int]] = []
    n = len(chunks)

    for i in range(n - 1):
        cur = chunks[i]
        nxt = chunks[i + 1]
        if cur.chapter_id != nxt.chapter_id:
            continue
        if not _is_dialogue_chunk(cur) or _is_dialogue_chunk(nxt):
            continue

        ntxt = (nxt.normalized_text or nxt.raw_text or "").strip()
        if not ntxt:
            continue
        # Tylko didaskalia z tego samego akapitu (pauza 220 ms lub rozpoczęte małą literą)
        is_same_para_didaskalia = (cur.pause_after_ms == 220) or ntxt[0].islower()
        if not is_same_para_didaskalia:
            continue

        # Jeśli po didaskaliach (220 ms) następuje dalszy ciąg dialogu w tym samym akapicie -> zwiąż je!
        has_cont = (
            nxt.pause_after_ms == 220
            and (i + 2) < n
            and chunks[i + 2].chapter_id == cur.chapter_id
            and _is_dialogue_chunk(chunks[i + 2])
        )
        if has_cont:
            linked_pairs.append((cur.id, chunks[i + 2].id))

        gender = extract_didaskalia_gender(ntxt)
        if gender in ("male", "female"):
            anchors[cur.id] = gender
            if has_cont:
                anchors[chunks[i + 2].id] = gender

    # Propagacja kotwic przez łańcuchy 3-częściowych przerwanych akapitów (A - did - B - did - C)
    changed = True
    while changed:
        changed = False
        for a_id, b_id in linked_pairs:
            if a_id in anchors and b_id not in anchors:
                anchors[b_id] = anchors[a_id]
                changed = True
            elif b_id in anchors and a_id not in anchors:
                anchors[a_id] = anchors[b_id]
                changed = True

    return anchors, linked_pairs


def apply_grammar_anchors_to_project(db) -> dict[str, int]:
    """Koryguje w bazie projektu wszystkie dialogi sprzeczne z polską gramatyką didaskaliów i spójnością akapitu.

    Dla każdego zmienionego chunka (w tym już wygenerowanych 'done') ustawia poprawną rolę i status='pending'.
    """
    chapters = db.list_chapters()
    fixed_total = 0
    fixed_done_reset = 0
    fixed_pending = 0
    anchors_total = 0

    for ch in chapters:
        if not ch.multi_voice:
            continue
        chunks = db.list_chunks(chapter_id=ch.id)
        if not chunks:
            continue
        anchors, linked_pairs = compute_hard_grammar_anchors(chunks)
        anchors_total += len(anchors)

        by_id = {c.id: c for c in chunks}
        target_roles: dict[int, str] = dict(anchors)

        # Dla powiązanych połówek bez twardej kotwicy (didaskalia bezpłciowe) wyrównaj drugą do pierwszej
        for a_id, b_id in linked_pairs:
            a_role = target_roles.get(a_id, by_id[a_id].voice_role)
            if a_role in ("male", "female") and b_id not in target_roles:
                if by_id[b_id].voice_role != a_role:
                    target_roles[b_id] = a_role

        updates: dict[int, str] = {}
        for cid, new_role in target_roles.items():
            cur_c = by_id[cid]
            if cur_c.voice_role != new_role:
                updates[cid] = new_role
                if cur_c.status == "done":
                    fixed_done_reset += 1
                else:
                    fixed_pending += 1

        if updates:
            fixed_total += db.set_chunks_voice_roles_bulk(updates, reset_changed_to_pending=True)

    return {
        "anchors_total": anchors_total,
        "fixed_total": fixed_total,
        "fixed_done_reset": fixed_done_reset,
        "fixed_pending": fixed_pending,
    }


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
        """Taguje wszystkie dialogowe chunki rozdziału łącząc 100%-pewne kotwice gramatyczne PL z Bielikiem AI.

        Zwraca słownik {chunk_id: 'narrator' | 'male' | 'female'}.
        """
        if not any(_is_dialogue_chunk(c) for c in chunks):
            return {}

        # 1) Najpierw twarde kotwice z polskiej gramatyki didaskaliów (Bielik NIE MOŻE ich nadpisać!)
        anchors, linked_pairs = compute_hard_grammar_anchors(chunks)
        assignments: dict[int, str] = dict(anchors)

        n = len(chunks)
        start = 0
        last_known_role = "male"

        while start < n:
            end = min(n, start + window_size)
            ctx_start = max(0, start - context_overlap)

            # Zbierz tylko te dialogi w oknie [start, end), które NIE mają jeszcze twardej kotwicy
            open_dialogues: list[ChunkRow] = [
                c for c in chunks[start:end] if _is_dialogue_chunk(c) and c.id not in anchors
            ]
            if not open_dialogues:
                for c in chunks[start:end]:
                    if c.id in anchors:
                        last_known_role = anchors[c.id]
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

            lines.append("BIEŻĄCY FRAGMENT (określ M / K / N WYŁĄCZNIE dla linii [D<id>]; linie [PEWNE: M/K] są już ustalone):")
            for c in chunks[start:end]:
                txt = (c.raw_text or c.normalized_text).strip()
                if c.id in anchors:
                    sr = "M" if anchors[c.id] == "male" else "K"
                    lines.append(f"[PEWNE: {sr}] {txt}")
                elif _is_dialogue_chunk(c):
                    lines.append(f"[D{c.id}] {txt}")
                else:
                    lines.append(f"[N] {txt}")

            prompt = "\n".join(lines)
            parsed_roles: dict[str, str] = {}
            try:
                res = self._chat_json(prompt, max_tokens=max(768, len(open_dialogues) * 64))
                if isinstance(res, dict):
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
                if c.id in anchors:
                    assignments[c.id] = anchors[c.id]
                    last_known_role = anchors[c.id]
                    continue
                if not _is_dialogue_chunk(c):
                    continue
                code = parsed_roles.get(str(c.id), "")
                if code.startswith("M"):
                    role = "male"
                elif code.startswith("K") or code.startswith("F"):
                    role = "female"
                elif code.startswith("N"):
                    role = "narrator"
                else:
                    prev_txt = chunks[idx_in_ch - 1].raw_text if idx_in_ch > 0 else ""
                    next_txt = chunks[idx_in_ch + 1].raw_text if idx_in_ch + 1 < n else ""
                    role = heuristic_guess_dialogue_gender(c.raw_text or c.normalized_text, next_txt, prev_txt) or last_known_role
                assignments[c.id] = role
                if role in ("male", "female"):
                    last_known_role = role

            start = end

        # 2) Gwarancja spójności przerwanego akapitu (obie połówki przed i po didaskaliach 220 ms mają tego samego mówcę)
        for a_id, b_id in linked_pairs:
            if a_id in anchors:
                assignments[b_id] = anchors[a_id]
            elif b_id in anchors:
                assignments[a_id] = anchors[b_id]
            elif a_id in assignments and assignments[a_id] in ("male", "female"):
                assignments[b_id] = assignments[a_id]

        return assignments
