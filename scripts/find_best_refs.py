"""Analizator długiego nagrania lektora: wybiera TOP 4 najlepsze próbki referencyjne 7-10s dla F5-TTS IPA
i generuje test porównawczy (A/B/C/D vs obecny ref_7s.wav) na tym samym tekście literackim."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from f5_audiobook.config import WHISPER_CACHE_DIR
from f5_audiobook.engine.f5_ipa import F5IpaEngine
from f5_audiobook.voices import load_voice_profile

SRC_AUDIO = Path("voices/moj_lektor/gosztyla1.mp3")
CAND_DIR = Path("voices/moj_lektor/candidates")
OUT_DIR = Path("output/ref_candidates_test")

# Realny, wymagający fragment testowy (narracja + dialog + pytanie + spokojne zakończenie)
TEST_CHUNKS = [
    ("Nie wolno ci ponownie schodzić pod wodę, Kowalska, powiedział porucznik Bartosz.", 0.60),
    ("Wiatr znowu wieje od północy, a zaledwie przed chwilą otrzymaliśmy ostrzeżenie o możliwym sztormie.", 0.60),
    ("Wanda przypinała już do stelaża dwie nowe butle z tlenem, ale przerwała tę czynność i wyjrzała przez okno bazy nurków.", 0.60),
    ("Czy naprawdę nie widzisz, że nadciąga burza? Zapytał cicho, opierając dłoń na zimnej szybie.", 0.0),
]

POLISH_DIACRITICS = set("ąćęłńóśźż")


@dataclass
class Candidate:
    start: float
    end: float
    duration: float
    text: str
    cps: float
    sentences: int
    diacritics_count: int
    avg_logprob: float
    rms: float
    rms_std: float
    speech_ratio: float
    max_internal_silence: float
    score: float
    tag: str = ""


def decode_to_wavs(src: Path, tmp_dir: Path) -> tuple[Path, Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_16k = tmp_dir / "_scan_16k.wav"
    wav_24k = tmp_dir / "_scan_24k.wav"

    if not wav_16k.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_16k)],
            check=True, capture_output=True,
        )
    if not wav_24k.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav_24k)],
            check=True, capture_output=True,
        )
    return wav_16k, wav_24k


def analyze_audio_slice(wav: np.ndarray, sr: int, start: float, end: float) -> tuple[float, float, float, float, float]:
    s = max(0, int(start * sr))
    e = min(len(wav), int(end * sr))
    clip = wav[s:e]
    if len(clip) < sr:
        return 0.0, 0.0, 0.0, 99.0, 1.0

    peak = float(np.abs(clip).max())
    rms = float(np.sqrt(np.mean(clip**2)))

    win = int(sr * 0.05)  # 50 ms okna
    n_win = len(clip) // win
    frames = clip[: n_win * win].reshape(n_win, win)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))

    voiced = frame_rms > 0.015
    speech_ratio = float(np.mean(voiced))
    voiced_rms = frame_rms[voiced]
    rms_std = float(np.std(voiced_rms)) if len(voiced_rms) > 3 else 0.0

    # Maksymalna cisza wewnątrz (pomijając pierwsze/ostatnie 200 ms)
    inner = ~voiced[4:-4] if len(voiced) > 10 else ~voiced
    max_run = 0
    cur = 0
    for is_sil in inner:
        if is_sil:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    max_sil_sec = max_run * 0.05
    return rms, rms_std, speech_ratio, max_sil_sec, peak


def extract_sentences(segments) -> list[dict]:
    """Łączy słowa z Whispera w czyste zdania ze znacznikami czasu początku i końca."""
    words: list[dict] = []
    for seg in segments:
        if not seg.words:
            continue
        seg_lp = float(getattr(seg, "avg_logprob", -0.5))
        seg_nsp = float(getattr(seg, "no_speech_prob", 0.0))
        if seg_nsp > 0.15:
            continue
        for w in seg.words:
            txt = w.word.strip()
            if not txt:
                continue
            words.append({
                "word": txt,
                "start": float(w.start),
                "end": float(w.end),
                "prob": float(getattr(w, "probability", 0.9)),
                "seg_lp": seg_lp,
            })

    sentences: list[dict] = []
    buf: list[dict] = []
    for w in words:
        buf.append(w)
        if re.search(r"[.!?]$", w["word"]):
            text = " ".join(x["word"] for x in buf)
            text = re.sub(r"\s+([.,!?;:])", r"\1", text).strip()
            avg_prob = float(np.mean([x["prob"] for x in buf]))
            avg_lp = float(np.mean([x["seg_lp"] for x in buf]))
            sentences.append({
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": text,
                "avg_prob": avg_prob,
                "avg_lp": avg_lp,
            })
            buf = []
    return sentences


def find_candidates(wav_24k: np.ndarray, sr: int, sentences: list[dict]) -> list[Candidate]:
    cands: list[Candidate] = []
    n = len(sentences)

    for i in range(n):
        # Pomijamy pierwsze 40 s (czołówka/tytuł/autor/muzyka)
        if sentences[i]["start"] < 40.0:
            continue

        # Sprawdzamy okna 1-zdaniowe i 2-zdaniowe
        for span in (1, 2):
            j = i + span - 1
            if j >= n:
                break

            start = sentences[i]["start"]
            end = sentences[j]["end"]
            dur = end - start

            if not (6.8 <= dur <= 10.6):
                continue

            # Sprawdź czyste pauzy przed początkiem i po końcu (żeby cięcie było idealne!)
            prev_gap = (start - sentences[i - 1]["end"]) if i > 0 else 0.5
            next_gap = (sentences[j + 1]["start"] - end) if j + 1 < n else 0.5
            if prev_gap < 0.22 or next_gap < 0.22:
                continue

            # Dokładamy delikatny oddech 80ms przed i 120ms po
            clip_start = max(0.0, start - min(0.08, prev_gap * 0.4))
            clip_end = end + min(0.12, next_gap * 0.4)
            clip_dur = clip_end - clip_start

            text = " ".join(sentences[k]["text"] for k in range(i, j + 1)).strip()

            # Filtrujemy czystą narrację: bez cyfr, bez myślników dialogowych, bez krzyków/pytań, pełna kropka na końcu
            if not text or not text[0].isupper() or not text.endswith("."):
                continue
            if re.search(r"[\d\"„”«»()\[\]—–!?…]", text):
                continue
            # Unikamy skrótów w środku
            if re.search(r"\b(np|tj|tzw|prof|dr|godz|ul|nr|r|w)\.", text, re.IGNORECASE):
                continue

            avg_prob = float(np.mean([sentences[k]["avg_prob"] for k in range(i, j + 1)]))
            avg_lp = float(np.mean([sentences[k]["avg_lp"] for k in range(i, j + 1)]))
            if avg_prob < 0.85 or avg_lp < -0.35:
                continue

            cps = len(text) / clip_dur
            if not (9.5 <= cps <= 15.0):
                continue

            rms, rms_std, speech_ratio, max_sil, peak = analyze_audio_slice(wav_24k, sr, clip_start, clip_end)
            if rms < 0.04 or peak > 0.99 or speech_ratio < 0.72 or max_sil > 0.75:
                continue

            unique_diacritics = len(set(text.lower()) & POLISH_DIACRITICS)
            has_comma = "," in text
            has_r = "r" in text.lower()

            # Punktacja "złotej próbki"
            score = 0.0
            # 1. Optymalna długość (7.8 - 9.5 s)
            score += max(0.0, 2.0 - abs(clip_dur - 8.6) * 0.8)
            # 2. Naturalne tempo lektorskie (11.2 - 13.0 zn/s)
            score += max(0.0, 2.5 - abs(cps - 12.1) * 0.8)
            # 3. Bogactwo polskich znaków i interpunkcji
            score += min(unique_diacritics, 7) * 0.45
            score += 0.8 if has_comma else 0.0
            score += 0.4 if has_r else 0.0
            # 4. Jakość audio i ekspresyjna, ale równa dynamika
            score += speech_ratio * 2.0
            score += max(0.0, 1.2 - max_sil * 1.5)
            score += avg_prob * 2.0
            score += min(rms_std * 10.0, 1.0)  # lekka naturalna dynamika głosu, nie płaski robot

            cands.append(
                Candidate(
                    start=round(clip_start, 3),
                    end=round(clip_end, 3),
                    duration=round(clip_dur, 2),
                    text=text,
                    cps=round(cps, 2),
                    sentences=span,
                    diacritics_count=unique_diacritics,
                    avg_logprob=round(avg_lp, 3),
                    rms=round(rms, 4),
                    rms_std=round(rms_std, 4),
                    speech_ratio=round(speech_ratio, 3),
                    max_internal_silence=round(max_sil, 2),
                    score=round(score, 3),
                )
            )

    return cands


def pick_diverse_top4(cands: list[Candidate]) -> list[Candidate]:
    """Wybiera 4 zróżnicowane, najlepsze próbki (oddalone od siebie o min. 45s)."""
    chosen: list[Candidate] = []

    def _far_enough(c: Candidate) -> bool:
        return all(abs(c.start - x.start) > 45.0 for x in chosen)

    def _pick(pool: list[Candidate], tag: str) -> None:
        for c in sorted(pool, key=lambda x: x.score, reverse=True):
            if _far_enough(c):
                c.tag = tag
                chosen.append(c)
                return

    # 1. Absolutny #1 — najbardziej zbalansowana narracja (11.3 - 13.2 zn/s)
    _pick([c for c in cands if 11.3 <= c.cps <= 13.2], "01_zbalansowany_top")
    # 2. Dwuzdaniowy (zawiera pełną kropkę + start nowego zdania wewnątrz referencji — uczy akcentowania końca zdania!)
    _pick([c for c in cands if c.sentences == 2 and c.max_internal_silence <= 0.65], "02_dwuzdaniowy_kadencja")
    # 3. Spokojniejszy, nieco wolniejszy klasyczny Gosztyła (9.8 - 11.4 zn/s)
    _pick([c for c in cands if 9.8 <= c.cps < 11.4], "03_spokojny_gleboki")
    # 4. Żywszy, bardziej dynamiczny (wyższe rms_std / tempo 12.2 - 14.2 zn/s)
    _pick([c for c in cands if c.cps >= 12.2], "04_dynamiczny_ekspresyjny")

    # Uzupełnij do 4, jeśli któryś koszyk był pusty
    for idx in range(len(chosen), 4):
        _pick(cands, f"0{idx + 1}_top_ogolny")

    return chosen


def generate_comparison(engine: F5IpaEngine, ref_wav: Path, ref_text: str, speed: float, out_path: Path) -> float:
    waves = []
    sr = engine.sample_rate
    for i, (txt, pause_s) in enumerate(TEST_CHUNKS):
        res = engine.synthesize(
            text=txt,
            ref_audio_path=ref_wav,
            ref_text=ref_text,
            speed=speed,
            cfg_strength=2.0,
            nfe_step=32,
            seed=101 + i,
        )
        waves.append(res.wav)
        if pause_s > 0:
            waves.append(np.zeros(int(sr * pause_s), dtype=np.float32))
    full = np.concatenate(waves)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full, sr)
    return len(full) / sr


def main() -> None:
    if not SRC_AUDIO.exists():
        raise FileNotFoundError(f"Brak pliku: {SRC_AUDIO}")

    print(f"=== 1/4: Dekodowanie {SRC_AUDIO.name} do WAV (16kHz + 24kHz mono) ===")
    t0 = time.perf_counter()
    wav_16k_path, wav_24k_path = decode_to_wavs(SRC_AUDIO, CAND_DIR / "_tmp")
    wav_16k, _ = sf.read(str(wav_16k_path), dtype="float32")
    wav_24k, sr_24k = sf.read(str(wav_24k_path), dtype="float32")
    print(f"Zdekodowano {len(wav_24k)/sr_24k/60:.1f} min audio w {time.perf_counter() - t0:.1f}s")

    print("\n=== 2/4: Transkrypcja i wykrywanie zdań (faster-whisper small) ===")
    t1 = time.perf_counter()
    whisper = None
    if torch.cuda.is_available():
        try:
            whisper = WhisperModel(
                "small",
                device="cuda",
                compute_type="float16",
                download_root=str(WHISPER_CACHE_DIR),
                local_files_only=True,
            )
            # Przekazujemy numpy array float32 16kHz bezpośrednio (omija PyAV 19 metadata_errors)
            segments_gen, info = whisper.transcribe(
                wav_16k,
                language="pl",
                beam_size=1,
                vad_filter=True,
                word_timestamps=True,
            )
            segments = list(segments_gen)
            print(f"Przeskanowano na GPU ({len(segments)} segmentów) w {time.perf_counter() - t1:.1f}s")
        except Exception as e:
            print(f"CUDA dla faster-whisper niedostępne ({e}), przełączam na CPU int8...")
            whisper = None

    if whisper is None:
        t1 = time.perf_counter()
        whisper = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_CACHE_DIR),
            local_files_only=True,
        )
        segments_gen, info = whisper.transcribe(
            wav_16k,
            language="pl",
            beam_size=1,
            vad_filter=True,
            word_timestamps=True,
        )
        segments = list(segments_gen)
        print(f"Przeskanowano na CPU ({len(segments)} segmentów) w {time.perf_counter() - t1:.1f}s")

    # Zwalniamy pamięć przed załadowaniem F5-TTS
    del whisper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sentences = extract_sentences(segments)
    print(f"Wykryto {len(sentences)} pełnych zdań.")

    print("\n=== 3/4: Ocena kandydatów i wybór TOP 4 próbek (7-10s) ===")
    cands = find_candidates(wav_24k, sr_24k, sentences)
    print(f"Spełniło rygorystyczne kryteria jakościowe: {len(cands)} kandydatów.")
    top4 = pick_diverse_top4(cands)

    CAND_DIR.mkdir(parents=True, exist_ok=True)
    saved_meta = []
    for c in top4:
        s = int(c.start * sr_24k)
        e = int(c.end * sr_24k)
        clip = wav_24k[s:e]
        wav_path = CAND_DIR / f"{c.tag}.wav"
        txt_path = CAND_DIR / f"{c.tag}.txt"
        sf.write(str(wav_path), clip, sr_24k)
        txt_path.write_text(c.text, encoding="utf-8")
        m_start = f"{int(c.start // 60):02d}:{c.start % 60:05.2f}"
        print(
            f"  [{c.tag}] @{m_start} | {c.duration}s | {c.cps} zn/s | {c.sentences} zd. | "
            f"score={c.score} | \"{c.text}\""
        )
        saved_meta.append({**asdict(c), "wav": str(wav_path), "txt": str(txt_path)})

    (CAND_DIR / "candidates.json").write_text(
        json.dumps(saved_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Sprzątamy ciężkie tymczasowe WAV z całego 33-min pliku
    wav_16k_path.unlink(missing_ok=True)
    wav_24k_path.unlink(missing_ok=True)
    try:
        (CAND_DIR / "_tmp").rmdir()
    except OSError:
        pass

    print("\n=== 4/4: Generowanie porównania A/B na 4-zdaniowym fragmencie z Otchłani ===")
    engine = F5IpaEngine()
    engine.load()

    # Baseline: obecny profil ref_7s.wav (speed=1.15)
    cur_voice = load_voice_profile("moj_lektor")
    d0 = generate_comparison(
        engine, cur_voice.ref_audio_path, cur_voice.ref_text, speed=cur_voice.speed,
        out_path=OUT_DIR / "00_obecny_ref7s.wav",
    )
    print(f"  [OK] 00_obecny_ref7s.wav ({d0:.1f}s) — obecna próbka (speed={cur_voice.speed})")

    # Nowe 4 kandydatki (z naturalnym speed=1.02, żeby nie naciągać ich tempa!)
    for c in top4:
        wav_path = CAND_DIR / f"{c.tag}.wav"
        # Dopasowujemy delikatnie speed tylko dla najwolniejszej kandydatki (<11 zn/s), reszta idzie naturalnie 1.0!
        spd = 1.08 if c.cps < 11.0 else 1.0
        d = generate_comparison(
            engine, wav_path, c.text, speed=spd,
            out_path=OUT_DIR / f"{c.tag}_test.wav",
        )
        print(f"  [OK] {c.tag}_test.wav ({d:.1f}s, speed={spd})")

    print(f"\nGotowe! Pliki porównawcze zapisano w: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
