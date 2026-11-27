"""Opcjonalny Quality Guard: wykrywanie zająknięć, powtórzeń i zgubionych sylab przez lokalny faster-whisper (CPU)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from f5_audiobook.config import WHISPER_CACHE_DIR, load_pronunciation_dict
from f5_audiobook.text.normalizer_pl import normalize_polish_text

_WORD_RE = re.compile(r"[a-ząćęłńóśźż0-9]+", re.IGNORECASE)


@dataclass
class QualityReport:
    passed: bool
    score: float  # 0.0 - 1.0 (im wyżej tym lepiej)
    reason: str
    heard_text: str = ""


def _tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _has_unexpected_stutter(expected_words: list[str], heard_words: list[str]) -> str | None:
    """Wykrywa nieoczekiwane powtórzenie tego samego słowa obok siebie (np. 'test test')."""
    expected_pairs = set(zip(expected_words, expected_words[1:]))
    for w1, w2 in zip(heard_words, heard_words[1:]):
        if w1 == w2 and len(w1) >= 2 and (w1, w2) not in expected_pairs:
            return w1
    return None


def _basic_audio_check(text: str, wav: np.ndarray, sr: int) -> QualityReport | None:
    dur = len(wav) / sr if sr > 0 else 0.0
    chars = max(1, len(text.strip()))
    min_dur = 0.18 if chars <= 8 else 0.25
    if dur < min_dur:
        return QualityReport(False, 0.0, f"Za krótkie audio ({dur:.2f}s)")
    rms = float(np.sqrt(np.mean(wav**2)))
    if rms < 0.012:
        return QualityReport(False, 0.05, f"Audio niemal ciche (RMS={rms:.4f})")

    cps = chars / dur
    min_cps = 2.5 if chars <= 8 else 4.0
    if cps < min_cps:
        return QualityReport(False, 0.25, f"Nienaturalnie wolne tempo ({cps:.1f} zn/s)")
    if cps > 26.0:
        return QualityReport(False, 0.25, f"Nienaturalnie szybkie tempo / ucięte słowa ({cps:.1f} zn/s)")
    return None


def _find_quietest_sample(wav: np.ndarray, sr: int, t_start: float, t_end: float) -> int:
    """Znajduje środek najcichszego okna 15 ms w przedziale [t_start, t_end] (w sekundach)."""
    i0 = max(0, int(round(t_start * sr)))
    i1 = min(len(wav), int(round(t_end * sr)))
    if i1 <= i0:
        return i0
    win = max(1, int(sr * 0.015))
    if i1 - i0 <= win:
        return (i0 + i1) // 2
    step = max(1, win // 3)
    best_idx = (i0 + i1) // 2
    best_rms = float("inf")
    for pos in range(i0, i1 - win + 1, step):
        seg = wav[pos : pos + win]
        rms = float(np.sqrt(np.mean(seg**2)))
        if rms < best_rms:
            best_rms = rms
            best_idx = pos + win // 2
    return best_idx


def _find_acoustic_pause_cut(wav: np.ndarray, sr: int, expected_ratio: float) -> int | None:
    """Fallback akustyczny: szuka najgłębszej przerwy ciszy (>= 90 ms) najbliżej oczekiwanej granicy nośnika."""
    win = max(1, int(sr * 0.015))
    n_win = len(wav) // win
    if n_win < 15:
        return None
    frames = wav[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    thresh = 10.0 ** (-34.0 / 20.0)
    is_silent = rms <= thresh

    lo_w = int(n_win * 0.25)
    hi_w = int(n_win * 0.90)
    target_w = int(n_win * max(0.30, min(0.88, expected_ratio)))

    candidates: list[tuple[float, int]] = []
    i = lo_w
    while i < hi_w:
        if is_silent[i]:
            j = i
            while j < hi_w and is_silent[j]:
                j += 1
            gap_len_ms = (j - i) * 15
            if gap_len_ms >= 90:
                mid_w = (i + j) // 2
                dist = abs(mid_w - target_w)
                # Preferujemy dłuższą pauzę blisko expected_ratio
                score = dist - min(gap_len_ms, 450) * 0.04
                candidates.append((score, mid_w * win))
            i = j
        else:
            i += 1
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class QualityGuard:
    """Weryfikator jakości chunków TTS (faster-whisper: GPU int8_float16 z fallbackiem na CPU int8)."""

    def __init__(
        self,
        model_size: str = "small",
        min_similarity: float = 0.78,
        cache_dir: Path = WHISPER_CACHE_DIR,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
    ) -> None:
        self.model_size = model_size
        self.min_similarity = min_similarity
        self.cache_dir = Path(cache_dir)
        self.requested_device = device
        self.active_device: str = "cpu"
        self._whisper = None
        self._load_error: str | None = None
        self._resamplers: dict[int, object] = {}

    def unload(self) -> None:
        """Zwalnia model Whisper z pamięci RAM/VRAM (ważne przed startem Bielika AI)."""
        import gc

        self._whisper = None
        self._resamplers.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resample_to_16k(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr == 16000:
            return wav.astype(np.float32)
        import torch
        import torchaudio

        resampler = self._resamplers.get(sr)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            self._resamplers[sr] = resampler
        t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        return resampler(t).squeeze(0).numpy()

    def _ensure_loaded(self) -> bool:
        if self._whisper is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from faster_whisper import WhisperModel

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            has_local_cache = any(self.cache_dir.glob(f"*faster-whisper-{self.model_size}*"))

            want_cuda = self.requested_device in ("auto", "cuda") and torch.cuda.is_available()
            if want_cuda:
                try:
                    self._whisper = WhisperModel(
                        self.model_size,
                        device="cuda",
                        compute_type="int8_float16",
                        download_root=str(self.cache_dir),
                        local_files_only=has_local_cache,
                    )
                    self.active_device = "cuda"
                    return True
                except Exception:
                    # Bezpieczny fallback na CPU int8, gdyby CUDA/cuDNN lub VRAM nie pozwolił
                    pass

            self._whisper = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=str(self.cache_dir),
                local_files_only=has_local_cache,
            )
            self.active_device = "cpu"
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    def check(self, expected_text: str, wav: np.ndarray, sr: int) -> QualityReport:
        basic_fail = _basic_audio_check(expected_text, wav, sr)
        if basic_fail is not None:
            return basic_fail

        if not self._ensure_loaded():
            # Fallback do samej heurystyki audio jeśli Whisper niedostępny
            return QualityReport(
                True,
                0.85,
                f"OK (tylko heurystyka audio; Whisper niedostępny: {self._load_error})",
            )

        # faster-whisper oczekuje 16 kHz mono float32
        wav_16k = self._resample_to_16k(wav, sr)

        segments, _info = self._whisper.transcribe(
            wav_16k,
            language="pl",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        heard = " ".join(seg.text.strip() for seg in segments).strip()

        exp_words = _tokenize_words(expected_text)
        # Normalizujemy transkrypcję Whispera tym samym normalizatorem PL,
        # bo Whisper zapisuje usłyszane liczby jako cyfry (np. "1989", "14:30") i oryginalną pisownię ("audiobooka")
        heard_norm = normalize_polish_text(heard, load_pronunciation_dict())
        heard_words = _tokenize_words(heard_norm)

        if not exp_words:
            return QualityReport(True, 1.0, "OK", heard_text=heard)

        stutter_word = _has_unexpected_stutter(exp_words, heard_words)
        exp_joined = " ".join(exp_words)
        heard_joined = " ".join(heard_words)

        # Łączymy podobieństwo na poziomie znaków (wyłapuje "zapadał" vs "zapał" i "audioboboka")
        # oraz na poziomie słów; ignorujemy czyste artefakty spacji Whispera (np. "po wiedział" vs "powiedział")
        char_sim = SequenceMatcher(None, exp_joined, heard_joined).ratio()
        word_sim = SequenceMatcher(None, exp_words, heard_words).ratio()
        nospace_sim = SequenceMatcher(None, "".join(exp_words), "".join(heard_words)).ratio()
        base_score = 0.6 * char_sim + 0.4 * word_sim
        if nospace_sim >= 0.95:
            base_score = max(base_score, 0.95 * nospace_sim)
        score = round(base_score, 3)

        if stutter_word:
            return QualityReport(
                False,
                min(score, 0.5),
                f"Wykryto powtórzenie słowa '{stutter_word}'",
                heard_text=heard,
            )

        if score < self.min_similarity:
            return QualityReport(
                False,
                score,
                f"Niska zgodność wymowy ({score:.2f} < {self.min_similarity:.2f})",
                heard_text=heard,
            )

        return QualityReport(True, score, f"OK ({score:.2f})", heard_text=heard)

    def slice_after_carrier(
        self,
        wav: np.ndarray,
        sr: int,
        carrier_text: str,
        target_text: str,
    ) -> tuple[np.ndarray | None, str]:
        """Wycina docelową krótką frazę po zdaniu-nośniku za pomocą word_timestamps Whispera + minimum RMS.

        Nigdy nie tnie za początkiem pierwszego słowa docelowego; jeśli Whisper nie rozpozna nośnika,
        używa bezpiecznego fallbacku akustycznego po pauzie ciszy.
        """
        from f5_audiobook.engine.f5_ipa import trim_silence

        cut_sample: int | None = None
        note = ""

        if self._ensure_loaded() and self._whisper is not None:
            wav_16k = self._resample_to_16k(wav, sr)

            segments, _ = self._whisper.transcribe(
                wav_16k,
                language="pl",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
            words: list[tuple[str, float, float]] = []
            for seg in segments:
                for w in seg.words or []:
                    tok = "".join(_WORD_RE.findall(w.word)).lower()
                    if tok:
                        words.append((tok, float(w.start), float(w.end)))

            carrier_tokens = _tokenize_words(carrier_text)
            if len(words) >= 2 and carrier_tokens:
                last_carrier_tok = carrier_tokens[-1]
                cut_after_idx: int | None = None
                search_limit = min(len(words) - 1, len(carrier_tokens) + 2)
                for idx in range(search_limit):
                    tok = words[idx][0]
                    if tok == last_carrier_tok or (len(last_carrier_tok) >= 4 and tok.startswith(last_carrier_tok[:4])):
                        cut_after_idx = idx
                        break

                if cut_after_idx is None and len(words) > len(carrier_tokens):
                    if len(carrier_tokens[0]) >= 4 and words[0][0].startswith(carrier_tokens[0][:4]):
                        cut_after_idx = len(carrier_tokens) - 1

                if cut_after_idx is not None and cut_after_idx + 1 < len(words):
                    carrier_end_t = words[cut_after_idx][2]
                    target_start_t = words[cut_after_idx + 1][1]
                    # Tniemy ściśle w ciszy przed początkiem docelowego słowa (nigdy za target_start_t!)
                    if target_start_t > carrier_end_t + 0.03:
                        search_lo = carrier_end_t
                        search_hi = max(search_lo + 0.015, target_start_t - 0.015)
                    else:
                        # Gdy Whisper podał stykające się timestampy, szukamy najcichszego punktu tuż przed target_start_t
                        search_lo = max(0.0, carrier_end_t - 0.04)
                        search_hi = max(search_lo + 0.02, target_start_t)
                    cut_sample = _find_quietest_sample(wav, sr, search_lo, search_hi)
                    note = f"whisper cut@{cut_sample / sr:.2f}s ({carrier_end_t:.2f}..{target_start_t:.2f}s)"

        if cut_sample is None:
            expected_ratio = len(carrier_text) / max(1, len(carrier_text) + len(target_text))
            cut_sample = _find_acoustic_pause_cut(wav, sr, expected_ratio)
            if cut_sample is not None:
                note = f"acoustic cut@{cut_sample / sr:.2f}s"

        if cut_sample is None or cut_sample >= len(wav) - int(sr * 0.12):
            return None, "nie znaleziono granicy po zdaniu-nośniku"

        sliced = wav[cut_sample:].copy()
        sliced = trim_silence(sliced, sr, threshold_db=-38.0, keep_lead_ms=30, keep_tail_ms=55)

        fade = min(int(sr * 0.01), len(sliced) // 4)
        if fade > 0:
            sliced[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            sliced[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        dur = len(sliced) / sr if sr > 0 else 0.0
        rms = float(np.sqrt(np.mean(sliced**2))) if len(sliced) > 0 else 0.0
        if dur < 0.16 or rms < 0.012:
            return None, f"wycinek za krótki/cichy (dur={dur:.2f}s, rms={rms:.4f})"

        return sliced, note
