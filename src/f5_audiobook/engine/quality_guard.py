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
    if dur < 0.25:
        return QualityReport(False, 0.0, f"Za krótkie audio ({dur:.2f}s)")
    rms = float(np.sqrt(np.mean(wav**2)))
    if rms < 0.012:
        return QualityReport(False, 0.05, f"Audio niemal ciche (RMS={rms:.4f})")

    chars = max(1, len(text.strip()))
    cps = chars / dur
    if cps < 4.0:
        return QualityReport(False, 0.25, f"Nienaturalnie wolne tempo ({cps:.1f} zn/s)")
    if cps > 24.0:
        return QualityReport(False, 0.25, f"Nienaturalnie szybkie tempo / ucięte słowa ({cps:.1f} zn/s)")
    return None


class QualityGuard:
    """Weryfikator jakości chunków TTS działający na CPU (0 MB VRAM)."""

    def __init__(
        self,
        model_size: str = "small",
        min_similarity: float = 0.78,
        cache_dir: Path = WHISPER_CACHE_DIR,
    ) -> None:
        self.model_size = model_size
        self.min_similarity = min_similarity
        self.cache_dir = Path(cache_dir)
        self._whisper = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._whisper is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from faster_whisper import WhisperModel

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            has_local_cache = any(self.cache_dir.glob(f"*faster-whisper-{self.model_size}*"))
            # Działa na CPU w int8 — zero obciążenia 6 GB VRAM karty RTX 4050!
            self._whisper = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=str(self.cache_dir),
                local_files_only=has_local_cache,
            )
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
        if sr != 16000:
            import torch
            import torchaudio

            t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
            wav_16k = torchaudio.transforms.Resample(sr, 16000)(t).squeeze(0).numpy()
        else:
            wav_16k = wav.astype(np.float32)

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
        # oraz na poziomie słów
        char_sim = SequenceMatcher(None, exp_joined, heard_joined).ratio()
        word_sim = SequenceMatcher(None, exp_words, heard_words).ratio()
        score = round(0.6 * char_sim + 0.4 * word_sim, 3)

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
