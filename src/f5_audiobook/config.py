"""Konfiguracja ścieżek projektu i zarządzanie słownikiem wymowy."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
VOICES_DIR = PROJECT_ROOT / "voices"
WORK_DIR = PROJECT_ROOT / "work"
OUTPUT_DIR = PROJECT_ROOT / "output"
PRONUNCIATION_FILE = PROJECT_ROOT / "pronunciation.json"

MULTI3_MODEL_DIR = MODELS_DIR / "gregniuki-f5tts-multi3" / "multi3"
VOCODER_DIR = MODELS_DIR / "vocos-mel-24khz"
WHISPER_CACHE_DIR = MODELS_DIR / "faster-whisper"

DEFAULT_PRONUNCIATION: dict[str, str] = {
    "audiobook": "audiobuk",
    "audiobooka": "audiobuka",
    "audiobooki": "audiobuki",
    "audiobooków": "audiobuków",
    "audiobookiem": "audiobukiem",
    "ebook": "ibuk",
    "ebooka": "ibuka",
    "ebooki": "ibuki",
    "ebooków": "ibuków",
    "e-book": "ibuk",
    "e-booka": "ibuka",
    "NVIDIA": "Enwidia",
    "Nvidia": "Enwidia",
    "Windows": "Łindołs",
    "iPhone": "Ajfon",
    "iPhone'a": "Ajfona",
    "AI": "ej aj",
    "GPS": "dżi pi es",
    "PDF": "pe de ef",
    "MP3": "em pe trzy",
    "USB": "u es be",
    "Wi-Fi": "łaj faj",
    "WiFi": "łaj faj",
    "Bluetooth": "blutuf",
    "FBI": "ef bi aj",
    "CIA": "si aj ej",
    "USA": "u es a",
    "ONZ": "o en zet",
    "NATO": "nato",
    "SMS": "es em es",
    "SMS-a": "es em esa",
    "OK": "okej",
}


def ensure_dirs() -> None:
    for d in (MODELS_DIR, VOICES_DIR, WORK_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_pronunciation_dict(path: Path = PRONUNCIATION_FILE) -> dict[str, str]:
    if not path.exists():
        save_pronunciation_dict(DEFAULT_PRONUNCIATION, path)
        return dict(DEFAULT_PRONUNCIATION)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return dict(DEFAULT_PRONUNCIATION)


def save_pronunciation_dict(
    mapping: dict[str, str], path: Path = PRONUNCIATION_FILE
) -> None:
    path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
