"""Zarządzanie profilami lektorów (voices/<nazwa>/profile.json) oraz wycinanie próbek przez FFmpeg."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from f5_audiobook.config import VOICES_DIR


@dataclass
class VoiceProfile:
    name: str
    display_name: str
    ref_audio: str  # ścieżka względna wobec katalogu profilu lub absolutna
    ref_text: str
    speed: float = 0.94
    cfg_strength: float = 2.2
    nfe_step: int = 36
    seed: int = 202
    notes: str = ""

    @property
    def profile_dir(self) -> Path:
        return VOICES_DIR / self.name

    @property
    def ref_audio_path(self) -> Path:
        p = Path(self.ref_audio)
        return p if p.is_absolute() else (self.profile_dir / p)


def slugify_voice_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ_-]+", "_", name.strip()).strip("_")
    return slug.lower() or "lektor"


def list_voice_profiles(voices_dir: Path = VOICES_DIR) -> list[VoiceProfile]:
    voices_dir.mkdir(parents=True, exist_ok=True)
    profiles: list[VoiceProfile] = []
    for sub in sorted(voices_dir.iterdir()):
        if not sub.is_dir():
            continue
        pfile = sub / "profile.json"
        if not pfile.exists():
            continue
        try:
            data = json.loads(pfile.read_text(encoding="utf-8"))
            profiles.append(
                VoiceProfile(
                    name=sub.name,
                    display_name=data.get("display_name", sub.name),
                    ref_audio=data.get("ref_audio", "ref.wav"),
                    ref_text=data.get("ref_text", ""),
                    speed=float(data.get("speed", 1.15)),
                    cfg_strength=float(data.get("cfg_strength", 2.0)),
                    nfe_step=int(data.get("nfe_step", 32)),
                    seed=int(data.get("seed", 101)),
                    notes=str(data.get("notes", "")),
                )
            )
        except Exception:
            continue
    return profiles


def load_voice_profile(name: str, voices_dir: Path = VOICES_DIR) -> VoiceProfile:
    pfile = voices_dir / name / "profile.json"
    if not pfile.exists():
        raise FileNotFoundError(f"Nie znaleziono profilu lektora: {pfile}")
    data = json.loads(pfile.read_text(encoding="utf-8"))
    return VoiceProfile(
        name=name,
        display_name=data.get("display_name", name),
        ref_audio=data.get("ref_audio", "ref.wav"),
        ref_text=data.get("ref_text", ""),
        speed=float(data.get("speed", 1.15)),
        cfg_strength=float(data.get("cfg_strength", 2.0)),
        nfe_step=int(data.get("nfe_step", 32)),
        seed=int(data.get("seed", 101)),
        notes=str(data.get("notes", "")),
    )


def save_voice_profile(profile: VoiceProfile, voices_dir: Path = VOICES_DIR) -> Path:
    target_dir = voices_dir / profile.name
    target_dir.mkdir(parents=True, exist_ok=True)
    pfile = target_dir / "profile.json"
    pfile.write_text(
        json.dumps(asdict(profile), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return pfile


def cut_reference_sample(
    input_audio: Path,
    output_wav: Path,
    start: str,
    duration: float,
    sample_rate: int = 24000,
) -> Path:
    """Wycina czystą próbkę WAV 24 kHz mono z istniejącego pliku audio (MP3/M4B/WAV)."""
    if not input_audio.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku audio: {input_audio}")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(input_audio),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Błąd FFmpeg podczas wycinania próbki:\n{res.stderr}")
    return output_wav
