"""Pomocniczy skrypt do wycinania czystej próbki referencyjnej (6-10s WAV 24kHz mono) z audiobooka."""

import argparse
import subprocess
import sys
from pathlib import Path


def cut_sample(
    input_path: Path,
    output_path: Path,
    start: str,
    duration: float,
    sample_rate: int = 24000,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {input_path}")

    if duration < 4.0 or duration > 12.0:
        print(
            f"[UWAGA] Długość {duration:.1f}s jest poza optymalnym zakresem dla F5-TTS (6-10s, max 12s).",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        start,
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Błąd FFmpeg:\n{result.stderr}")

    print(f"Zapisano próbkę referencyjną: {output_path} ({duration:.1f}s, {sample_rate}Hz mono)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wytnij 6-10s próbkę głosu lektora z istniejącego audiobooka (MP3/M4B/WAV)."
    )
    parser.add_argument("input", type=Path, help="Ścieżka do pliku audio z audiobookiem")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("voices/lektor/ref.wav"),
        help="Ścieżka wyjściowa WAV (domyślnie: voices/lektor/ref.wav)",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=str,
        required=True,
        help="Czas początkowy, np. 00:01:25 lub 85.5",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=8.0,
        help="Długość wycinka w sekundach (optymalnie 6-10s, domyślnie: 8.0)",
    )
    args = parser.parse_args()
    cut_sample(args.input, args.output, args.start, args.duration)


if __name__ == "__main__":
    main()
