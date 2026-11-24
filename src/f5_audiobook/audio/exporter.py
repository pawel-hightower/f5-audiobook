"""Postprocessing audio (łączenie chunków, EBU R128 -16 LUFS) i eksport do MP3 oraz M4B z rozdziałami i okładką."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from f5_audiobook.config import OUTPUT_DIR
from f5_audiobook.pipeline.project_db import ProjectDB, slugify_project_name


@dataclass
class ExportResult:
    output_dir: Path
    mp3_files: list[Path]
    m4b_file: Path | None
    total_duration_sec: float


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "", name).strip()
    return cleaned[:90] or "rozdzial"


def _escape_ffmetadata(text: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        text = text.replace(ch, f"\\{ch}")
    return text


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"Błąd FFmpeg:\n{' '.join(cmd)}\n\n{res.stderr}")
    return res


def _master_wav_ebu_r128(raw_wav: Path, mastered_wav: Path, target_lufs: float = -16.0) -> None:
    """Dwuprzebiegowy mastering EBU R128 (-16 LUFS) + delikatna kompresja dynamiki pod audiobooki."""
    pre_filter = "highpass=f=70,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=1"

    # Przebieg 1: pomiar głośności (loudnorm print_format=json)
    cmd1 = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_wav),
        "-af",
        f"{pre_filter},loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    res1 = _run_ffmpeg(cmd1)

    # Wyciągamy blok JSON z stderr FFmpeg
    m = re.search(r"\{\s*\"input_i\".*?\}", res1.stderr, flags=re.DOTALL)
    if m:
        stats = json.loads(m.group(0))
        loudnorm_pass2 = (
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
            f"measured_I={stats['input_i']}:"
            f"measured_LRA={stats['input_lra']}:"
            f"measured_TP={stats['input_tp']}:"
            f"measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:linear=true"
        )
    else:
        loudnorm_pass2 = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"

    # Przebieg 2: właściwa normalizacja + zapis PCM WAV 24 kHz mono
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_wav),
        "-af",
        f"{pre_filter},{loudnorm_pass2}",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(mastered_wav),
    ]
    _run_ffmpeg(cmd2)


def export_audiobook(
    db: ProjectDB,
    export_mp3: bool = True,
    export_m4b: bool = True,
    output_root: Path = OUTPUT_DIR,
    mp3_bitrate: str = "96k",
    m4b_bitrate: str = "64k",
    normalize_audio: bool = True,
    pause_scale: float = 1.0,
    log_cb: Callable[[str], None] | None = None,
) -> ExportResult:
    meta = db.get_all_meta()
    book_title = meta.get("title") or db.project_dir.name
    book_author = meta.get("author") or "Nieznany autor"
    cover_path = Path(meta["cover_path"]) if meta.get("cover_path") else None
    if cover_path and not cover_path.exists():
        cover_path = None

    book_out_dir = output_root / _safe_filename(book_title)
    book_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = db.project_dir / "_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chapters = [ch for ch in db.list_chapters() if ch.selected]
    if not chapters:
        raise ValueError("Brak zaznaczonych rozdziałów do eksportu.")

    chapter_wavs: list[tuple[str, Path, float]] = []  # (tytuł, mastered_wav, czas_s)
    mp3_files: list[Path] = []
    sr = 24000

    for idx, ch in enumerate(chapters, start=1):
        chunks = [c for c in db.list_chunks(chapter_id=ch.id) if c.status == "done" and c.audio_path]
        if not chunks:
            if log_cb:
                log_cb(f"[Pominięto] Rozdział '{ch.title}' nie ma jeszcze wygenerowanych fragmentów.")
            continue

        if log_cb:
            log_cb(f"Łączenie rozdziału {idx}/{len(chapters)}: {ch.title} ({len(chunks)} fragmentów)...")

        parts: list[np.ndarray] = []
        for c_i, c in enumerate(chunks):
            p = db.resolve_chunk_audio_path(c)
            if p is None or not p.exists():
                continue
            wav_np, file_sr = sf.read(str(p), dtype="float32")
            sr = file_sr
            if wav_np.ndim == 2:
                wav_np = wav_np.mean(axis=1)
            parts.append(wav_np)
            if c_i < len(chunks) - 1 and c.pause_after_ms > 0:
                scaled_ms = max(0.0, c.pause_after_ms * pause_scale)
                parts.append(np.zeros(int(sr * scaled_ms / 1000.0), dtype=np.float32))

        if not parts:
            continue

        # Mała pauza końcowa na końcu rozdziału (1.2 s)
        parts.append(np.zeros(int(sr * 1.2), dtype=np.float32))
        combined = np.concatenate(parts)

        raw_wav = tmp_dir / f"ch_{idx:03d}_raw.wav"
        mastered_wav = tmp_dir / f"ch_{idx:03d}_master.wav"
        sf.write(str(raw_wav), combined, sr)

        if normalize_audio:
            if log_cb:
                log_cb(f"Mastering EBU R128 (-16 LUFS): {ch.title}...")
            _master_wav_ebu_r128(raw_wav, mastered_wav)
        else:
            mastered_wav = raw_wav

        dur_sec = float(sf.info(str(mastered_wav)).duration)
        chapter_wavs.append((ch.title, mastered_wav, dur_sec))

        if export_mp3:
            mp3_name = f"{idx:02d} - {_safe_filename(ch.title)}.mp3"
            mp3_path = book_out_dir / mp3_name
            cmd = ["ffmpeg", "-y", "-i", str(mastered_wav)]
            if cover_path:
                cmd += ["-i", str(cover_path), "-map", "0:a", "-map", "1:v", "-c:v", "copy", "-id3v2_version", "3"]
            cmd += [
                "-c:a",
                "libmp3lame",
                "-b:a",
                mp3_bitrate,
                "-metadata",
                f"title={ch.title}",
                "-metadata",
                f"album={book_title}",
                "-metadata",
                f"artist={book_author}",
                "-metadata",
                f"album_artist={book_author}",
                "-metadata",
                f"track={idx}/{len(chapters)}",
                str(mp3_path),
            ]
            _run_ffmpeg(cmd)
            mp3_files.append(mp3_path)

    if not chapter_wavs:
        raise RuntimeError("Nie udało się złożyć żadnego rozdziału (brak wygenerowanych plików WAV).")

    total_dur = sum(d for _, _, d in chapter_wavs)
    m4b_file: Path | None = None

    if export_m4b:
        if log_cb:
            log_cb("Pakowanie kompletnego audiobooka M4B ze spisem rozdziałów i okładką...")

        # 1. Plik listy do concat demuxera
        concat_list = tmp_dir / "concat.txt"
        concat_lines = [
            "file '" + str(wav_p.resolve()).replace("'", "'\\''") + "'"
            for _, wav_p, _ in chapter_wavs
        ]
        concat_list.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        # 2. Plik metadanych rozdziałów FFMETADATA1
        meta_file = tmp_dir / "chapters.ffmeta"
        meta_lines = [
            ";FFMETADATA1",
            f"title={_escape_ffmetadata(book_title)}",
            f"album={_escape_ffmetadata(book_title)}",
            f"artist={_escape_ffmetadata(book_author)}",
            f"album_artist={_escape_ffmetadata(book_author)}",
            "genre=Audiobook",
            "",
        ]
        cursor_ms = 0
        for ch_title, _wav_p, dur_s in chapter_wavs:
            end_ms = cursor_ms + int(round(dur_s * 1000))
            meta_lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={cursor_ms}",
                f"END={end_ms}",
                f"title={_escape_ffmetadata(ch_title)}",
                "",
            ]
            cursor_ms = end_ms
        meta_file.write_text("\n".join(meta_lines), encoding="utf-8")

        m4b_file = book_out_dir / f"{_safe_filename(book_title)}.m4b"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-i",
            str(meta_file),
        ]
        if cover_path:
            cmd += [
                "-i",
                str(cover_path),
                "-map",
                "0:a",
                "-map",
                "2:v",
                "-map_metadata",
                "1",
                "-c:v",
                "copy",
                "-disposition:v",
                "attached_pic",
            ]
        else:
            cmd += ["-map", "0:a", "-map_metadata", "1"]

        cmd += [
            "-c:a",
            "aac",
            "-b:a",
            m4b_bitrate,
            "-movflags",
            "+faststart",
            str(m4b_file),
        ]
        _run_ffmpeg(cmd)

    # Sprzątanie plików tymczasowych eksportu
    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)

    if log_cb:
        log_cb(f"Eksport zakończony! Folder wyjściowy: {book_out_dir}")

    return ExportResult(
        output_dir=book_out_dir,
        mp3_files=mp3_files,
        m4b_file=m4b_file,
        total_duration_sec=total_dur,
    )
