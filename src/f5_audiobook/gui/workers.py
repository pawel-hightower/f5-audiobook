"""Wątki robocze QThread dla operacji GPU, Quality Guard i eksportu FFmpeg (nie blokują GUI)."""

from __future__ import annotations

import threading
from pathlib import Path

import soundfile as sf
from PySide6.QtCore import QThread, Signal

from f5_audiobook.audio.exporter import ExportResult, export_audiobook
from f5_audiobook.config import OUTPUT_DIR
from f5_audiobook.engine.f5_ipa import F5IpaEngine
from f5_audiobook.pipeline.batch_runner import BatchOptions, BatchProgress, BatchRunner
from f5_audiobook.pipeline.project_db import ProjectDB
from f5_audiobook.voices import VoiceProfile


class BatchWorker(QThread):
    progress = Signal(object)   # BatchProgress
    log = Signal(str)
    finished_ok = Signal(object)  # BatchProgress
    failed = Signal(str)

    def __init__(
        self,
        engine: F5IpaEngine,
        db: ProjectDB,
        voice: VoiceProfile,
        options: BatchOptions,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.db = db
        self.voice = voice
        self.options = options
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()

    def run(self) -> None:
        try:
            runner = BatchRunner(self.engine)
            res = runner.run(
                db=self.db,
                voice=self.voice,
                options=self.options,
                pause_event=self.pause_event,
                stop_event=self.stop_event,
                progress_cb=lambda p: self.progress.emit(p),
                log_cb=lambda msg: self.log.emit(msg),
            )
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class SingleTestWorker(QThread):
    finished_ok = Signal(str, float)  # wav_path, duration_sec
    failed = Signal(str)

    def __init__(
        self,
        engine: F5IpaEngine,
        voice: VoiceProfile,
        text: str,
        out_path: Path,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.voice = voice
        self.text = text
        self.out_path = out_path

    def run(self) -> None:
        try:
            res = self.engine.synthesize(
                text=self.text,
                ref_audio_path=self.voice.ref_audio_path,
                ref_text=self.voice.ref_text,
                speed=self.voice.speed,
                cfg_strength=self.voice.cfg_strength,
                nfe_step=self.voice.nfe_step,
                seed=self.voice.seed,
            )
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.out_path), res.wav, res.sample_rate)
            self.finished_ok.emit(str(self.out_path), res.duration_sec)
        except Exception as e:
            self.failed.emit(str(e))


class RegenerateChunkWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(object)  # ChunkRow
    failed = Signal(str)

    def __init__(
        self,
        engine: F5IpaEngine,
        db: ProjectDB,
        chunk_id: int,
        voice: VoiceProfile,
        options: BatchOptions,
        new_text: str | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.db = db
        self.chunk_id = chunk_id
        self.voice = voice
        self.options = options
        self.new_text = new_text

    def run(self) -> None:
        try:
            runner = BatchRunner(self.engine)
            updated = runner.regenerate_chunk(
                db=self.db,
                chunk_id=self.chunk_id,
                voice=self.voice,
                options=self.options,
                new_text=self.new_text,
                log_cb=lambda m: self.log.emit(m),
            )
            self.finished_ok.emit(updated)
        except Exception as e:
            self.failed.emit(str(e))


class ExportWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(object)  # ExportResult
    failed = Signal(str)

    def __init__(
        self,
        db: ProjectDB,
        export_mp3: bool,
        export_m4b: bool,
        normalize_audio: bool,
        bitrate: str = "64k",
        pause_scale: float = 1.0,
        output_root: Path = OUTPUT_DIR,
    ) -> None:
        super().__init__()
        self.db = db
        self.export_mp3 = export_mp3
        self.export_m4b = export_m4b
        self.normalize_audio = normalize_audio
        self.bitrate = bitrate
        self.pause_scale = pause_scale
        self.output_root = output_root

    def run(self) -> None:
        try:
            res: ExportResult = export_audiobook(
                db=self.db,
                export_mp3=self.export_mp3,
                export_m4b=self.export_m4b,
                mp3_bitrate=self.bitrate,
                m4b_bitrate=self.bitrate,
                normalize_audio=self.normalize_audio,
                pause_scale=self.pause_scale,
                output_root=self.output_root,
                log_cb=lambda m: self.log.emit(m),
            )
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e))
