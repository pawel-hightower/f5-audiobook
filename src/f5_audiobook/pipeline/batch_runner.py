"""Silnik wsadowy (BatchRunner) z obsługą Pauzy/Wznów, Quality Guard i regeneracji pojedynczych zdań."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable

import soundfile as sf
import torch

from f5_audiobook.engine.f5_ipa import F5IpaEngine, SynthesisResult
from f5_audiobook.engine.quality_guard import QualityGuard, QualityReport
from f5_audiobook.pipeline.project_db import ChunkRow, ProjectDB
from f5_audiobook.voices import VoiceProfile


@dataclass
class BatchProgress:
    done: int
    total: int
    errors: int
    current_chunk: ChunkRow | None
    eta_sec: float | None
    vram_mb: float
    message: str


@dataclass
class BatchOptions:
    use_quality_guard: bool = True
    qg_max_retries: int = 2
    qg_min_similarity: float = 0.78
    qg_whisper_model: str = "small"
    cooldown_sec: float = 0.0  # opcjonalna przerwa między chunkami (ochrona termiczna laptopa)


class BatchRunner:
    def __init__(
        self,
        engine: F5IpaEngine,
        quality_guard: QualityGuard | None = None,
    ) -> None:
        self.engine = engine
        self.quality_guard = quality_guard

    def _synthesize_with_qg(
        self,
        text: str,
        voice: VoiceProfile,
        base_seed: int,
        options: BatchOptions,
        log_cb: Callable[[str], None] | None = None,
    ) -> tuple[SynthesisResult, QualityReport | None]:
        attempts = 1 + (options.qg_max_retries if options.use_quality_guard else 0)
        best_res: SynthesisResult | None = None
        best_rep: QualityReport | None = None

        if options.use_quality_guard and self.quality_guard is None:
            self.quality_guard = QualityGuard(
                model_size=options.qg_whisper_model,
                min_similarity=options.qg_min_similarity,
            )

        for attempt in range(attempts):
            seed = base_seed if attempt == 0 else random.randint(1000, 9_999_999)
            res = self.engine.synthesize(
                text=text,
                ref_audio_path=voice.ref_audio_path,
                ref_text=voice.ref_text,
                speed=voice.speed,
                cfg_strength=voice.cfg_strength,
                nfe_step=voice.nfe_step,
                seed=seed,
            )

            if not options.use_quality_guard or self.quality_guard is None:
                return res, None

            rep = self.quality_guard.check(text, res.wav, res.sample_rate)
            if best_rep is None or rep.score > best_rep.score:
                best_res = res
                best_rep = rep

            if rep.passed:
                return res, rep

            if log_cb and attempt < attempts - 1:
                log_cb(
                    f"[Quality Guard] Próba {attempt + 1}/{attempts} odrzucona ({rep.reason}) "
                    f"-> ponawiam z nowym seedem..."
                )

        assert best_res is not None
        return best_res, best_rep

    def regenerate_chunk(
        self,
        db: ProjectDB,
        chunk_id: int,
        voice: VoiceProfile,
        options: BatchOptions,
        new_text: str | None = None,
        new_seed: int | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> ChunkRow:
        if new_text is not None:
            db.update_chunk_text(chunk_id, new_text)
        chunk = db.get_chunk(chunk_id)
        seed = new_seed if new_seed is not None else random.randint(1000, 9_999_999)

        res, rep = self._synthesize_with_qg(
            text=chunk.normalized_text,
            voice=voice,
            base_seed=seed,
            options=options,
            log_cb=log_cb,
        )
        out_path = db.chunk_audio_path(chunk)
        sf.write(str(out_path), res.wav, res.sample_rate)

        db.mark_chunk_done(
            chunk_id=chunk.id,
            audio_path=out_path,
            duration_sec=res.duration_sec,
            seed=res.seed,
            qg_score=rep.score if rep else None,
            qg_note=rep.reason if rep else "",
            heard_text=rep.heard_text if rep else "",
        )
        return db.get_chunk(chunk.id)

    def run(
        self,
        db: ProjectDB,
        voice: VoiceProfile,
        options: BatchOptions,
        pause_event: threading.Event | None = None,
        stop_event: threading.Event | None = None,
        progress_cb: Callable[[BatchProgress], None] | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> BatchProgress:
        db.set_meta("voice_name", voice.name)
        if log_cb:
            log_cb("Ładowanie silnika F5-TTS IPA (Gregniuki/multi3)...")
        self.engine.load()

        pending = db.get_pending_chunks()
        done, errors, total = db.get_progress()

        if not pending:
            prog = BatchProgress(
                done=done,
                total=total,
                errors=errors,
                current_chunk=None,
                eta_sec=0.0,
                vram_mb=self._current_vram_mb(),
                message="Wszystkie zaznaczone fragmenty zostały już wygenerowane.",
            )
            if progress_cb:
                progress_cb(prog)
            return prog

        recent_times: list[float] = []

        for chunk in pending:
            if stop_event and stop_event.is_set():
                if log_cb:
                    log_cb("Zatrzymano generowanie na żądanie użytkownika. Postęp został zapisany.")
                break

            while pause_event and pause_event.is_set():
                if stop_event and stop_event.is_set():
                    break
                time.sleep(0.2)

            if stop_event and stop_event.is_set():
                break

            t0 = time.perf_counter()
            try:
                res, rep = self._synthesize_with_qg(
                    text=chunk.normalized_text,
                    voice=voice,
                    base_seed=chunk.seed,
                    options=options,
                    log_cb=log_cb,
                )
                out_path = db.chunk_audio_path(chunk)
                sf.write(str(out_path), res.wav, res.sample_rate)

                db.mark_chunk_done(
                    chunk_id=chunk.id,
                    audio_path=out_path,
                    duration_sec=res.duration_sec,
                    seed=res.seed,
                    qg_score=rep.score if rep else None,
                    qg_note=rep.reason if rep else "",
                    heard_text=rep.heard_text if rep else "",
                )
                msg = (
                    f"[{chunk.chapter_title}] #{chunk.chunk_index + 1}: "
                    f"{res.duration_sec:.1f}s"
                    + (f" | QG: {rep.reason}" if rep else "")
                )
            except Exception as e:
                db.mark_chunk_error(chunk.id, str(e))
                msg = f"[BŁĄD] [{chunk.chapter_title}] #{chunk.chunk_index + 1}: {e}"

            elapsed = time.perf_counter() - t0
            recent_times.append(elapsed)
            if len(recent_times) > 15:
                recent_times.pop(0)

            done, errors, total = db.get_progress()
            remaining = max(0, total - done - errors)
            eta = (sum(recent_times) / len(recent_times)) * remaining if recent_times else None

            if log_cb:
                log_cb(msg)
            if progress_cb:
                progress_cb(
                    BatchProgress(
                        done=done,
                        total=total,
                        errors=errors,
                        current_chunk=chunk,
                        eta_sec=eta,
                        vram_mb=self._current_vram_mb(),
                        message=msg,
                    )
                )

            if options.cooldown_sec > 0 and remaining > 0:
                time.sleep(options.cooldown_sec)

        done, errors, total = db.get_progress()
        return BatchProgress(
            done=done,
            total=total,
            errors=errors,
            current_chunk=None,
            eta_sec=0.0,
            vram_mb=self._current_vram_mb(),
            message=f"Zakończono: {done}/{total} gotowych (błędów: {errors}).",
        )

    @staticmethod
    def _current_vram_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0
