"""Silnik wsadowy (BatchRunner) z obsługą Pauzy/Wznów, Quality Guard i regeneracji pojedynczych zdań."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable

import soundfile as sf

from f5_audiobook.engine.f5_ipa import F5IpaEngine, SynthesisResult
from f5_audiobook.engine.quality_guard import QualityGuard, QualityReport
from f5_audiobook.pipeline.project_db import ChunkRow, ProjectDB
from f5_audiobook.voices import VoiceProfile

SHORT_CHUNK_MAX_CHARS = 22
SHORT_CHUNK_CARRIER = "Zatrzymał się na moment."


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
    qg_device: str = "auto"  # "auto" (GPU int8_float16 z fallbackiem na CPU) | "cpu"
    gpu_batch_size: int = 3  # 1 = sekwencyjnie, 3 = złoty środek na RTX 4050 (2.4x szybsze F5-TTS)
    cooldown_sec: float = 0.0  # opcjonalna przerwa między chunkami (ochrona termiczna laptopa)


def build_length_buckets(
    chunks: list[ChunkRow],
    voice_for_chunk: Callable[[ChunkRow], VoiceProfile],
    batch_size: int,
    max_len_ratio: float = 1.05,
) -> list[tuple[VoiceProfile, list[ChunkRow]]]:
    """Grupuje oczekujące chunki według profilu lektora i niemal identycznej długości tekstu do GPU batchingu.

    - Bardzo krótkie frazy (< SHORT_CHUNK_MAX_CHARS) idą zawsze pojedynczo (batch=1), aby zachować ratunek nośnikiem.
    - Wewnątrz każdego lektora sortuje po długości tekstu i łączy TYLKO „bliźniaki / trojaczki” o różnicy <= 5% (+2 zn.),
      ponieważ większy padding w paczce F5-TTS potrafi zniekształcić krótsze zdania.
    - Zdania bez odpowiedniego bliźniaka idą automatycznie osobno (batch=1).
    """
    if batch_size <= 1 or len(chunks) <= 1:
        return [(voice_for_chunk(c), [c]) for c in chunks]

    by_voice: dict[str, tuple[VoiceProfile, list[ChunkRow]]] = {}
    singles: list[tuple[VoiceProfile, list[ChunkRow]]] = []

    for c in chunks:
        v = voice_for_chunk(c)
        if len(c.normalized_text.strip()) < SHORT_CHUNK_MAX_CHARS:
            singles.append((v, [c]))
            continue
        if v.name not in by_voice:
            by_voice[v.name] = (v, [])
        by_voice[v.name][1].append(c)

    batches: list[tuple[VoiceProfile, list[ChunkRow]]] = []
    for _vname, (voice, v_chunks) in by_voice.items():
        # Sortujemy po długości tekstu (przy równych długościach zachowujemy kolejność książkową)
        v_chunks.sort(key=lambda c: (len(c.normalized_text.strip()), c.chapter_index, c.chunk_index))
        cur_batch: list[ChunkRow] = []
        min_len = 0
        for c in v_chunks:
            clen = max(1, len(c.normalized_text.strip()))
            if not cur_batch:
                cur_batch = [c]
                min_len = clen
            elif len(cur_batch) < batch_size and clen <= int(min_len * max_len_ratio) + 2:
                cur_batch.append(c)
            else:
                batches.append((voice, cur_batch))
                cur_batch = [c]
                min_len = clen
        if cur_batch:
            batches.append((voice, cur_batch))

    batches.extend(singles)
    return batches


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
                device=options.qg_device,
            )

        is_short = len(text.strip()) < SHORT_CHUNK_MAX_CHARS

        for attempt in range(attempts):
            seed = base_seed if attempt == 0 else random.randint(1000, 9_999_999)
            # Najpierw ZAWSZE synteza bezpośrednia (naturalna intonacja opadająca bez ryzyka ucięcia).
            # Zdanie-nośnik + cięcie w ciszy uruchamiamy WYŁĄCZNIE jako ratunek przy ponowieniu,
            # gdy synteza bezpośrednia faktycznie oblała Quality Guard (np. "Silos." -> "Silusz?").
            use_carrier = (
                is_short
                and attempt > 0
                and attempt == attempts - 1
                and options.use_quality_guard
                and self.quality_guard is not None
            )
            if use_carrier:
                clean_target = text.strip()
                if clean_target and clean_target[-1] not in ".!?…":
                    clean_target += "."
                carrier_prompt = f"{SHORT_CHUNK_CARRIER} {clean_target}"
                raw_res = self.engine.synthesize(
                    text=carrier_prompt,
                    ref_audio_path=voice.ref_audio_path,
                    ref_text=voice.ref_text,
                    speed=voice.speed,
                    cfg_strength=voice.cfg_strength,
                    nfe_step=voice.nfe_step,
                    seed=seed,
                )
                assert self.quality_guard is not None
                sliced_wav, cut_note = self.quality_guard.slice_after_carrier(
                    raw_res.wav, raw_res.sample_rate, SHORT_CHUNK_CARRIER, clean_target
                )
                if sliced_wav is not None:
                    import numpy as np

                    res = SynthesisResult(
                        wav=sliced_wav,
                        sample_rate=raw_res.sample_rate,
                        duration_sec=len(sliced_wav) / raw_res.sample_rate,
                        rms=float(np.sqrt(np.mean(sliced_wav**2))),
                        seed=raw_res.seed,
                        ipa_text=raw_res.ipa_text,
                    )
                    if log_cb:
                        log_cb(f"[Krótka fraza: ratunek nośnikiem] {text.strip()!r} ({cut_note})")
                else:
                    res = self.engine.synthesize(
                        text=text,
                        ref_audio_path=voice.ref_audio_path,
                        ref_text=voice.ref_text,
                        speed=voice.speed,
                        cfg_strength=voice.cfg_strength,
                        nfe_step=voice.nfe_step,
                        seed=seed,
                    )
            else:
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

    @staticmethod
    def _resolve_voice_for_chunk(
        chunk: ChunkRow,
        default_voice: VoiceProfile,
        voice_map: dict[str, VoiceProfile] | None,
    ) -> VoiceProfile:
        if not voice_map:
            return default_voice
        role = chunk.voice_role or "narrator"
        key = "male" if role == "dialogue" else role
        chosen = voice_map.get(key) or voice_map.get("narrator") or default_voice
        if not chosen.ref_audio_path.exists() or not chosen.ref_text.strip():
            return default_voice
        return chosen

    def regenerate_chunk(
        self,
        db: ProjectDB,
        chunk_id: int,
        voice: VoiceProfile,
        options: BatchOptions,
        new_text: str | None = None,
        new_seed: int | None = None,
        log_cb: Callable[[str], None] | None = None,
        voice_map: dict[str, VoiceProfile] | None = None,
    ) -> ChunkRow:
        if new_text is not None:
            db.update_chunk_text(chunk_id, new_text)
        chunk = db.get_chunk(chunk_id)
        chunk_voice = self._resolve_voice_for_chunk(chunk, voice, voice_map)
        seed = new_seed if new_seed is not None else random.randint(1000, 9_999_999)

        res, rep = self._synthesize_with_qg(
            text=chunk.normalized_text,
            voice=chunk_voice,
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
        voice_map: dict[str, VoiceProfile] | None = None,
    ) -> BatchProgress:
        db.set_meta("voice_name", voice.name)
        if log_cb:
            log_cb("Ładowanie silnika F5-TTS IPA (Gregniuki/multi3)...")
        import torch

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
        multi_active = bool(voice_map and len({v.name for v in voice_map.values() if v}) > 1)
        role_badges = {"narrator": "N", "male": "M", "female": "K", "dialogue": "M?"}

        if options.use_quality_guard and self.quality_guard is None:
            self.quality_guard = QualityGuard(
                model_size=options.qg_whisper_model,
                min_similarity=options.qg_min_similarity,
                device=options.qg_device,
            )
            if self.quality_guard._ensure_loaded() and log_cb:
                dev_lbl = "GPU (int8_float16)" if self.quality_guard.active_device == "cuda" else "CPU (int8)"
                log_cb(f"Quality Guard gotowy: faster-whisper {options.qg_whisper_model} na {dev_lbl}.")

        use_gpu_batch = options.gpu_batch_size > 1 and self.engine.device == "cuda" and len(pending) > 1
        eff_batch_size = max(1, int(options.gpu_batch_size)) if use_gpu_batch else 1

        batches = build_length_buckets(
            pending,
            voice_for_chunk=lambda c: self._resolve_voice_for_chunk(c, voice, voice_map),
            batch_size=eff_batch_size,
        )
        if use_gpu_batch and log_cb:
            paired = sum(len(b) for _, b in batches if len(b) > 1)
            log_cb(
                f"⚡ Tryb turbo: pogrupowano {len(pending)} fragm. w {len(batches)} paczek "
                f"(batch={eff_batch_size}, w paczkach: {paired}/{len(pending)})."
            )

        def _emit_chunk_done(chunk: ChunkRow, elapsed_per_chunk: float, msg: str) -> int:
            recent_times.append(elapsed_per_chunk)
            if len(recent_times) > 20:
                recent_times.pop(0)
            d_cnt, e_cnt, t_cnt = db.get_progress()
            rem = max(0, t_cnt - d_cnt - e_cnt)
            eta = (sum(recent_times) / len(recent_times)) * rem if recent_times else None
            if log_cb:
                log_cb(msg)
            if progress_cb:
                progress_cb(
                    BatchProgress(
                        done=d_cnt,
                        total=t_cnt,
                        errors=e_cnt,
                        current_chunk=chunk,
                        eta_sec=eta,
                        vram_mb=self._current_vram_mb(),
                        message=msg,
                    )
                )
            return rem

        try:
            for batch_voice, batch_chunks in batches:
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

                if len(batch_chunks) == 1:
                    chunk = batch_chunks[0]
                    t0 = time.perf_counter()
                    try:
                        res, rep = self._synthesize_with_qg(
                            text=chunk.normalized_text,
                            voice=batch_voice,
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
                        v_tag = f"[{role_badges.get(chunk.voice_role, 'N')}] " if multi_active else ""
                        msg = (
                            f"{v_tag}[{chunk.chapter_title}] #{chunk.chunk_index + 1}: "
                            f"{res.duration_sec:.1f}s"
                            + (f" | QG: {rep.reason}" if rep else "")
                        )
                    except Exception as e:
                        db.mark_chunk_error(chunk.id, str(e))
                        msg = f"[BŁĄD] [{chunk.chapter_title}] #{chunk.chunk_index + 1}: {e}"

                    remaining = _emit_chunk_done(chunk, time.perf_counter() - t0, msg)
                    if options.cooldown_sec > 0 and remaining > 0:
                        time.sleep(options.cooldown_sec)
                    continue

                # Paczka >= 2 zdań tego samego lektora o zbliżonej długości -> 1 przejście GPU!
                t_batch = time.perf_counter()
                batch_results: list[SynthesisResult] | None = None
                try:
                    batch_results = self.engine.synthesize_batch(
                        texts=[c.normalized_text for c in batch_chunks],
                        ref_audio_path=batch_voice.ref_audio_path,
                        ref_text=batch_voice.ref_text,
                        speed=batch_voice.speed,
                        cfg_strength=batch_voice.cfg_strength,
                        nfe_step=batch_voice.nfe_step,
                        seeds=[c.seed for c in batch_chunks],
                    )
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if log_cb:
                        log_cb(f"[VRAM] Brak pamięci dla paczki {len(batch_chunks)} zdań → przełączam te zdania na batch=1.")
                    batch_results = None
                except Exception as e:
                    if log_cb:
                        log_cb(f"[Batch fallback] {e} → liczę te zdania pojedynczo.")
                    batch_results = None

                synth_per_item = (time.perf_counter() - t_batch) / len(batch_chunks)

                if batch_results is None:
                    # Bezpieczny fallback 1-po-1 dla tej paczki
                    for chunk in batch_chunks:
                        if stop_event and stop_event.is_set():
                            break
                        t0 = time.perf_counter()
                        try:
                            res, rep = self._synthesize_with_qg(
                                text=chunk.normalized_text,
                                voice=batch_voice,
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
                            v_tag = f"[{role_badges.get(chunk.voice_role, 'N')}] " if multi_active else ""
                            msg = (
                                f"{v_tag}[{chunk.chapter_title}] #{chunk.chunk_index + 1}: "
                                f"{res.duration_sec:.1f}s"
                                + (f" | QG: {rep.reason}" if rep else "")
                            )
                        except Exception as e:
                            db.mark_chunk_error(chunk.id, str(e))
                            msg = f"[BŁĄD] [{chunk.chapter_title}] #{chunk.chunk_index + 1}: {e}"
                        _emit_chunk_done(chunk, time.perf_counter() - t0, msg)
                    continue

                # Weryfikacja Quality Guard dla każdego zdania z paczki (tylko oblane dostają pojedynczy retry!)
                remaining = 0
                for chunk, res in zip(batch_chunks, batch_results):
                    t_item = time.perf_counter()
                    try:
                        rep: QualityReport | None = None
                        if options.use_quality_guard and self.quality_guard is not None:
                            rep = self.quality_guard.check(chunk.normalized_text, res.wav, res.sample_rate)
                            if not rep.passed and options.qg_max_retries > 0:
                                if log_cb:
                                    log_cb(
                                        f"[Quality Guard] [{chunk.chapter_title}] #{chunk.chunk_index + 1} "
                                        f"odrzucono w paczce ({rep.reason}) -> ponawiam pojedynczo..."
                                    )
                                retry_res, retry_rep = self._synthesize_with_qg(
                                    text=chunk.normalized_text,
                                    voice=batch_voice,
                                    base_seed=random.randint(1000, 9_999_999),
                                    options=BatchOptions(
                                        use_quality_guard=True,
                                        qg_max_retries=max(0, options.qg_max_retries - 1),
                                        qg_min_similarity=options.qg_min_similarity,
                                        qg_whisper_model=options.qg_whisper_model,
                                        qg_device=options.qg_device,
                                        gpu_batch_size=1,
                                    ),
                                    log_cb=log_cb,
                                )
                                if retry_rep is None or retry_rep.score >= rep.score:
                                    res = retry_res
                                    rep = retry_rep

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
                        v_tag = f"[{role_badges.get(chunk.voice_role, 'N')}] " if multi_active else ""
                        msg = (
                            f"⚡{v_tag}[{chunk.chapter_title}] #{chunk.chunk_index + 1}: "
                            f"{res.duration_sec:.1f}s"
                            + (f" | QG: {rep.reason}" if rep else "")
                        )
                    except Exception as e:
                        db.mark_chunk_error(chunk.id, str(e))
                        msg = f"[BŁĄD] [{chunk.chapter_title}] #{chunk.chunk_index + 1}: {e}"

                    item_elapsed = synth_per_item + (time.perf_counter() - t_item)
                    remaining = _emit_chunk_done(chunk, item_elapsed, msg)

                if options.cooldown_sec > 0 and remaining > 0:
                    time.sleep(options.cooldown_sec)
        finally:
            if self.quality_guard is not None:
                self.quality_guard.unload()

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
        import sys

        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and torch_mod.cuda.is_available():
            return torch_mod.cuda.memory_allocated() / (1024**2)
        return 0.0
