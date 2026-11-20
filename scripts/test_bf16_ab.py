"""Test A/B: fp32 vs bfloat16 autocast na RTX 4050 (ten sam tekst, ten sam seed, ODE=32, nowe naturalne pauzy 600ms)."""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from f5_audiobook.engine.f5_ipa import F5IpaEngine
from f5_audiobook.voices import load_voice_profile

OUT_DIR = Path("output/ab_bf16_test")

# Realny fragment z dłuższymi zdaniami i naturalnymi pauzami (600 ms między zdaniami)
TEST_CHUNKS = [
    ("Wanda przypinała już do stelaża dwie nowe butle z tlenem, ale przerwała tę czynność i wyjrzała przez okno bazy nurków.", 0.60),
    ("W tej chwili powierzchnią morza targał jedynie ostry południowo-wschodni wiatr, więc mocno lśniło w porannym słońcu.", 0.60),
    ("Zobaczyła też w oddali krótki błysk wyładowania elektrycznego.", 0.0),
]


def run_pass(engine: F5IpaEngine, voice, use_bf16: bool, out_path: Path) -> tuple[float, float, float]:
    waves = []
    sr = engine.sample_rate
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i, (text, pause_s) in enumerate(TEST_CHUNKS):
        # W trybie bf16 autocast owijamy forward transformera (kroki ODE i Vocos zostają bezpieczne w fp32)
        if use_bf16 and engine.device == "cuda":
            orig_forward = engine._tts.ema_model.transformer.forward

            def _bf16_forward(*args, _orig=orig_forward, **kwargs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return _orig(*args, **kwargs)

            engine._tts.ema_model.transformer.forward = _bf16_forward
            try:
                res = engine.synthesize(
                    text=text,
                    ref_audio_path=voice.ref_audio_path,
                    ref_text=voice.ref_text,
                    speed=voice.speed,
                    cfg_strength=voice.cfg_strength,
                    nfe_step=voice.nfe_step,
                    seed=voice.seed + i,
                )
            finally:
                engine._tts.ema_model.transformer.forward = orig_forward
        else:
            res = engine.synthesize(
                text=text,
                ref_audio_path=voice.ref_audio_path,
                ref_text=voice.ref_text,
                speed=voice.speed,
                cfg_strength=voice.cfg_strength,
                nfe_step=voice.nfe_step,
                seed=voice.seed + i,
            )

        waves.append(res.wav)
        if pause_s > 0:
            waves.append(np.zeros(int(sr * pause_s), dtype=np.float32))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_t = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    full = np.concatenate(waves)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full, sr)
    dur = len(full) / sr
    return gen_t, dur, peak_vram


def main() -> None:
    voice = load_voice_profile("moj_lektor")
    engine = F5IpaEngine()
    engine.load()

    # Krótka rozgrzewka CUDA przed pomiarem czasu
    _ = engine.synthesize("Test.", voice.ref_audio_path, voice.ref_text, nfe_step=8, seed=1)

    t_fp32, dur_fp32, vram_fp32 = run_pass(engine, voice, use_bf16=False, out_path=OUT_DIR / "A_fp32.wav")
    print(f"[A - fp32]         gen={t_fp32:.2f}s | audio={dur_fp32:.2f}s | RTF={t_fp32/dur_fp32:.2f} | VRAM={vram_fp32:.0f} MiB")

    t_bf16, dur_bf16, vram_bf16 = run_pass(engine, voice, use_bf16=True, out_path=OUT_DIR / "B_bf16_autocast.wav")
    print(f"[B - bf16 autocast] gen={t_bf16:.2f}s | audio={dur_bf16:.2f}s | RTF={t_bf16/dur_bf16:.2f} | VRAM={vram_bf16:.0f} MiB")
    print(f"Przyspieszenie: {t_fp32 / t_bf16:.2f}x szybciej!")


if __name__ == "__main__":
    main()
