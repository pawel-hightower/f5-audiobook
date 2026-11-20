"""Test wielojęzycznego modelu Gregniuki/multi3 (F5-TTS + pełny polski fonetyzator IPA przez espeak-ng)."""

import argparse
import re
import time
from pathlib import Path

import espeakng_loader
import numpy as np
import soundfile as sf
import torch
import torchaudio
from f5_tts.api import F5TTS
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from tokenizers import Tokenizer

MODEL_DIR = Path("models/gregniuki-f5tts-multi3/multi3")
VOCODER_DIR = Path("models/vocos-mel-24khz")
DEFAULT_REF_AUDIO = Path("voices/moj_lektor/ref.wav")
DEFAULT_REF_TEXT = (
    "Smród panujący w chlewni zdawał się mieć niemal płynną formę. "
    "Jerzy starał się oddychać przez usta."
)
DEFAULT_SENTENCES = [
    "Wieczór zapadał nad miastem powoli, a chłodny wiatr niósł zapach zbliżającego się deszczu.",
    "Zatrzymał się na skraju pustej ulicy, nasłuchując odległego echa kroków i zastanawiając się, czy powinien jeszcze zawrócić.",
]


def init_espeak() -> None:
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def text_to_ipa(text: str, language: str = "pl") -> str:
    ipa = phonemize(
        text,
        language=language,
        backend="espeak",
        strip=False,
        preserve_punctuation=True,
        with_stress=True,
    )
    ipa = re.sub(r"\([a-z]{2,3}\)", "", ipa)
    return ipa.strip()


def load_ref_audio(ref_path: Path, target_sr: int, target_rms: float, device: str):
    wav_np, sr = sf.read(str(ref_path), dtype="float32")
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    audio = torch.from_numpy(wav_np).unsqueeze(0)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    return audio.to(device), float(rms.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gregniuki/multi3 (IPA) dla języka polskiego.")
    parser.add_argument("--ckpt", type=Path, default=MODEL_DIR / "model_1100000.pt")
    parser.add_argument("--vocab", type=Path, default=MODEL_DIR / "vocab.txt")
    parser.add_argument("--tokenizer", type=Path, default=MODEL_DIR / "tokenizer.json")
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", type=str, default=DEFAULT_REF_TEXT)
    parser.add_argument("--text", type=str, default=None, help="Opcjonalny własny tekst (zamiast domyślnego akapitu)")
    parser.add_argument("--output", type=Path, default=Path("output/f5_pl_fix/06_gregniuki_multi3_ipa.wav"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    init_espeak()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Ładowanie Gregniuki/multi3 IPA ({device}, dtype={args.dtype}) ===")
    t0 = time.perf_counter()
    tts = F5TTS(
        model="F5TTS_Base",
        ckpt_file=str(args.ckpt),
        vocab_file=str(args.vocab),
        vocoder_local_path=str(VOCODER_DIR),
        device=device,
    )
    if args.dtype == "bf16":
        # Reload cleanly in bfloat16 from fp32 checkpoint so fp16 overflow cannot corrupt weights
        ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=True)
        sd = {k.replace("ema_model.", ""): v for k, v in ckpt["ema_model_state_dict"].items() if k not in ["initted", "step"]}
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            sd.pop(key, None)
        tts.ema_model.to(torch.bfloat16)
        tts.ema_model.load_state_dict(sd)
        tts.ema_model.to(device).eval()
    elif args.dtype == "fp32":
        ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=True)
        sd = {k.replace("ema_model.", ""): v for k, v in ckpt["ema_model_state_dict"].items() if k not in ["initted", "step"]}
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            sd.pop(key, None)
        tts.ema_model.to(torch.float32)
        tts.ema_model.load_state_dict(sd)
        tts.ema_model.to(device).eval()
    print(f"Załadowano w {time.perf_counter() - t0:.2f}s (model dtype={next(tts.ema_model.parameters()).dtype})")

    target_sr = tts.target_sample_rate
    hop_length = 256
    target_rms = 0.1
    audio, rms = load_ref_audio(args.ref_audio, target_sr, target_rms, device)
    ref_audio_len = audio.shape[-1] // hop_length

    ipa_ref = text_to_ipa(args.ref_text, language="pl")
    print(f"\nRef tekst: {args.ref_text}")
    print(f"Ref IPA:   {ipa_ref}\n")

    sentences = [args.text] if args.text else DEFAULT_SENTENCES
    waves = []
    t1 = time.perf_counter()

    for i, sent in enumerate(sentences):
        ipa_gen = text_to_ipa(sent, language="pl")
        full_ipa = ipa_ref + " " + ipa_gen
        enc = tokenizer.encode(full_ipa)

        print(f"Zdanie {i + 1}: {sent}")
        print(f"IPA {i + 1}:    {ipa_gen} ({len(enc.ids)} tokenów)")

        # Czas trwania proporcjonalny do długości tekstu referencyjnego i docelowego (jak w F5-TTS)
        ref_len_b = len(args.ref_text.encode("utf-8"))
        gen_len_b = len(sent.encode("utf-8"))
        duration = max(250, ref_audio_len + int((ref_audio_len / ref_len_b) * gen_len_b / args.speed))

        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed + i)

        text_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

        with torch.inference_mode():
            generated, _ = tts.ema_model.sample(
                cond=audio,
                text=text_ids,
                duration=duration,
                steps=args.nfe_step,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=-1.0,
            )
            generated = generated.to(torch.float32)[:, ref_audio_len:, :].permute(0, 2, 1)
            wav = tts.vocoder.decode(generated)
            if rms < target_rms:
                wav = wav * rms / target_rms
            wav_np = wav.squeeze().cpu().numpy()

        nan_cnt = int(np.isnan(wav_np).sum())
        rms_out = float(np.sqrt(np.mean(wav_np**2))) if nan_cnt == 0 else float("nan")
        peak_out = float(np.abs(wav_np).max()) if nan_cnt == 0 else float("nan")
        print(f"   -> audio: {len(wav_np) / target_sr:.2f}s | rms={rms_out:.4f} | peak={peak_out:.4f} | nans={nan_cnt}")
        if nan_cnt > 0:
            raise RuntimeError(f"Wykryto {nan_cnt} wartości NaN w wygenerowanym audio!")

        waves.append(wav_np)
        if i < len(sentences) - 1:
            waves.append(np.zeros(int(target_sr * 0.35), dtype=np.float32))

    full_wav = np.concatenate(waves)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), full_wav, target_sr)

    dur = len(full_wav) / target_sr
    gen_t = time.perf_counter() - t1
    print(f"\n[OK] {args.output.resolve()} | {dur:.2f}s audio | gen: {gen_t:.1f}s (RTF: {gen_t / dur:.2f})")


if __name__ == "__main__":
    main()
