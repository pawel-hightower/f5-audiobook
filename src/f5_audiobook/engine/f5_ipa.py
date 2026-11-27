"""Silnik F5-TTS (Gregniuki/multi3 IPA) z polskim fonetyzatorem espeak-ng i kontrolą pauz."""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass
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

from f5_audiobook.config import MULTI3_MODEL_DIR, VOCODER_DIR

_ESPEAK_INITIALIZED = False
_BPE_SPLIT_MARK = "\u2060"
_AFFRICATE_TXT_RE = re.compile(r"trz|drz|cz|dż|tsz|dsz", re.IGNORECASE)
_AFFRICATE_IPA_RE = re.compile(r"tʃ|dʒ")


def init_espeak() -> None:
    global _ESPEAK_INITIALIZED
    if _ESPEAK_INITIALIZED:
        return
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    _ESPEAK_INITIALIZED = True


def mark_polish_trz_drz(text: str, ipa: str) -> str:
    """Wstawia znacznik rozbicia BPE tylko w tʃ z 'trz' i dʒ z 'drz' (nie rusza 'cz'/'dż')."""
    if not re.search(r"trz|drz", text, re.IGNORECASE):
        return ipa

    src_matches = [m.group(0).lower() for m in _AFFRICATE_TXT_RE.finditer(text)]
    ipa_matches = list(_AFFRICATE_IPA_RE.finditer(ipa))
    if len(src_matches) == len(ipa_matches):
        out = ipa
        for kind, m in reversed(list(zip(src_matches, ipa_matches))):
            if kind in ("trz", "drz"):
                sym = m.group(0)
                out = out[: m.start()] + f"{sym[0]}{_BPE_SPLIT_MARK}{sym[1]}" + out[m.end() :]
        return out

    # Fallback: jeśli w zdaniu nie ma żadnych 'cz'/'dż', rozbij bezpiecznie wszystkie tʃ/dʒ
    out = ipa
    if re.search(r"trz", text, re.IGNORECASE) and not re.search(r"cz", text, re.IGNORECASE):
        out = out.replace("tʃ", f"t{_BPE_SPLIT_MARK}ʃ")
    if re.search(r"drz", text, re.IGNORECASE) and not re.search(r"dż", text, re.IGNORECASE):
        out = out.replace("dʒ", f"d{_BPE_SPLIT_MARK}ʒ")
    return out


def encode_marked_ipa(tokenizer: Tokenizer, marked_ipa: str) -> list[int]:
    """Koduje IPA do ID tokenów BPE, wymuszając granicę tokenu w miejscach _BPE_SPLIT_MARK."""
    if _BPE_SPLIT_MARK not in marked_ipa:
        return tokenizer.encode(marked_ipa).ids
    ids: list[int] = []
    for seg in marked_ipa.split(_BPE_SPLIT_MARK):
        if seg:
            ids.extend(tokenizer.encode(seg).ids)
    return ids


def text_to_ipa_pl(text: str, mark_clusters: bool = False) -> str:
    init_espeak()
    ipa = phonemize(
        text,
        language="pl",
        backend="espeak",
        strip=False,
        preserve_punctuation=True,
        with_stress=True,
    )
    # Usuwamy tagi przełączania języka np. (en), (pl) oraz znak przedłużenia 'ː'
    ipa = re.sub(r"\([a-z]{2,3}\)", "", ipa)
    ipa = ipa.replace("ː", "")
    # Usuwamy akcent poboczny 'ˌ' bezpośrednio po tʃ/dʒ (np. czyściciele: tʃˌɨ -> tʃɨ),
    # aby BPE zachowało pełny polski token sylabowy 'tʃɨ' zamiast gołego 'tʃ'
    ipa = re.sub(r"(tʃ|dʒ)ˌ", r"\1", ipa)
    # Zapobiega zlewaniu grupy końcowej '-tr' przed dźwięczną (np. "Wiatr znowu" -> espeak daje vʲˈadr zamiast vʲˈatr)
    if re.search(r"tr(?=\s)", text, re.IGNORECASE):
        ipa = re.sub(r"dr(?=\s)", "tr", ipa)
    ipa = re.sub(r"\s+", " ", ipa).strip()
    if mark_clusters:
        ipa = mark_polish_trz_drz(text, ipa)
    return ipa


def trim_silence(
    wav: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    keep_lead_ms: int = 30,
    keep_tail_ms: int = 55,
) -> np.ndarray:
    if len(wav) == 0:
        return wav
    thresh = 10.0 ** (threshold_db / 20.0)
    win = max(1, int(sr * 0.02))
    n_win = len(wav) // win
    if n_win == 0:
        return wav
    frames = wav[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    voiced = np.where(rms > thresh)[0]
    if len(voiced) == 0:
        return wav
    start = max(0, voiced[0] * win - int(sr * keep_lead_ms / 1000))
    end = min(len(wav), (voiced[-1] + 1) * win + int(sr * keep_tail_ms / 1000))
    return wav[start:end]


def compress_internal_silences(
    wav: np.ndarray,
    sr: int,
    max_silence_sec: float = 0.45,
    threshold_db: float = -40.0,
) -> np.ndarray:
    thresh = 10.0 ** (threshold_db / 20.0)
    win = max(1, int(sr * 0.02))
    n_win = len(wav) // win
    if n_win == 0:
        return wav
    frames = wav[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    is_silent = rms <= thresh

    max_silent_wins = max(1, int(max_silence_sec / 0.02))
    out_parts: list[np.ndarray] = []
    i = 0
    while i < n_win:
        if not is_silent[i]:
            j = i
            while j < n_win and not is_silent[j]:
                j += 1
            out_parts.append(wav[i * win : j * win])
            i = j
        else:
            j = i
            while j < n_win and is_silent[j]:
                j += 1
            keep = min(j - i, max_silent_wins)
            out_parts.append(wav[i * win : (i + keep) * win])
            i = j
    if n_win * win < len(wav):
        out_parts.append(wav[n_win * win :])
    return np.concatenate(out_parts) if out_parts else wav


@dataclass
class SynthesisResult:
    wav: np.ndarray
    sample_rate: int
    duration_sec: float
    rms: float
    seed: int
    ipa_text: str


class F5IpaEngine:
    """Wrapper na Gregniuki/multi3 IPA F5-TTS z cache referencji i bezpieczną precyzją fp32."""

    def __init__(
        self,
        model_dir: Path = MULTI3_MODEL_DIR,
        vocoder_dir: Path = VOCODER_DIR,
        device: str | None = None,
        use_bf16_autocast: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.vocoder_dir = Path(vocoder_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16_autocast = (
            use_bf16_autocast
            and self.device == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        self.target_rms = 0.1
        self.hop_length = 256

        self._tts: F5TTS | None = None
        self._tokenizer: Tokenizer | None = None
        # Cache wielu głosów jednocześnie (Narrator + Mężczyzna + Kobieta -> 0 ms narzutu przy zmianie głosu!)
        self._ref_cache: dict[tuple[str, str], tuple[torch.Tensor, float, str]] = {}
        self._ref_audio_tensor: torch.Tensor | None = None
        self._ref_rms: float = 0.1
        self._ref_ipa: str = ""
        self._chunks_since_gc: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._tts is not None

    @property
    def sample_rate(self) -> int:
        return self._tts.target_sample_rate if self._tts is not None else 24000

    def load(self) -> None:
        if self._tts is not None:
            return
        init_espeak()

        ckpt_path = self.model_dir / "model_1100000.pt"
        vocab_path = self.model_dir / "vocab.txt"
        tok_path = self.model_dir / "tokenizer.json"

        for p in (ckpt_path, vocab_path, tok_path):
            if not p.exists():
                raise FileNotFoundError(f"Brak wymaganego pliku modelu: {p}")

        self._tokenizer = Tokenizer.from_file(str(tok_path))
        vocoder_local = str(self.vocoder_dir) if (self.vocoder_dir / "pytorch_model.bin").exists() else None

        tts = F5TTS(
            model="F5TTS_Base",
            ckpt_file=str(ckpt_path),
            vocab_file=str(vocab_path),
            vocoder_local_path=vocoder_local,
            device=self.device,
        )

        # Ładujemy wagi EMA w pełnym float32 (zapobiega przepełnieniom NaN z float16 w multi3)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        sd = {
            k.replace("ema_model.", ""): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in ("initted", "step")
        }
        for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
            sd.pop(key, None)
        tts.ema_model.to(torch.float32)
        tts.ema_model.load_state_dict(sd)
        tts.ema_model.to(self.device).eval()

        # Przyspieszenie Tensor Cores (bfloat16) na transformerze przy zachowaniu kroków ODE i Vocos w fp32
        if self.use_bf16_autocast:
            orig_forward = tts.ema_model.transformer.forward

            def _bf16_forward(*args, _orig=orig_forward, **kwargs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return _orig(*args, **kwargs)

            tts.ema_model.transformer.forward = _bf16_forward

        del ckpt, sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._tts = tts

    def unload(self) -> None:
        self._tts = None
        self._ref_audio_tensor = None
        self._ref_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_reference(self, ref_audio_path: Path, ref_text: str) -> None:
        cache_key = (str(ref_audio_path.resolve()), ref_text.strip())
        cached = self._ref_cache.get(cache_key)
        if cached is not None:
            self._ref_audio_tensor, self._ref_rms, self._ref_ipa = cached
            return

        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Nie znaleziono próbki głosu: {ref_audio_path}")

        wav_np, sr = sf.read(str(ref_audio_path), dtype="float32")
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)

        wav_np = trim_silence(wav_np, sr, threshold_db=-38.0, keep_lead_ms=30, keep_tail_ms=80)
        wav_np = np.concatenate([wav_np, np.zeros(int(sr * 0.10), dtype=np.float32)])

        audio = torch.from_numpy(wav_np).unsqueeze(0)
        rms = float(torch.sqrt(torch.mean(torch.square(audio))).item())
        if rms > 0 and rms < self.target_rms:
            audio = audio * (self.target_rms / rms)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)

        audio_dev = audio.to(self.device)
        ref_ipa = text_to_ipa_pl(ref_text, mark_clusters=True)
        if len(self._ref_cache) >= 8:
            self._ref_cache.pop(next(iter(self._ref_cache)))
        self._ref_cache[cache_key] = (audio_dev, rms, ref_ipa)
        self._ref_audio_tensor = audio_dev
        self._ref_rms = rms
        self._ref_ipa = ref_ipa

    def synthesize(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        speed: float = 1.15,
        cfg_strength: float = 2.0,
        nfe_step: int = 32,
        seed: int = 101,
        max_internal_silence_sec: float = 0.45,
    ) -> SynthesisResult:
        self.load()
        assert self._tts is not None and self._tokenizer is not None

        self._prepare_reference(ref_audio_path, ref_text)
        assert self._ref_audio_tensor is not None

        ipa_gen_marked = text_to_ipa_pl(text, mark_clusters=True)
        ipa_gen = ipa_gen_marked.replace(_BPE_SPLIT_MARK, "")
        if not ipa_gen:
            raise ValueError(f"Pusta transkrypcja IPA dla tekstu: {text!r}")

        full_ipa_marked = f"{self._ref_ipa} {ipa_gen_marked}"
        token_ids = encode_marked_ipa(self._tokenizer, full_ipa_marked)

        ref_audio_len = self._ref_audio_tensor.shape[-1] // self.hop_length
        ref_ipa_len = max(1, len(self._ref_ipa.replace(_BPE_SPLIT_MARK, "")))
        gen_ipa_len = max(1, len(ipa_gen))

        # Bardzo krótkie frazy dostają lekki zapas czasu, żeby nie ucinało końcówek
        local_speed = speed * 0.78 if len(text.strip()) < 22 else speed
        duration = max(
            ref_audio_len + 65,
            ref_audio_len + int((ref_audio_len / ref_ipa_len) * gen_ipa_len / max(0.5, local_speed)),
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        text_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            generated, _ = self._tts.ema_model.sample(
                cond=self._ref_audio_tensor,
                text=text_ids,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=-1.0,
            )
            generated = generated.to(torch.float32)[:, ref_audio_len:, :].permute(0, 2, 1)
            wav = self._tts.vocoder.decode(generated)
            if self._ref_rms > 0 and self._ref_rms < self.target_rms:
                wav = wav * (self._ref_rms / self.target_rms)
            wav_np = wav.squeeze().cpu().numpy().astype(np.float32)

        if np.isnan(wav_np).any() or np.isinf(wav_np).any():
            raise RuntimeError("Wykryto wartości NaN/Inf w wygenerowanym sygnale audio.")

        sr = self.sample_rate
        wav_np = trim_silence(wav_np, sr, threshold_db=-40.0, keep_lead_ms=30, keep_tail_ms=55)
        wav_np = compress_internal_silences(wav_np, sr, max_silence_sec=max_internal_silence_sec)

        peak = float(np.abs(wav_np).max()) if len(wav_np) > 0 else 0.0
        if peak > 0.95:
            wav_np = wav_np * (0.95 / peak)

        # Miękki 10 ms fade-in / fade-out eliminujący kliki na złączach
        fade = min(int(sr * 0.01), len(wav_np) // 4)
        if fade > 0:
            wav_np[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wav_np[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        rms_out = float(np.sqrt(np.mean(wav_np**2))) if len(wav_np) > 0 else 0.0

        self._chunks_since_gc += 1
        if self._chunks_since_gc >= 15 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._chunks_since_gc = 0

        return SynthesisResult(
            wav=wav_np,
            sample_rate=sr,
            duration_sec=len(wav_np) / sr,
            rms=rms_out,
            seed=seed,
            ipa_text=ipa_gen,
        )

    def synthesize_batch(
        self,
        texts: list[str],
        ref_audio_path: Path,
        ref_text: str,
        speed: float = 1.15,
        cfg_strength: float = 2.0,
        nfe_step: int = 32,
        seeds: list[int] | None = None,
        max_internal_silence_sec: float = 0.45,
    ) -> list[SynthesisResult]:
        """Syntetyzuje paczkę (2–4) zdań tego samego lektora o zbliżonej długości w jednym przejściu GPU."""
        if not texts:
            return []
        eff_seeds = seeds if (seeds and len(seeds) == len(texts)) else [101 + i for i in range(len(texts))]
        if len(texts) == 1:
            return [
                self.synthesize(
                    text=texts[0],
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    speed=speed,
                    cfg_strength=cfg_strength,
                    nfe_step=nfe_step,
                    seed=eff_seeds[0],
                    max_internal_silence_sec=max_internal_silence_sec,
                )
            ]

        self.load()
        assert self._tts is not None and self._tokenizer is not None

        self._prepare_reference(ref_audio_path, ref_text)
        assert self._ref_audio_tensor is not None

        from torch.nn.utils.rnn import pad_sequence

        ref_audio_len = self._ref_audio_tensor.shape[-1] // self.hop_length
        ref_ipa_len = max(1, len(self._ref_ipa.replace(_BPE_SPLIT_MARK, "")))

        token_tensors: list[torch.Tensor] = []
        durations: list[int] = []
        ipa_gens: list[str] = []

        for txt in texts:
            ipa_gen_marked = text_to_ipa_pl(txt, mark_clusters=True)
            ipa_gen = ipa_gen_marked.replace(_BPE_SPLIT_MARK, "")
            if not ipa_gen:
                raise ValueError(f"Pusta transkrypcja IPA dla tekstu: {txt!r}")
            ipa_gens.append(ipa_gen)
            full_ipa_marked = f"{self._ref_ipa} {ipa_gen_marked}"
            token_ids = encode_marked_ipa(self._tokenizer, full_ipa_marked)
            token_tensors.append(torch.tensor(token_ids, dtype=torch.long, device=self.device))

            gen_ipa_len = max(1, len(ipa_gen))
            local_speed = speed * 0.78 if len(txt.strip()) < 22 else speed
            dur = max(
                ref_audio_len + 65,
                ref_audio_len + int((ref_audio_len / ref_ipa_len) * gen_ipa_len / max(0.5, local_speed)),
            )
            durations.append(dur)

        bs = len(texts)
        # Drugi bezpiecznik po przeliczeniu IPA: jeśli rzeczywista długość klatek audio (gen_dur)
        # w paczce różni się o > 8% (np. przez dwuznaki sz/cz/rz), liczymy te zdania bezpiecznie osobno!
        gen_durs = [max(1, d - ref_audio_len) for d in durations]
        if max(gen_durs) > int(min(gen_durs) * 1.08) + 6:
            return [
                self.synthesize(
                    text=t,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    speed=speed,
                    cfg_strength=cfg_strength,
                    nfe_step=nfe_step,
                    seed=s,
                    max_internal_silence_sec=max_internal_silence_sec,
                )
                for t, s in zip(texts, eff_seeds)
            ]

        text_batch = pad_sequence(token_tensors, batch_first=True, padding_value=-1)
        cond_batch = self._ref_audio_tensor.expand(bs, -1)
        dur_tensor = torch.tensor(durations, dtype=torch.long, device=self.device)

        batch_seed = eff_seeds[0]
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(batch_seed)

        results: list[SynthesisResult] = []
        sr = self.sample_rate

        with torch.inference_mode():
            generated, _ = self._tts.ema_model.sample(
                cond=cond_batch,
                text=text_batch,
                duration=dur_tensor,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=-1.0,
                seed=batch_seed,
            )
            for i, dur in enumerate(durations):
                mel_i = generated[i : i + 1, ref_audio_len:dur, :].to(torch.float32).permute(0, 2, 1)
                wav_i = self._tts.vocoder.decode(mel_i)
                if self._ref_rms > 0 and self._ref_rms < self.target_rms:
                    wav_i = wav_i * (self._ref_rms / self.target_rms)
                wav_np = wav_i.squeeze().cpu().numpy().astype(np.float32)

                if np.isnan(wav_np).any() or np.isinf(wav_np).any():
                    raise RuntimeError("Wykryto wartości NaN/Inf w wygenerowanym sygnale audio (batch).")

                wav_np = trim_silence(wav_np, sr, threshold_db=-40.0, keep_lead_ms=30, keep_tail_ms=55)
                wav_np = compress_internal_silences(wav_np, sr, max_silence_sec=max_internal_silence_sec)

                peak = float(np.abs(wav_np).max()) if len(wav_np) > 0 else 0.0
                if peak > 0.95:
                    wav_np = wav_np * (0.95 / peak)

                fade = min(int(sr * 0.01), len(wav_np) // 4)
                if fade > 0:
                    wav_np[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                    wav_np[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

                rms_out = float(np.sqrt(np.mean(wav_np**2))) if len(wav_np) > 0 else 0.0
                results.append(
                    SynthesisResult(
                        wav=wav_np,
                        sample_rate=sr,
                        duration_sec=len(wav_np) / sr,
                        rms=rms_out,
                        seed=batch_seed,
                        ipa_text=ipa_gens[i],
                    )
                )

        self._chunks_since_gc += bs
        if self._chunks_since_gc >= 15 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._chunks_since_gc = 0

        return results
