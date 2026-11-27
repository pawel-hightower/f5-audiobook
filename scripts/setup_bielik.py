"""Pobiera lokalnie model speakleash/Bielik-4.5B-v3.0-Instruct-GGUF (Q4_K_M)
oraz oficjalne binaria llama.cpp (Windows CUDA 12.4) do katalogu models/."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import list_repo_files

from f5_audiobook.config import MODELS_DIR

BIELIK_Q4_REPOS = [
    "oskarraszkiewicz/Bielik-4.5B-v3.0-Instruct-Q4_K_M-GGUF",
    "second-state/Bielik-4.5B-v3.0-Instruct-GGUF",
]
BIELIK_OFFICIAL_REPO = "speakleash/Bielik-4.5B-v3.0-Instruct-GGUF"
BIELIK_DIR = MODELS_DIR / "bielik-4.5b-v3"
LLAMA_CPP_DIR = MODELS_DIR / "llama-cpp"


def _download_hf_file_direct(repo: str, filename: str, dest: Path) -> Path:
    """Pobiera plik z HuggingFace przez bezpośredni HTTP streaming z postępem (omija zawieszenia Xet na Windows)."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "f5-audiobook-setup"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_mb = -50
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                mb = done // (1024 * 1024)
                if mb - last_mb >= 100 or (total and done >= total):
                    pct = f" ({100.0 * done / total:.1f}%)" if total else ""
                    tot_s = f" / {total / 1e6:.0f} MB" if total else ""
                    print(f"  Pobrano {done / 1e6:.0f} MB{tot_s}{pct}", flush=True)
                    last_mb = mb
    tmp.replace(dest)
    return dest


def download_bielik_gguf() -> Path:
    BIELIK_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in BIELIK_DIR.glob("*.gguf") if "q4_k_m" in p.name.lower() and p.stat().st_size > 1_000_000_000]
    if existing:
        print(f"Bielik GGUF już pobrany: {existing[0]} ({existing[0].stat().st_size / 1e9:.2f} GB)", flush=True)
        return existing[0]

    # 1) Spróbuj pobrać gotowy plik Q4_K_M (~2.88 GB — idealny na 6 GB VRAM RTX 4050)
    for repo in BIELIK_Q4_REPOS:
        try:
            files = list_repo_files(repo)
            q4_files = [f for f in files if "q4_k_m" in f.lower() and f.lower().endswith(".gguf")]
            if q4_files:
                fname = q4_files[0]
                dest = BIELIK_DIR / Path(fname).name
                print(f"Pobieram {repo} / {fname} -> {dest}...", flush=True)
                out = _download_hf_file_direct(repo, fname, dest)
                print(f"Pobrano Bielik GGUF: {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
                return out
        except Exception as e:
            print(f"Pomijam repo {repo}: {e}", flush=True)

    # 2) Fallback: pobierz oficjalny Q8_0 z speakleash i skwantyzuj lokalnie przez llama-quantize.exe do Q4_K_M
    files = list_repo_files(BIELIK_OFFICIAL_REPO)
    q8_files = [f for f in files if "q8_0" in f.lower() and f.lower().endswith(".gguf")]
    if not q8_files:
        raise RuntimeError(f"Nie znaleziono pliku GGUF w {BIELIK_OFFICIAL_REPO}: {files}")
    q8_name = q8_files[0]
    q8_dest = BIELIK_DIR / Path(q8_name).name
    print(f"Pobieram oficjalny {BIELIK_OFFICIAL_REPO} / {q8_name}...", flush=True)
    q8_path = _download_hf_file_direct(BIELIK_OFFICIAL_REPO, q8_name, q8_dest)
    quant_exe = LLAMA_CPP_DIR / "llama-quantize.exe"
    if quant_exe.exists():
        q4_path = BIELIK_DIR / "Bielik-4.5B-v3.0-Instruct.Q4_K_M.gguf"
        print(f"Kwantyzuję lokalnie {q8_path.name} -> {q4_path.name} (Q4_K_M)...", flush=True)
        subprocess.run(
            [str(quant_exe), str(q8_path), str(q4_path), "Q4_K_M"],
            cwd=str(LLAMA_CPP_DIR),
            check=True,
        )
        q8_path.unlink(missing_ok=True)
        print(f"Gotowy skwantyzowany Bielik: {q4_path} ({q4_path.stat().st_size / 1e9:.2f} GB)", flush=True)
        return q4_path
    return q8_path


def download_llama_cpp_win_cuda() -> Path:
    LLAMA_CPP_DIR.mkdir(parents=True, exist_ok=True)
    server_exe = LLAMA_CPP_DIR / "llama-server.exe"
    if server_exe.exists() and any(LLAMA_CPP_DIR.glob("cudart64_*.dll")):
        print(f"llama.cpp CUDA już obecne: {server_exe}", flush=True)
        return server_exe

    req = urllib.request.Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15",
        headers={"User-Agent": "f5-audiobook-setup"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.loads(resp.read().decode("utf-8"))

    main_zip = None
    cudart_zip = None
    picked_tag = ""

    def _pick(assets_list, pattern_fn):
        for a in assets_list:
            if pattern_fn(a["name"].lower()):
                return a
        return None

    for rel in releases:
        assets = rel.get("assets", [])
        if len(assets) < 5:
            continue
        m = _pick(assets, lambda n: n.startswith("llama-") and "bin-win-cuda-12.4-x64.zip" in n)
        c = _pick(assets, lambda n: n.startswith("cudart-llama-bin-win-cuda-12.4-x64.zip"))
        if m is None:
            m = _pick(assets, lambda n: n.startswith("llama-") and "bin-win-cuda-12" in n and "x64.zip" in n)
            c = _pick(assets, lambda n: n.startswith("cudart-llama-bin-win-cuda-12") and "x64.zip" in n)
        if m is None:
            m = _pick(assets, lambda n: n.startswith("llama-") and "bin-win-vulkan-x64.zip" in n)
            c = None
        if m is not None:
            main_zip = m
            cudart_zip = c
            picked_tag = rel.get("tag_name", "")
            break

    if main_zip is None:
        raise RuntimeError("Nie znaleziono paczki Windows CUDA/Vulkan w ostatnich 15 wydaniach llama.cpp")
    print(f"Wybrane wydanie llama.cpp: {picked_tag} ({main_zip['name']})", flush=True)

    to_fetch = [main_zip] + ([cudart_zip] if cudart_zip else [])
    for asset in to_fetch:
        url = asset["browser_download_url"]
        name = asset["name"]
        dst_zip = LLAMA_CPP_DIR / name
        print(f"Pobieram {name} ({asset.get('size', 0) / 1e6:.1f} MB)...", flush=True)
        urllib.request.urlretrieve(url, str(dst_zip))
        print(f"Rozpakowuję {name}...", flush=True)
        with zipfile.ZipFile(dst_zip, "r") as zf:
            zf.extractall(LLAMA_CPP_DIR)
        dst_zip.unlink(missing_ok=True)

    # Gdyby archiwum miało podfolder, przenieś zawartość na poziom LLAMA_CPP_DIR
    if not server_exe.exists():
        found = list(LLAMA_CPP_DIR.rglob("llama-server.exe"))
        if found:
            server_exe = found[0]
    if not server_exe.exists():
        raise RuntimeError(f"Nie znaleziono llama-server.exe po rozpakowaniu w {LLAMA_CPP_DIR}")

    print(f"Gotowe binaria llama.cpp: {server_exe}", flush=True)
    return server_exe


if __name__ == "__main__":
    download_llama_cpp_win_cuda()
    download_bielik_gguf()
    print("BIELIK + LLAMA.CPP SETUP COMPLETE!", flush=True)
