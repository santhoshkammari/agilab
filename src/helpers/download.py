#!/usr/bin/env python3
"""
download.py – simple Hugging Face Hub downloader

supports:
  • snapshot download (full repo clone into ./models)
  • targeted GGUF file download (constructs GGUF repo/filenames dynamically)

examples
--------

# 1. download a model snapshot (default repo_type="model")
python download.py -m gpt2

# 2. download a dataset snapshot
python download.py -m google/fleurs --type dataset

# 3. snapshot with a specific revision (branch/tag/commit)
python download.py -m lysandre/arxiv-nlp -r v1.0

# 4. download GGUF quantized weights (default owner=bartowski, quants=Q4_K_M bf16)
python download.py -m Qwen/Qwen3-4B-Instruct-2507 --gguf

# this pulls:
#   bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
#   bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/Qwen_Qwen3-4B-Instruct-2507-bf16.gguf
# into ./models/Qwen_Qwen3-4B-Instruct-2507-GGUF/

# 5. specify custom quantizations
python download.py -m Qwen/Qwen3-4B-Instruct-2507 --gguf --quant Q5_K_S Q8_0

# 6. change GGUF repo owner
python download.py -m Qwen/Qwen3-4B-Instruct-2507 --gguf --gguf-owner someuser

# 7. change output directory
python download.py -m gpt2 -o /tmp/hf_models
python download.py -m Qwen/Qwen3-4B-Instruct-2507 --gguf -o /tmp/hf_models
"""

import argparse
import os
import re
import sys
from typing import List
from huggingface_hub import snapshot_download, hf_hub_download

def download_snapshot(model_id: str, revision: str | None, repo_type: str, save_dir: str, token: str | None = None) -> str:
    os.makedirs(save_dir, exist_ok=True)
    print(f"[snapshot] downloading {model_id}@{revision or 'main'} -> {save_dir}")
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        repo_type=repo_type,
        local_dir=save_dir,
        token=token,
    )
    print(f"[snapshot] done: {path}")
    return path

def build_gguf_repo_and_files(
    base_model_id: str,
    owner: str,
    quants: List[str]
) -> tuple[str, List[str], str]:
    """
    base_model_id: e.g. 'Qwen/Qwen3-4B-Instruct-2507'
    owner: e.g. 'bartowski'
    quants: e.g. ['Q4_K_M','bf16']
    returns: (repo_id, filenames, base_token)
    """
    base_token = base_model_id.replace("/", "_")  # Qwen_Qwen3-4B-Instruct-2507
    repo_id = f"{owner}/{base_token}-GGUF"
    filenames = [f"{base_token}-{q}.gguf" for q in quants]
    return repo_id, filenames, base_token

def download_gguf(
    base_model_id: str,
    owner: str,
    quants: List[str],
    revision: str | None,
    save_root: str,
    token: str | None = None,
) -> List[str]:
    repo_id, filenames, base_token = build_gguf_repo_and_files(base_model_id, owner, quants)
    target_dir = os.path.join(save_root, f"{base_token}-GGUF")
    os.makedirs(target_dir, exist_ok=True)

    print(f"[gguf] repo: {repo_id}  revision: {revision or 'main'}")
    saved = []
    for fname in filenames:
        print(f"[gguf] downloading {fname} -> {target_dir}")
        # keep original folder structure; avoid corrupting cache by writing a copy to local_dir
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            revision=revision,
            local_dir=target_dir,
            repo_type="model",
            token=token,
        )
        print(f"[gguf] saved: {local_path}")
        saved.append(local_path)
    return saved

def parse_args():
    p = argparse.ArgumentParser(description="Download from Hugging Face Hub (snapshot or GGUF).")
    p.add_argument("-m", "--model", required=True, help="Base model repo id (e.g. gpt2 or Qwen/Qwen3-4B-Instruct-2507)")
    p.add_argument("-r", "--revision", default=None, help="Optional revision (branch/tag/commit)")
    p.add_argument("--type", default="model", choices=["model", "dataset", "space"], help="Repo type for snapshot mode")
    p.add_argument("-o", "--out", default="models", help="Output root directory (default: ./models)")

    # GGUF options
    g = p.add_argument_group("gguf")
    g.add_argument("--gguf", action="store_true", help="Download GGUF files instead of a full snapshot")
    g.add_argument("--gguf-owner", default="bartowski", help="Owner/org of GGUF repo (default: bartowski)")
    g.add_argument(
        "--quant",
        nargs="+",
        default=["Q4_K_M", "bf16"],
        help="Quant(s) for GGUF; space or comma-separated (default: Q4_K_M bf16). Examples: --quant Q2_K Q5_K_S or --quant Q2_K,Q5_K_S,bf16",
    )
    # Auth token
    p.add_argument(
        "--token",
        default="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",
        help="Hugging Face access token. Defaults to the provided token.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    save_root = os.path.abspath(args.out)
    
    # Allow comma-separated quants alongside space-separated (e.g., "Q4_K_M,bf16 Q8_0")
    if isinstance(getattr(args, "quant", None), list):
        quants: List[str] = []
        for token in args.quant:
            parts = [p.strip() for p in re.split(r",+", token) if p.strip()]
            quants.extend(parts)
    else:
        quants = ["Q4_K_M", "bf16"]

    if args.gguf:
        # GGUF path: {owner}/{MODEL_WITH_SLASHES_REPLACED}-GGUF / {MODEL_WITH_SLASHES_REPLACED}-{quant}.gguf
        # e.g. for Qwen/Qwen3-4B-Instruct-2507:
        # repo: bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF
        # files: Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf, Qwen_Qwen3-4B-Instruct-2507-bf16.gguf
        download_gguf(
            base_model_id=args.model,
            owner=args.gguf_owner,
            quants=quants,
            revision=args.revision,
            save_root=save_root,
            token=args.token,
        )
    else:
        download_snapshot(
            model_id=args.model,
            revision=args.revision,
            repo_type=args.type,
            save_dir=save_root,
            token=args.token,
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("cancelled.", file=sys.stderr)
        sys.exit(130)
