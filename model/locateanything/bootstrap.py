"""
Bootstrap script for downloading and initializing the LocateAnything-3B model.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor, AutoTokenizer

def _load_env_config() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "env_config.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            from env_config import load_env_file

            load_env_file()
            return


_load_env_config()


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and initialize LocateAnything-3B.")
    parser.add_argument("--download-only", action="store_true", help="Only download model files.")
    args = parser.parse_args()

    model_id = os.environ.get("MODEL_ID", "nvidia/LocateAnything-3B")
    model_dir = os.environ.get("MODEL_DIR", "/models/LocateAnything-3B")
    hf_token = os.environ.get("HF_TOKEN")

    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        token=hf_token,
        resume_download=True,
    )
    print(f"Downloaded {model_id} to {model_dir}")

    if args.download_only:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    print(
        "LocateAnything initialized",
        {
            "device": device,
            "dtype": str(dtype),
            "tokenizer": tokenizer.__class__.__name__,
            "processor": processor.__class__.__name__,
            "model": model.__class__.__name__,
        },
    )


if __name__ == "__main__":
    main()
