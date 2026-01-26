# download_model.py
import os
from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-OCR")
# Same logic as gateway.py, no leading slash in env var name
DOWNLOAD_DIR = os.getenv("VLLM_DOWNLOAD_DIR", os.getenv("HF_HOME", "/models"))

print(f"Downloading model {MODEL_ID} into {DOWNLOAD_DIR} ...", flush=True)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    cache_dir=DOWNLOAD_DIR,
    local_dir=DOWNLOAD_DIR,
    local_dir_use_symlinks=False,
)

print("Download complete.", flush=True)
