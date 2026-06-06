FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/.cache/huggingface \
    TRANSFORMERS_CACHE=/models/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=nvidia/LocateAnything-3B \
    MODEL_DIR=/models/LocateAnything-3B \
    MAX_IMAGE_EDGE=1600 \
    MAX_IMAGE_PIXELS=2560000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
    accelerate==1.5.2 \
    av \
    bitsandbytes \
    decord \
    deepspeed==0.15.4 \
    filetype \
    einops \
    einops-exts \
    lmdb \
    hf_transfer \
    huggingface_hub \
    imagehash \
    "numpy>=1.25,<2" \
    opencv-python-headless \
    peft==0.12.0 \
    pillow \
    pydantic==2.7.1 \
    requests \
    safetensors \
    scikit-image \
    "scipy>=1.10.0" \
    sentencepiece==0.2.0 \
    "timm>=1.0.11" \
    tokenizers==0.22.0 \
    transformers==4.57.1 \
    fastapi \
    uvicorn

COPY env_config.py /app/env_config.py
COPY model /app/model

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN mkdir -p /models && python /app/model/locateanything/bootstrap.py --download-only

CMD ["sh", "-c", "uvicorn model.locateanything.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
