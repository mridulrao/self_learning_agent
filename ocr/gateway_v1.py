# gateway.py
from __future__ import annotations

import io
import os
from typing import List, Optional, Dict, Any

import numpy as np
import easyocr
from rapidfuzz import fuzz

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageEnhance

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# ------------------------------------------------------------------------------
# Environment / config
# ------------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-OCR")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
NGRAM_SIZE = int(os.getenv("NGRAM_SIZE", "30"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "90"))
WHITELIST_TOKEN_IDS = os.getenv("WHITELIST_TOKEN_IDS", "128821,128822")  # <td>, </td>
DOWNLOAD_DIR = os.getenv("VLLM_DOWNLOAD_DIR", os.getenv("HF_HOME", "/models"))
TP_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", "<image>\nFree OCR.")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# vLLM tuning knobs
GPU_MEM_UTIL = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))

# Screenshot preprocessing (for OCR + UI text)
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))
CONTRAST = float(os.getenv("CONTRAST", "1.15"))
SHARPNESS = float(os.getenv("SHARPNESS", "1.2"))

# EasyOCR config
EASYOCR_LANGS = os.getenv("EASYOCR_LANGS", "en").split(",")
EASYOCR_GPU = os.getenv("EASYOCR_GPU", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
FUZZY_MATCH_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))


def _parse_whitelist(s: str) -> set[int]:
    """Parse a comma-separated list of ints into a set."""
    try:
        return {int(x.strip()) for x in s.split(",") if x.strip()}
    except Exception:
        return set()


whitelist_token_ids = _parse_whitelist(WHITELIST_TOKEN_IDS)

# ------------------------------------------------------------------------------
# Decide where to load model from (local vs HF)
# ------------------------------------------------------------------------------

def _pick_model_source() -> str:
    """
    If download_model.py has already put a full HF snapshot in DOWNLOAD_DIR,
    prefer that local path. Otherwise fall back to MODEL_ID (HF).
    """
    local_config = os.path.join(DOWNLOAD_DIR, "config.json")
    if os.path.isdir(DOWNLOAD_DIR) and os.path.exists(local_config):
        print(f"[gateway] Using local model at: {DOWNLOAD_DIR}", flush=True)
        return DOWNLOAD_DIR

    model_path_env = os.getenv("MODEL_PATH")
    if model_path_env and os.path.exists(model_path_env):
        print(f"[gateway] Using MODEL_PATH={model_path_env}", flush=True)
        return model_path_env

    print(f"[gateway] Using HF model id: {MODEL_ID} (download_dir={DOWNLOAD_DIR})", flush=True)
    return MODEL_ID


MODEL_SOURCE = _pick_model_source()

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(
    title="DeepSeek-OCR Gateway (OCR + Click Target)",
    version="2.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# vLLM engine (initialize once per process)
# ------------------------------------------------------------------------------
llm = LLM(
    model=MODEL_SOURCE,          # local path or HF id
    tensor_parallel_size=TP_SIZE,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    download_dir=DOWNLOAD_DIR,   # vLLM will reuse /models; no re-download needed
    logits_processors=[NGramPerReqLogitsProcessor],
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
)

# ------------------------------------------------------------------------------
# EasyOCR reader (for bounding boxes + click coords)
# ------------------------------------------------------------------------------
# This is the key piece that gives you deterministic bboxes.
reader = easyocr.Reader(EASYOCR_LANGS, gpu=EASYOCR_GPU)

# ------------------------------------------------------------------------------
# Models / helpers
# ------------------------------------------------------------------------------
class OCRResponse(BaseModel):
    texts: List[str]


class ClickTargetResponse(BaseModel):
    ok: bool
    label: str
    match_text: Optional[str] = None
    bbox_px: Optional[List[int]] = None     # [x1,y1,x2,y2]
    click_px: Optional[List[int]] = None    # [x,y]
    bbox_norm: Optional[List[float]] = None # [x1/w,y1/h,x2/w,y2/h]
    click_norm: Optional[List[float]] = None# [x/w,y/h]
    candidates: Optional[List[Dict[str, Any]]] = None


def _pil_from_upload(file: UploadFile) -> Image.Image:
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("empty file")
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image '{file.filename}': {e}",
        ) from e


def _preprocess_for_ui(img: Image.Image) -> Image.Image:
    """
    Preprocess screenshot to improve OCR stability on UI text.
    Also downscales to prevent huge screenshots from slowing OCR or bloating multimodal tokens.
    """
    w, h = img.size
    scale = min(1.0, float(MAX_SIDE) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    if abs(CONTRAST - 1.0) > 1e-3:
        img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    if abs(SHARPNESS - 1.0) > 1e-3:
        img = ImageEnhance.Sharpness(img).enhance(SHARPNESS)

    return img


def _bbox_from_poly(poly) -> List[int]:
    # poly: [[x,y], [x,y], [x,y], [x,y]]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _center(b: List[int]) -> List[int]:
    x1, y1, x2, y2 = b
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def _norm_bbox(b: List[int], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = b
    # clamp not strictly needed since OCR returns valid coords, but safe
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def _norm_point(p: List[int], w: int, h: int) -> List[float]:
    return [p[0] / w, p[1] / h]


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "DeepSeek-OCR gateway is running."}


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
        "tensor_parallel_size": TP_SIZE,
        "download_dir": DOWNLOAD_DIR,
        "has_hf_token": bool(HF_TOKEN),
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "max_model_len": MAX_MODEL_LEN,
        "easyocr_gpu": EASYOCR_GPU,
        "easyocr_langs": EASYOCR_LANGS,
        "max_side": MAX_SIDE,
        "contrast": CONTRAST,
        "sharpness": SHARPNESS,
    }


# ------------------------------
# Your original OCR endpoint
# ------------------------------
@app.post("/ocr", response_model=OCRResponse)
def ocr(
    files: List[UploadFile] = File(..., description="One or more image files"),
    prompt: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images = [_pil_from_upload(f) for f in files]
    user_prompt = (prompt or DEFAULT_PROMPT).strip()

    model_inputs = [{"prompt": user_prompt, "multi_modal_data": {"image": img}} for img in images]

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_args=dict(
            ngram_size=NGRAM_SIZE,
            window_size=WINDOW_SIZE,
            whitelist_token_ids=whitelist_token_ids,
        ),
        skip_special_tokens=False,
    )

    try:
        outputs = llm.generate(model_inputs, sampling)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
        else:
            texts.append(out.outputs[0].text)

    return OCRResponse(texts=texts)


# ------------------------------
# New: click target endpoint
# ------------------------------
@app.post("/click_target", response_model=ClickTargetResponse)
def click_target(
    file: UploadFile = File(..., description="Screenshot image"),
    label: str = Form(..., description="Target label text to click (e.g. 'New Document')"),
    strategy: str = Form("lowest", description="lowest|highest|first"),
    return_candidates: bool = Form(False),
):
    """
    Returns coordinates for a given label.
    If multiple matches exist:
      - strategy=lowest picks the one with largest y-center (bottom-most)
      - strategy=highest picks the one with smallest y-center (top-most)
      - strategy=first returns the first matched candidate
    """
    if not label or not label.strip():
        raise HTTPException(status_code=400, detail="label is empty")

    img = _pil_from_upload(file)
    img = _preprocess_for_ui(img)
    w, h = img.size

    np_img = np.array(img)
    results = reader.readtext(np_img, detail=1)  # [(poly, text, conf), ...]

    target = label.strip()
    candidates: List[Dict[str, Any]] = []

    for poly, text, conf in results:
        t = (text or "").strip()
        if not t:
            continue

        score = max(
            fuzz.ratio(target.lower(), t.lower()),
            fuzz.partial_ratio(target.lower(), t.lower()),
        )

        if score >= FUZZY_MATCH_THRESHOLD:
            bbox = _bbox_from_poly(poly)
            click = _center(bbox)
            candidates.append(
                {
                    "text": t,
                    "conf": float(conf),
                    "score": float(score),
                    "bbox_px": bbox,
                    "click_px": click,
                }
            )

    if not candidates:
        return ClickTargetResponse(ok=False, label=label, candidates=candidates if return_candidates else None)

    if strategy == "lowest":
        best = max(candidates, key=lambda c: c["click_px"][1])
    elif strategy == "highest":
        best = min(candidates, key=lambda c: c["click_px"][1])
    else:
        best = candidates[0]

    bbox_px = best["bbox_px"]
    click_px = best["click_px"]

    return ClickTargetResponse(
        ok=True,
        label=label,
        match_text=best["text"],
        bbox_px=bbox_px,
        click_px=click_px,
        bbox_norm=_norm_bbox(bbox_px, w, h),
        click_norm=_norm_point(click_px, w, h),
        candidates=candidates if return_candidates else None,
    )


if __name__ == "__main__":
    import uvicorn

    port_env = os.getenv("PORT", "8000")
    try:
        port = int(port_env)
    except ValueError:
        port = 8000

    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
