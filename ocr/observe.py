#!/usr/bin/env python3
"""
Calls /observe and saves the JSON observation.
Optionally draws bounding boxes for all OCR elements with confidence >= threshold.

Usage:
  python observe_draw_high_conf.py \
    --url "https://YOUR.proxy.runpod.net" \
    --image "sample.png" \
    --out-json "observation.json" \
    --draw \
    --min-conf 0.90 \
    --out-image "annotated.png"

Notes:
- /observe returns bbox_norm/click_norm w.r.t. the server-processed image size.
- We draw on the local image using bbox_norm -> local px, so it stays correct even
  if the server resized before OCR.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# HTTP
# -----------------------------
def call_observe(base_url: str, image_path: str, timeout: int = 120) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/observe"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        resp = requests.post(endpoint, files=files, timeout=timeout)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

    return resp.json()


# -----------------------------
# Drawing helpers
# -----------------------------
def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _norm_bbox_to_px(bn: List[float], w: int, h: int) -> List[int]:
    x1 = int(round(bn[0] * w))
    y1 = int(round(bn[1] * h))
    x2 = int(round(bn[2] * w))
    y2 = int(round(bn[3] * h))
    return [x1, y1, x2, y2]


def _norm_point_to_px(pn: List[float], w: int, h: int) -> List[int]:
    x = int(round(pn[0] * w))
    y = int(round(pn[1] * h))
    return [x, y]


def save_json(obj: Dict[str, Any], out_json: str) -> None:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved observation JSON: {out_json}")


def annotate_high_conf(
    image_path: str,
    observation: Dict[str, Any],
    out_image: str,
    min_conf: float,
    max_elements: Optional[int] = None,
    text_maxlen: int = 60,
    box_thickness: int = 3,
    draw_click: bool = False,
) -> None:
    """
    Draw bounding boxes for all OCR elements with confidence >= min_conf.
    Uses bbox_norm (robust to server resizing).
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(16)
    W, H = img.size

    if not observation.get("ok"):
        raise RuntimeError(f"/observe returned ok=false. Full response: {observation}")

    elements = observation.get("elements") or []
    if not isinstance(elements, list):
        raise RuntimeError(f"Invalid elements in observation: {type(elements)}")

    # Filter by conf + valid coords
    filtered: List[Dict[str, Any]] = []
    for el in elements:
        try:
            conf = float(el.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        bn = el.get("bbox_norm")
        pn = el.get("click_norm")
        if conf < min_conf:
            continue
        if not bn or len(bn) != 4:
            continue
        if draw_click and (not pn or len(pn) != 2):
            continue

        filtered.append(el)

    # Sort high->low conf so the most reliable labels get drawn first
    filtered.sort(key=lambda e: float(e.get("confidence", 0.0)), reverse=True)

    if max_elements is not None:
        filtered = filtered[:max_elements]

    # Header
    header = f"drawn={len(filtered)} / total={len(elements)} (min_conf={min_conf})"
    pad = 6
    tb = draw.textbbox((0, 0), header, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0))
    draw.text((pad, pad), header, fill=(255, 255, 255), font=font)

    for idx, el in enumerate(filtered):
        bn = el["bbox_norm"]
        bbox_px = _norm_bbox_to_px(bn, W, H)
        x1, y1, x2, y2 = bbox_px

        # bbox outline
        for t in range(box_thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(255, 0, 0))

        # optional click point
        if draw_click:
            pn = el["click_norm"]
            cx, cy = _norm_point_to_px(pn, W, H)
            r = 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=2)
            draw.line([cx - 12, cy, cx + 12, cy], fill=(0, 255, 0), width=2)
            draw.line([cx, cy - 12, cx, cy + 12], fill=(0, 255, 0), width=2)

        # label
        text = (el.get("text") or "").strip().replace("\n", " ")
        if len(text) > text_maxlen:
            text = text[: text_maxlen - 1] + "…"

        el_id = el.get("id", f"idx_{idx}")
        conf = float(el.get("confidence", 0.0))
        banner = f"{conf:.2f} | {el_id} | {text}"

        pad2 = 4
        tb2 = draw.textbbox((0, 0), banner, font=font)
        tw2, th2 = tb2[2] - tb2[0], tb2[3] - tb2[1]
        tx = x1
        ty = max(0, y1 - (th2 + pad2 * 2 + 2))
        draw.rectangle([tx, ty, tx + tw2 + pad2 * 2, ty + th2 + pad2 * 2], fill=(0, 0, 0))
        draw.text((tx + pad2, ty + pad2), banner, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)
    img.save(out_image)
    print(f"✅ Saved annotated image: {out_image}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://34o88l6b3a2hh2-8000.proxy.runpod.net")
    ap.add_argument("--image", default="sample_simple.png")
    ap.add_argument("--out-json", default="observation.json")
    ap.add_argument("--timeout", type=int, default=120)

    ap.add_argument("--draw", action="store_true", help="Draw bboxes for high-confidence elements")
    ap.add_argument("--out-image", default="annotated_observe.png")
    ap.add_argument("--min-conf", type=float, default=0.90, help="Only draw elements with confidence >= this")
    ap.add_argument("--max-elements", type=int, default=None)
    ap.add_argument("--text-maxlen", type=int, default=60)
    ap.add_argument("--box-thickness", type=int, default=3)
    ap.add_argument("--draw-click", action="store_true", help="Also draw click point crosshair")

    args = ap.parse_args()

    obs = call_observe(args.url, args.image, timeout=args.timeout)
    print("API response keys:", list(obs.keys()))
    print(f"elements: {len(obs.get('elements') or [])}")

    save_json(obs, args.out_json)

    if args.draw:
        annotate_high_conf(
            image_path=args.image,
            observation=obs,
            out_image=args.out_image,
            min_conf=args.min_conf,
            max_elements=args.max_elements,
            text_maxlen=args.text_maxlen,
            box_thickness=args.box_thickness,
            draw_click=args.draw_click,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
