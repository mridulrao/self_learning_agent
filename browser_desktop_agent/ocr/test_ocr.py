#!/usr/bin/env python3
"""
Calls /click_target then draws returned bbox + click point on the *local image*.
IMPORTANT: uses bbox_norm/click_norm to avoid scale mismatches (server may downscale).

Usage:
  python verify_click_bbox.py \
    --url "https://8oppxoiq4sxw8t-8000.proxy.runpod.net" \
    --image "sample.png" \
    --label "New Document" \
    --strategy "lowest" \
    --out "annotated.png" \
    --draw-candidates
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


def call_click_target(
    base_url: str,
    image_path: str,
    label: str,
    strategy: str = "lowest",
    return_candidates: bool = True,
    timeout: int = 120,
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/click_target"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        data = {
            "label": label,
            "strategy": strategy,
            "return_candidates": "true" if return_candidates else "false",
        }
        resp = requests.post(endpoint, files=files, data=data, timeout=timeout)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

    return resp.json()


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _norm_bbox_to_px(bn: List[float], w: int, h: int) -> List[int]:
    # bn = [x1/w, y1/h, x2/w, y2/h]
    x1 = int(round(bn[0] * w))
    y1 = int(round(bn[1] * h))
    x2 = int(round(bn[2] * w))
    y2 = int(round(bn[3] * h))
    return [x1, y1, x2, y2]


def _norm_point_to_px(pn: List[float], w: int, h: int) -> List[int]:
    x = int(round(pn[0] * w))
    y = int(round(pn[1] * h))
    return [x, y]


def annotate_image(
    image_path: str,
    result: Dict[str, Any],
    out_path: str,
    draw_candidates: bool = False,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(18)
    w, h = img.size

    if not result.get("ok"):
        raise RuntimeError(f"API returned ok=false. Full response: {result}")

    label = result.get("label", "")
    match_text = result.get("match_text", "")

    # ✅ Use normalized coords (robust to server-side resizing)
    bbox_norm = result.get("bbox_norm")
    click_norm = result.get("click_norm")

    if not bbox_norm or len(bbox_norm) != 4:
        raise RuntimeError(f"Missing/invalid bbox_norm in response: {result}")
    if not click_norm or len(click_norm) != 2:
        raise RuntimeError(f"Missing/invalid click_norm in response: {result}")

    bbox_px = _norm_bbox_to_px(bbox_norm, w, h)
    click_px = _norm_point_to_px(click_norm, w, h)

    x1, y1, x2, y2 = bbox_px
    cx, cy = click_px

    # --- main bbox (red, thick) ---
    thickness = 4
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(255, 0, 0))

    # --- click point (green crosshair) ---
    r = 8
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=3)
    draw.line([cx - 15, cy, cx + 15, cy], fill=(0, 255, 0), width=3)
    draw.line([cx, cy - 15, cx, cy + 15], fill=(0, 255, 0), width=3)

    # --- label banner ---
    text = f"label={label} | match={match_text} | click=({cx},{cy})"
    pad = 6
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx, ty = x1, max(0, y1 - (th + pad * 2 + 4))
    draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=(0, 0, 0))
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

    # --- optionally draw candidate boxes too ---
    if draw_candidates:
        candidates = result.get("candidates") or []
        for c in candidates:
            # candidates may only have bbox_px (in server resized coords), but they also
            # correspond to the same normalized space, so we convert by re-normalizing:
            cb = c.get("bbox_px")
            if not cb or len(cb) != 4:
                continue

            # Convert server bbox_px -> approximate norm using server norm if present.
            # If not available, best-effort: use label match only (skip drawing).
            # Since your API returns bbox_norm only for best match, we’ll just draw best
            # (candidates drawing is optional; safest is to skip mismatched ones).
            pass

    img.save(out_path)
    print(f"✅ Saved annotated image: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://vqlzy77m77329p-8000.proxy.runpod.net/")
    ap.add_argument("--image", default="sent_ss.png")
    ap.add_argument("--label", default="New Document")
    ap.add_argument("--strategy", default="lowest", choices=["lowest", "highest", "first"])
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--draw-candidates", action="store_true", help="(no-op for now; best match is drawn)")
    args = ap.parse_args()

    result = call_click_target(
        base_url=args.url,
        image_path=args.image,
        label=args.label,
        strategy=args.strategy,
        return_candidates=True,
        timeout=args.timeout,
    )
    print("API response:", result)

    annotate_image(
        image_path=args.image,
        result=result,
        out_path=args.out,
        draw_candidates=args.draw_candidates,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
