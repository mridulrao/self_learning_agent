#!/usr/bin/env python3
"""
validate_new_document_click.py

A standalone validation harness to debug the "New Document" click end-to-end.

What it does:
1) Capture a screenshot (pyautogui.screenshot) OR load an existing screenshot file.
2) Send it to your OCR server /click_target(label="New Document", return_candidates=true).
3) Print all candidates (click_norm + click_px + conf).
4) Choose a candidate using a deterministic policy (default: TOPMOST New Document).
5) Convert click_norm -> screen_px (with optional scaling correction if sizes mismatch).
6) Perform the click (pyautogui.click) with an optional overlay marker.
7) Take an "after" screenshot and optionally:
   - Call /observe on before/after to check for a UI change
   - Save annotated images (before + after) with chosen click point/bbox

Usage examples:

# A) Live capture + click (most useful)
python validate_new_document_click.py \
  --ocr-url "https://...proxy.runpod.net" \
  --label "New Document" \
  --mode topmost \
  --do-click \
  --observe \
  --out-dir "recordings/20260107_163715"

# B) Use an existing screenshot, don't click (pure analysis)
python validate_new_document_click.py \
  --ocr-url "https://...proxy.runpod.net" \
  --image "recordings/20260107_163715/sent_desktop_2__New_Document__fullscreen__....png" \
  --mode topmost \
  --out-dir "recordings/20260107_163715"

Notes:
- This script uses pyautogui.screenshot() for capture (better coordinate consistency on macOS).
- If your screenshot size != pyautogui.size(), it can optionally apply scale correction.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyautogui
import requests
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Helpers
# -----------------------------
def _safe_text(s: Any) -> str:
    return str(s or "").replace("\n", " ").strip()


def _normalize_text(s: str) -> str:
    s = _safe_text(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", (s or "").strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:80] if s else "empty"


def _norm_to_px(click_norm: List[float], w: int, h: int) -> Tuple[int, int]:
    return int(round(float(click_norm[0]) * w)), int(round(float(click_norm[1]) * h))


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _post_click_target(ocr_url: str, img: Image.Image, label: str, timeout_s: int) -> Dict[str, Any]:
    endpoint = ocr_url.rstrip("/") + "/click_target"
    png = _pil_to_png_bytes(img)
    files = {"file": ("screenshot.png", png, "image/png")}
    data = {"label": label, "return_candidates": "true"}
    resp = requests.post(endpoint, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _post_observe(ocr_url: str, img: Image.Image, timeout_s: int) -> Dict[str, Any]:
    endpoint = ocr_url.rstrip("/") + "/observe"
    png = _pil_to_png_bytes(img)
    files = {"file": ("screenshot.png", png, "image/png")}
    resp = requests.post(endpoint, files=files, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _pick_candidate(candidates: List[Dict[str, Any]], label: str, mode: str, rec_xy: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    """
    modes:
      - topmost: smallest y_norm among candidates matching label
      - bottommost: largest y_norm
      - highest_conf: max conf
      - nearest_rec: min distance to rec_xy (requires rec_xy)
    """
    if not candidates:
        return None

    # Filter to label matches (case-insensitive contains)
    label_n = _normalize_text(label)
    pool = []
    for c in candidates:
        txt = _normalize_text(c.get("text") or "")
        if label_n and label_n not in txt:
            continue
        cn = c.get("click_norm")
        if not (isinstance(cn, list) and len(cn) == 2):
            continue
        conf = float(c.get("conf") or 0.0)
        pool.append((c, conf, float(cn[0]), float(cn[1])))

    if not pool:
        # fall back: any candidate with click_norm
        for c in candidates:
            cn = c.get("click_norm")
            if isinstance(cn, list) and len(cn) == 2:
                conf = float(c.get("conf") or 0.0)
                pool.append((c, conf, float(cn[0]), float(cn[1])))

    if not pool:
        return None

    if mode == "topmost":
        pool.sort(key=lambda t: t[3])  # y_norm ascending
        return pool[0][0]

    if mode == "bottommost":
        pool.sort(key=lambda t: t[3], reverse=True)
        return pool[0][0]

    if mode == "highest_conf":
        pool.sort(key=lambda t: t[1], reverse=True)
        return pool[0][0]

    if mode == "nearest_rec":
        if not rec_xy:
            raise ValueError("--mode nearest_rec requires --rec-x and --rec-y")
        # distance uses click_px if present else click_norm * (arbitrary) later.
        def dist(item):
            c = item[0]
            cp = c.get("click_px")
            if isinstance(cp, list) and len(cp) == 2:
                dx = float(cp[0]) - rec_xy[0]
                dy = float(cp[1]) - rec_xy[1]
                return math.hypot(dx, dy)
            # fallback approximate using norm
            dx = item[2] - 0.5
            dy = item[3] - 0.5
            return math.hypot(dx, dy)

        pool.sort(key=dist)
        return pool[0][0]

    raise ValueError(f"Unknown mode: {mode}")


def _draw_annotation(img: Image.Image, click_xy: Tuple[int, int], bbox_px: Optional[List[int]] = None, title: str = "") -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)

    # bbox
    if bbox_px and len(bbox_px) == 4:
        x0, y0, x1, y1 = bbox_px
        d.rectangle([x0, y0, x1, y1], outline="red", width=3)

    # click crosshair
    x, y = click_xy
    r = 18
    d.ellipse([x - r, y - r, x + r, y + r], outline="lime", width=4)
    d.line([x - r, y, x + r, y], fill="lime", width=3)
    d.line([x, y - r, x, y + r], fill="lime", width=3)

    if title:
        # simple label at top-left
        d.rectangle([0, 0, 900, 38], fill=(0, 0, 0, 160))
        d.text((10, 8), title, fill="white")
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-url", default="https://vqlzy77m77329p-8000.proxy.runpod.net")
    ap.add_argument("--label", default="New Document")
    ap.add_argument("--timeout", type=int, default=120)

    ap.add_argument("--image", default="recordings/20260107_163715/sent_desktop_2__New_Document__fullscreen__1767835600428.png")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="")

    ap.add_argument("--mode", default="topmost", choices=["topmost", "bottommost", "highest_conf", "nearest_rec"])
    ap.add_argument("--rec-x", type=int, default=None)
    ap.add_argument("--rec-y", type=int, default=None, help="Recorded click y (optional)")

    ap.add_argument("--do-click", action="store_true", help="Actually click on screen")
    ap.add_argument("--sleep-after", type=float, default=1.0, help="Seconds to wait after click")
    ap.add_argument("--observe", action="store_true", help="Call /observe before and after")
    ap.add_argument("--apply-scale-fix", action="store_true",
                    help="If screenshot size != pyautogui.size(), scale click coords into pyautogui space")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = _slug(args.prefix) + "__" if args.prefix else ""

    # 1) Capture/load screenshot
    if args.image:
        img = Image.open(args.image).convert("RGB")
        source = f"file:{args.image}"
    else:
        img = pyautogui.screenshot().convert("RGB")
        source = "live"

    W, H = img.size
    sw, sh = pyautogui.size()

    print(f"[info] source={source}")
    print(f"[info] screenshot_size=({W},{H}) pyautogui.size=({sw},{sh})")

    # Save the exact input image
    ts = int(time.time() * 1000)
    in_path = out_dir / f"{prefix}input__{_slug(args.label)}__{ts}.png"
    img.save(in_path, "PNG")
    print(f"[saved] {in_path}")

    # Optional observe before
    obs_before = None
    if args.observe:
        try:
            obs_before = _post_observe(args.ocr_url, img, timeout_s=args.timeout)
            (out_dir / f"{prefix}observe_before__{_slug(args.label)}__{ts}.json").write_text(
                json.dumps(obs_before, indent=2),
                encoding="utf-8",
            )
            print("[observe] before ok=", bool(obs_before.get("ok")))
        except Exception as e:
            print("[observe] before failed:", e)

    # 2) click_target
    ct = _post_click_target(args.ocr_url, img, label=args.label, timeout_s=args.timeout)
    ct_path = out_dir / f"{prefix}click_target__{_slug(args.label)}__{ts}.json"
    ct_path.write_text(json.dumps(ct, indent=2), encoding="utf-8")
    print(f"[saved] {ct_path}")

    if not ct.get("ok"):
        print("[click_target] ok=false. Exiting.")
        return

    candidates = ct.get("candidates") or []
    print(f"[click_target] candidates={len(candidates)}")

    # Print candidates
    for i, c in enumerate(candidates[:20]):
        txt = c.get("text")
        conf = c.get("conf")
        cn = c.get("click_norm")
        cp = c.get("click_px")
        print(f"  - #{i} text={txt!r} conf={conf} click_norm={cn} click_px={cp}")

    rec_xy = (args.rec_x, args.rec_y) if (args.rec_x is not None and args.rec_y is not None) else None

    chosen = _pick_candidate(candidates, args.label, args.mode, rec_xy=rec_xy)
    if not chosen:
        print("[pick] no candidate chosen. Exiting.")
        return

    chosen_cn = chosen.get("click_norm")
    chosen_bbox = chosen.get("bbox_px")
    chosen_conf = float(chosen.get("conf") or 0.0)
    print(f"[pick] mode={args.mode} chosen_text={chosen.get('text')!r} conf={chosen_conf} click_norm={chosen_cn} bbox_px={chosen_bbox}")

    # 3) Convert to screen coords
    sx, sy = _norm_to_px(chosen_cn, W, H)

    # optional scale fix
    if args.apply_scale_fix and (W != sw or H != sh):
        scale_x = sw / float(W)
        scale_y = sh / float(H)
        sx2 = int(round(sx * scale_x))
        sy2 = int(round(sy * scale_y))
        print(f"[scale_fix] scale_x={scale_x:.4f} scale_y={scale_y:.4f} ({sx},{sy})->({sx2},{sy2})")
        sx, sy = sx2, sy2

    print(f"[click] screen_px=({sx},{sy})")

    # 4) Annotate input with chosen point
    ann = _draw_annotation(img, (int(round(float(chosen.get("click_px", [sx, sy])[0] if isinstance(chosen.get("click_px"), list) else sx))),
                                int(round(float(chosen.get("click_px", [sx, sy])[1] if isinstance(chosen.get("click_px"), list) else sy)))),
                          bbox_px=chosen_bbox,
                          title=f"{args.label} | chosen={args.mode} conf={chosen_conf:.3f}")
    ann_path = out_dir / f"{prefix}annotated_input__{_slug(args.label)}__{ts}.png"
    ann.save(ann_path, "PNG")
    print(f"[saved] {ann_path}")

    # 5) Optional click
    if args.do_click and not args.image:
        print("[do_click] clicking now...")
        pyautogui.click(sx, sy)
        time.sleep(float(args.sleep_after))

        # After screenshot
        img_after = pyautogui.screenshot().convert("RGB")
        after_path = out_dir / f"{prefix}after__{_slug(args.label)}__{ts}.png"
        img_after.save(after_path, "PNG")
        print(f"[saved] {after_path}")

        # Optional observe after
        if args.observe:
            try:
                obs_after = _post_observe(args.ocr_url, img_after, timeout_s=args.timeout)
                (out_dir / f"{prefix}observe_after__{_slug(args.label)}__{ts}.json").write_text(
                    json.dumps(obs_after, indent=2),
                    encoding="utf-8",
                )
                print("[observe] after ok=", bool(obs_after.get("ok")))

                # quick diff hints
                if obs_before and obs_before.get("ok") and obs_after.get("ok"):
                    before_texts = {(_normalize_text(e.get("text") or "")) for e in (obs_before.get("elements") or [])}
                    after_texts = {(_normalize_text(e.get("text") or "")) for e in (obs_after.get("elements") or [])}
                    added = sorted([t for t in after_texts - before_texts if t])[:30]
                    removed = sorted([t for t in before_texts - after_texts if t])[:30]
                    print("[observe] added (sample):", added[:10])
                    print("[observe] removed (sample):", removed[:10])

            except Exception as e:
                print("[observe] after failed:", e)

    print("\nDone.")


if __name__ == "__main__":
    main()
