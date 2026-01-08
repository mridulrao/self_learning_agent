#!/usr/bin/env python3
from __future__ import annotations

"""
desktop_agent.py — Mac Desktop Agent (Primitives + OCR Observe + Validator)

Key behaviors:
- Executes primitive desktop workflow steps with retries + validation.
- Uses OCR (/observe, /click_target) to ground clicks and validate outcomes.
- Handles Retina coordinate mismatch by always clicking in pyautogui coordinate space.
- When label == "New Document", saves the exact screenshot sent to OCR/Vision alongside the workflow JSON.
"""

import time
import subprocess
import base64
import logging
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import threading
import os
import argparse
import io
import math

import requests
from pynput.keyboard import Controller, Key
import pyautogui
from PIL import Image, ImageGrab  # ImageGrab kept for compatibility (not used for OCR-click screenshots)
import anthropic
from dotenv import load_dotenv

# Quartz click optional (pyautogui is default)
try:
    from Quartz import (
        CGEventCreateMouseEvent,
        CGEventPost,
        CGEventSetIntegerValueField,
        CGEventSourceCreate,
        kCGEventSourceStateHIDSystemState,
        kCGHIDEventTap,
        kCGEventMouseMoved,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGMouseButtonLeft,
        kCGMouseEventClickState,
    )
    _QUARTZ_OK = True
except Exception:
    _QUARTZ_OK = False

load_dotenv()


# -----------------------------
# Env helpers
# -----------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip()


# -----------------------------
# Logging
# -----------------------------
def _parse_log_level(s: str, default: int = logging.INFO) -> int:
    if not s:
        return default
    s = s.strip().upper()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }.get(s, default)


def setup_logger(name: str) -> logging.Logger:
    level = _parse_log_level(_env_str("DESKTOP_LOG_LEVEL", "INFO"), default=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setup_logger("MacDesktopAgent")


# -----------------------------
# Error types
# -----------------------------
ERROR_FOCUS = "focus_error"
ERROR_CLICK = "click_error"
ERROR_TYPE = "type_error"
ERROR_HOTKEY = "hotkey_error"
ERROR_ELEMENT_NOT_FOUND = "element_not_found"
ERROR_VERIFICATION = "verification_failed"
ERROR_FILE_VALIDATION = "file_validation_failed"
ERROR_OBSERVE = "observe_failed"
ERROR_UNKNOWN = "unknown_error"


# -----------------------------
# OS command helpers
# -----------------------------
def run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def run_osascript(script: str, timeout: int = 10) -> Tuple[int, str, str]:
    return run_cmd(["osascript", "-e", script], timeout=timeout)


def escape_applescript_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class Backoff:
    max_retries: int = 3
    base_delay_s: float = 0.3
    max_delay_s: float = 2.0

    def sleep(self, attempt: int) -> None:
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        jitter = (attempt % 3) * 0.05
        time.sleep(delay + jitter)


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None

    s = cleaned[start:]
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[: i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


# -----------------------------
# OCR helpers
# -----------------------------
def _safe_text(s: Any) -> str:
    return str(s or "").replace("\n", " ").strip()


def _normalize_text(s: str) -> str:
    s = _safe_text(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _ocr_contains(observation: Dict[str, Any], needle: str) -> bool:
    needle_n = _normalize_text(needle)
    if not needle_n:
        return False
    for el in (observation.get("elements") or []):
        if needle_n in _normalize_text(el.get("text") or ""):
            return True
    return False


def _ocr_contains_all(observation: Dict[str, Any], needles: List[str]) -> bool:
    return all(_ocr_contains(observation, n) for n in needles)


def _ocr_contains_any(observation: Dict[str, Any], needles: List[str]) -> bool:
    return any(_ocr_contains(observation, n) for n in needles)


def _norm_point_to_px(pn: List[float], w: int, h: int) -> Tuple[int, int]:
    return int(round(float(pn[0]) * w)), int(round(float(pn[1]) * h))


def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", (s or "").strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:80] if s else "empty"


# -----------------------------
# Agent
# -----------------------------
class MacDesktopAgent:
    """Primitive-first desktop agent for macOS (OCR grounded)."""

    MACOS_MENU_BAR_HEIGHT_PX = 0

    def __init__(
        self,
        backoff: Backoff = Backoff(),
        type_delay_s: float = 0.0,
        post_launch_sleep_s: float = 1.0,
        post_click_sleep_s: float = 0.25,
        anthropic_api_key: Optional[str] = None,
        save_screenshots: bool = True,
        debug_show_clicks: bool = True,
        apply_menubar_offset: bool = False,
        debug_dir: Optional[str] = None,
        ocr_base_url: Optional[str] = None,
        ocr_timeout_s: int = 120,
        ocr_min_conf: float = 0.55,
        click_post_observe_delay_s: float = 0.2,
        validator_use_claude_fallback: bool = True,
        log_observe: bool = False,
        log_observe_full: bool = False,
        auto_focus_before_step: bool = True,
        ocr_click_strategy: str = "lowest",
        click_do_observe_on_target: bool = True,
        use_quartz_click: bool = False,
        workflow_artifacts_dir: Optional[str] = None,
    ):
        self.backoff = backoff
        self.type_delay_s = type_delay_s
        self.post_launch_sleep_s = post_launch_sleep_s
        self.post_click_sleep_s = post_click_sleep_s
        self.click_post_observe_delay_s = click_post_observe_delay_s
        self.kb = Controller()

        self.save_screenshots = save_screenshots
        self.debug_show_clicks = debug_show_clicks
        self.apply_menubar_offset = apply_menubar_offset

        self.ocr_base_url = (ocr_base_url or os.getenv("DESKTOP_OCR_URL", "")).strip()
        self.ocr_timeout_s = int(ocr_timeout_s)
        self.ocr_min_conf = float(ocr_min_conf)

        self.validator_use_claude_fallback = bool(validator_use_claude_fallback)
        self.log_observe = bool(log_observe)
        self.log_observe_full = bool(log_observe_full)
        self.auto_focus_before_step = bool(auto_focus_before_step)

        self.ocr_click_strategy = (ocr_click_strategy or _env_str("DESKTOP_OCR_CLICK_STRATEGY", "lowest")).strip().lower()
        if self.ocr_click_strategy not in {"lowest", "highest", "first"}:
            self.ocr_click_strategy = "lowest"

        self.click_do_observe_on_target = bool(click_do_observe_on_target)
        env_quartz = _env_bool("DESKTOP_USE_QUARTZ_CLICK", False)
        self.use_quartz_click = bool(use_quartz_click or env_quartz) and _QUARTZ_OK

        self.debug_coord = _env_bool("DESKTOP_DEBUG_COORD", False)
        self._last_screenshot_size_px: Optional[Tuple[int, int]] = None

        if anthropic_api_key:
            self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.claude = None

        self.click_use_vision_fallback = _env_bool("DESKTOP_CLICK_VISION_FALLBACK", True) and (self.claude is not None)
        self.click_vision_min_conf = float(_env_float("DESKTOP_CLICK_VISION_MIN_CONF", 0.70))

        self.debug_dir = Path(debug_dir or "/tmp/desktop_agent_debug")
        if self.save_screenshots:
            self.debug_dir.mkdir(exist_ok=True)

        self.workflow_artifacts_dir = Path(workflow_artifacts_dir) if workflow_artifacts_dir else None
        if self.workflow_artifacts_dir:
            self.workflow_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # =============================
    # Screenshot backend
    # =============================
    def _grab_screen_rgb(self) -> Image.Image:
        # pyautogui aligns with the coordinate space used by pyautogui.click().
        img = pyautogui.screenshot()
        return img.convert("RGB")

    def _debug_save_sent_screenshot(self, img: Image.Image, *, step_id: str, label: str, tag: str) -> Optional[str]:
        """Save the exact image used for OCR/Vision, for reproducible debugging."""
        try:
            out_dir = self.workflow_artifacts_dir or self.debug_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            fn = f"sent_{_slug(step_id)}__{_slug(label)}__{_slug(tag)}__{ts}.png"
            path = out_dir / fn
            img.save(path, "PNG")
            logger.debug("[debug/sent] saved %s", path)
            return str(path)
        except Exception as e:
            logger.debug("[debug/sent] save failed: %s", e)
            return None

    # =============================
    # Coordinate mapping (Retina-safe)
    # =============================
    def _to_pyautogui_xy_from_norm(self, click_norm: List[float]) -> Tuple[int, int]:
        sw, sh = pyautogui.size()
        x = int(round(float(click_norm[0]) * sw))
        y = int(round(float(click_norm[1]) * sh))
        return x, y

    def _to_pyautogui_xy_from_screenshot_px(self, xy_px: Tuple[int, int], screenshot_wh: Tuple[int, int]) -> Tuple[int, int]:
        sw, sh = pyautogui.size()
        W, H = screenshot_wh
        if W <= 0 or H <= 0:
            return xy_px
        x = int(round(xy_px[0] * (sw / float(W))))
        y = int(round(xy_px[1] * (sh / float(H))))
        return x, y

    # =============================
    # Public runner API
    # =============================
    def run_workflow_steps(self, steps: List[Dict[str, Any]], ctx: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ctx = ctx or {}
        results: List[Dict[str, Any]] = []
        for s in steps:
            if s.get("agent") != "desktop":
                continue
            res = self.run_primitive_step(s, ctx)
            results.append(res)
            if not res.get("ok"):
                break
        return results

    def run_primitive_step(self, step: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        step_id = step.get("step_id") or step.get("id") or "unknown"
        kind = (step.get("type") or step.get("kind") or "").strip()
        summary = step.get("summary") or ""

        max_retries = int(step.get("retries", self.backoff.max_retries))
        app_name = step.get("app_name") or ""
        bundle_id = step.get("bundle_id") or ""

        logger.info("[execute] step=%s kind=%s summary=%s", step_id, kind, summary)

        # Focus is a best-effort precondition for most steps.
        if self.auto_focus_before_step and kind not in {"desktop_focus_app", "desktop_launch_fullscreen"} and (app_name or bundle_id):
            info = self.get_active_app_info()
            is_match = (info.get("bundle_id") == bundle_id) if bundle_id else (info.get("name") == app_name)
            if not is_match:
                logger.info("[focus] expected=%s current=%s", (bundle_id or app_name), (info.get("bundle_id") or info.get("name")))
                if app_name:
                    self.launch_fullscreen(app_name)
                else:
                    self.focus_app(app_name=app_name, bundle_id=bundle_id)
                time.sleep(0.3)

        obs_before = self.observe_screen() if self.ocr_base_url else None

        def _do() -> Dict[str, Any]:
            if kind in {"desktop_focus_app", "desktop_launch_fullscreen"}:
                params = (step.get("payload") or {}).get("params") or step.get("params") or {}
                an = params.get("app_name") or app_name
                if an:
                    return self.launch_fullscreen(an)
                return self.focus_app(app_name=an, bundle_id=params.get("bundle_id") or bundle_id)

            if kind == "desktop_click":
                return self.primitive_click(step)

            if kind == "desktop_type_text":
                payload = step.get("payload") or {}
                txt = payload.get("joined_text") or ""
                return self.primitive_type_text(txt)

            if kind == "desktop_hotkey":
                payload = step.get("payload") or {}
                hotkeys = (payload.get("hotkeys") or [])
                if hotkeys:
                    combo = str(hotkeys[0].get("combo") or "")
                    keys = self._parse_combo_to_keys(combo)
                    return self.primitive_hotkey(keys)
                keys = (payload.get("keys") or [])
                return self.primitive_hotkey(keys)

            if kind == "desktop_key":
                payload = step.get("payload") or {}
                keys = (payload.get("keys") or [])
                return self.primitive_press_keys(keys)

            if kind == "desktop_scroll":
                payload = step.get("payload") or {}
                scrolls = payload.get("scrolls") or []
                if not scrolls:
                    return self._ok("No scrolls (noop)", {}, t0)
                dy = sum(float(s.get("dy") or 0) for s in scrolls)
                dx = sum(float(s.get("dx") or 0) for s in scrolls)
                return self.primitive_scroll(dx=dx, dy=dy)

            return self._fail(ERROR_UNKNOWN, f"Unsupported desktop kind: {kind}", {"kind": kind}, t0)

        last = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info("[retry] step=%s attempt=%d/%d", step_id, attempt + 1, max_retries + 1)

            exec_res = _do()
            if not exec_res.get("ok"):
                last = exec_res
                if attempt < max_retries:
                    self.backoff.sleep(attempt)
                    continue
                break

            obs_after = self.observe_screen() if self.ocr_base_url else None
            time.sleep(self.click_post_observe_delay_s)

            verify_hint = step.get("verify_hint") or self._default_verify_hint_for_step(step)
            vres = self.validate_step(verify_hint, step=step, obs_before=obs_before, obs_after=obs_after)

            if vres.get("ok"):
                dt = time.time() - t0
                logger.info("[validate] step=%s ok duration=%.2fs", step_id, dt)
                return {
                    **exec_res,
                    "step_id": step_id,
                    "executor": "desktop",
                    "kind": kind,
                    "attempt": attempt + 1,
                    "validation": vres.get("validation"),
                }

            logger.info("[validate] step=%s fail: %s", step_id, vres.get("message"))
            last = {
                "ok": False,
                "error_type": ERROR_VERIFICATION,
                "message": f"Validation failed: {vres.get('message')}",
                "evidence": self._evidence(),
                "extracted": {
                    "verify_hint": verify_hint,
                    "validation": vres.get("validation"),
                    "exec": exec_res.get("extracted"),
                },
                "timing_ms": int((time.time() - t0) * 1000),
            }

            if attempt < max_retries:
                self.backoff.sleep(attempt)
                continue
            break

        last = last or self._fail(ERROR_UNKNOWN, "unknown failure", {}, t0)
        last["step_id"] = step_id
        last["executor"] = "desktop"
        last["kind"] = kind
        last["attempt"] = max_retries + 1
        return last

    # =============================
    # App lifecycle / focus
    # =============================
    def launch_fullscreen(self, app_name: str) -> Dict[str, Any]:
        t0 = time.time()
        if not app_name:
            return self._fail(ERROR_FOCUS, "launch_fullscreen requires app_name", {}, t0)

        try:
            rc, out, err = run_cmd(["open", "-a", app_name], timeout=10)
            if rc != 0:
                return self._fail(ERROR_FOCUS, f"Failed to launch {app_name}: {err or out}", {"app_name": app_name}, t0)

            activate_script = f'tell application "{escape_applescript_str(app_name)}" to activate'
            rc2, out2, err2 = run_osascript(activate_script, timeout=10)
            if rc2 != 0:
                return self._fail(ERROR_FOCUS, f"Failed to activate {app_name}: {err2 or out2}", {"app_name": app_name}, t0)

            time.sleep(0.5)

            maximize_script = f'''
tell application "System Events"
    tell process "{escape_applescript_str(app_name)}"
        try
            tell window 1
                set position to {{0, {self.MACOS_MENU_BAR_HEIGHT_PX}}}
                tell application "Finder"
                    set screenBounds to bounds of window of desktop
                    set screenWidth to item 3 of screenBounds
                    set screenHeight to item 4 of screenBounds
                end tell
                set size to {{screenWidth, screenHeight - {self.MACOS_MENU_BAR_HEIGHT_PX}}}
            end tell
        end try
    end tell
end tell
'''
            rc3, out3, err3 = run_osascript(maximize_script, timeout=10)
            if rc3 != 0:
                logger.debug("[launch] maximize warning: %s", (err3 or out3))

            time.sleep(self.post_launch_sleep_s)

            active = self.get_active_app_info()
            if active.get("name") != app_name:
                return self._fail(
                    ERROR_FOCUS,
                    f"App not frontmost after launch: expected {app_name}, got {active.get('name')}",
                    {"expected": app_name, "actual": active},
                    t0,
                )

            return self._ok("Launched and maximized", {"app_name": app_name, "active": active}, t0)

        except Exception as e:
            return self._fail(ERROR_FOCUS, f"Launch error: {e}", {"app_name": app_name}, t0)

    def focus_app(self, app_name: str, bundle_id: str = "") -> Dict[str, Any]:
        if app_name:
            return self.launch_fullscreen(app_name)

        t0 = time.time()
        if not bundle_id:
            return self._fail(ERROR_FOCUS, "focus_app requires app_name or bundle_id", {}, t0)

        try:
            rc, out, err = run_cmd(["open", "-b", bundle_id], timeout=10)
            if rc != 0:
                return self._fail(ERROR_FOCUS, f"Failed to open {bundle_id}: {err or out}", {"bundle_id": bundle_id}, t0)
            time.sleep(0.3)
            active = self.get_active_app_info()
            return self._ok("Focused app", {"bundle_id": bundle_id, "active": active}, t0)
        except Exception as e:
            return self._fail(ERROR_FOCUS, f"focus_app error: {e}", {"bundle_id": bundle_id}, t0)

    # =============================
    # OCR calls on PIL images
    # =============================
    def _call_click_target_on_pil(self, image: Image.Image, label: str) -> Dict[str, Any]:
        endpoint = self.ocr_base_url.rstrip("/") + "/click_target"
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("screenshot.png", buf, "image/png")}
        data = {"label": label, "strategy": self.ocr_click_strategy, "return_candidates": "true"}
        resp = requests.post(endpoint, files=files, data=data, timeout=self.ocr_timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _call_observe_on_pil(self, image: Image.Image) -> Dict[str, Any]:
        endpoint = self.ocr_base_url.rstrip("/") + "/observe"
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("screenshot.png", buf, "image/png")}
        resp = requests.post(endpoint, files=files, timeout=self.ocr_timeout_s)
        resp.raise_for_status()
        return resp.json()

    # =============================
    # Click selection
    # =============================
    def _choose_click_from_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        screen_w: int,
        screen_h: int,
        rec_xy: Optional[Tuple[int, int]],
        prefer_top_region: bool = True,
        top_region_y_norm: float = 0.60,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        def cand_conf(c: Dict[str, Any]) -> float:
            try:
                return float(c.get("conf") or c.get("confidence") or 0.0)
            except Exception:
                return 0.0

        def cand_click_norm(c: Dict[str, Any]) -> Optional[List[float]]:
            cn = c.get("click_norm")
            if cn and isinstance(cn, list) and len(cn) == 2:
                try:
                    return [float(cn[0]), float(cn[1])]
                except Exception:
                    return None
            bn = c.get("bbox_norm")
            if bn and isinstance(bn, list) and len(bn) == 4:
                try:
                    x0, y0, x1, y1 = map(float, bn)
                    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
                except Exception:
                    return None
            return None

        enriched: List[Dict[str, Any]] = []
        for c in candidates:
            cn = cand_click_norm(c)
            if not cn:
                continue
            sx, sy = _norm_point_to_px(cn, screen_w, screen_h)
            enriched.append({"c": c, "conf": cand_conf(c), "click_norm": cn, "screen_px": (sx, sy)})

        if not enriched:
            return None

        top_pref = []
        if prefer_top_region:
            top_pref = [e for e in enriched if e["click_norm"][1] <= top_region_y_norm]

        pool = top_pref if top_pref else enriched
        pool.sort(key=lambda e: e["conf"], reverse=True)
        best = pool[0]

        # Tie-break near the recorded click location (if available).
        if rec_xy and len(pool) > 1:
            best_conf = best["conf"]
            close = [e for e in pool if abs(e["conf"] - best_conf) <= 0.03]
            if len(close) > 1:
                def dist(e):
                    dx = e["screen_px"][0] - rec_xy[0]
                    dy = e["screen_px"][1] - rec_xy[1]
                    return math.hypot(dx, dy)
                close.sort(key=dist)
                best = close[0]

        return {
            "chosen": best["c"],
            "confidence": best["conf"],
            "click_norm": best["click_norm"],
            "screen_px": best["screen_px"],
            "pool_size": len(pool),
            "used_top_region": bool(top_pref),
            "top_region_y_norm": top_region_y_norm,
        }

    # =============================
    # Vision helpers (fallback)
    # =============================
    def _pil_to_b64_png(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _claude_find_click_xy_on_image(
        self,
        image: Image.Image,
        *,
        element_description: str,
        min_conf: float = 0.70,
    ) -> Optional[Dict[str, Any]]:
        if not self.claude:
            return None

        model = _env_str("DESKTOP_CLICK_VISION_MODEL", "claude-sonnet-4-20250514")
        b64 = self._pil_to_b64_png(image)

        prompt = f"""Find this UI element in the screenshot:

Element: {element_description}

Return JSON only:
{{
  "found": true/false,
  "x": <center_x>,
  "y": <center_y>,
  "left": <left>,
  "top": <top>,
  "right": <right>,
  "bottom": <bottom>,
  "confidence": 0.0-1.0,
  "reasoning": "<brief explanation>"
}}"""
        try:
            resp = self.claude.messages.create(
                model=model,
                max_tokens=450,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            raw = "".join([c.text for c in resp.content if hasattr(c, "text")])
            j = _extract_first_json(raw) or {}
            if not j.get("found"):
                return None
            conf = float(j.get("confidence") or 0.0)
            if conf < float(min_conf):
                return None
            return {
                "x": int(j["x"]),
                "y": int(j["y"]),
                "confidence": conf,
                "bounds": {"left": j.get("left"), "top": j.get("top"), "right": j.get("right"), "bottom": j.get("bottom")},
                "reasoning": j.get("reasoning") or "",
                "raw": j,
            }
        except Exception as e:
            logger.debug("[vision_click] claude error: %s", e)
            return None

    # =============================
    # Primitive: click
    # =============================
    def _click_screen_px(self, x: int, y: int) -> None:
        if self.apply_menubar_offset and self.MACOS_MENU_BAR_HEIGHT_PX:
            y = y - self.MACOS_MENU_BAR_HEIGHT_PX

        if self.debug_show_clicks:
            self._show_click_location(x, y, duration=0.8)

        if self.use_quartz_click and _QUARTZ_OK:
            src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
            move = CGEventCreateMouseEvent(src, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, move)
            time.sleep(0.01)
            down = CGEventCreateMouseEvent(src, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
            CGEventSetIntegerValueField(down, kCGMouseEventClickState, 1)
            CGEventPost(kCGHIDEventTap, down)
            time.sleep(0.01)
            up = CGEventCreateMouseEvent(src, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
            CGEventSetIntegerValueField(up, kCGMouseEventClickState, 1)
            CGEventPost(kCGHIDEventTap, up)
        else:
            pyautogui.click(x, y)

    def primitive_click(self, step: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        step_id = step.get("step_id") or step.get("id") or "unknown"

        payload = step.get("payload") or {}
        target = payload.get("target") or {}

        label = _safe_text(
            target.get("text_hint")
            or payload.get("ocr_label")
            or payload.get("label")
            or payload.get("click_text")
        ).strip()

        vision_desc = _safe_text(
            target.get("description")
            or payload.get("element_description")
            or payload.get("description")
            or (f"{label} button" if label else "")
            or (step.get("summary") or "")
        ).strip()

        clicks = payload.get("clicks") or []
        rec_xy = None
        if clicks:
            rec = (clicks[0].get("click_xy_px") or {})
            if rec.get("x") is not None and rec.get("y") is not None:
                rec_xy = (int(rec["x"]), int(rec["y"]))

        img = self._grab_screen_rgb()
        W, H = img.size
        sw, sh = pyautogui.size()

        if self.debug_coord:
            logger.debug("[coord] screenshot=(%d,%d) pyautogui=(%d,%d) scale=(%.3f,%.3f)", W, H, sw, sh, sw / float(W), sh / float(H))

        logger.info("[execute] click label=%r", label if label else "<empty>")
        logger.debug("vision_desc=%r rec_xy=%s", vision_desc, rec_xy)

        # Save screenshot being sent for New Document (exact bytes used downstream).
        if _normalize_text(label) == "new document":
            self._debug_save_sent_screenshot(img, step_id=step_id, label="New_Document", tag="fullscreen")

        # Empty label: rely on recorded coords only (scaled into pyautogui space).
        if not label:
            if rec_xy:
                rx, ry = self._to_pyautogui_xy_from_screenshot_px(rec_xy, (W, H))
                logger.info("[execute] click recorded_xy->(%d,%d)", rx, ry)
                self._click_screen_px(rx, ry)
                time.sleep(self.post_click_sleep_s)
                return self._ok("Clicked (recorded_empty_label_scaled)", {"recorded_xy": list(rec_xy), "clicked_pyautogui_xy": [rx, ry]}, t0)
            return self._fail(ERROR_CLICK, "Missing label and no recorded coords", {}, t0)

        # Optional pre-check: whether OCR sees the label (debug signal, not gating).
        if self.click_do_observe_on_target and self.ocr_base_url:
            try:
                if _normalize_text(label) == "new document":
                    self._debug_save_sent_screenshot(img, step_id=step_id, label="New_Document", tag="observe")
                obs = self._call_observe_on_pil(img)
                if obs.get("ok"):
                    present = _ocr_contains(obs, label)
                    logger.debug("[afford] observe label_present=%s", present)
            except Exception as e:
                logger.debug("[afford] observe failed: %s", e)

        # OCR candidate-based click (preferred).
        if self.ocr_base_url:
            try:
                ct = self._call_click_target_on_pil(img, label=label)
                if ct.get("ok"):
                    candidates = ct.get("candidates") or []
                    logger.debug("[afford] candidates=%d strategy=%s", len(candidates), self.ocr_click_strategy)

                    chosen = self._choose_click_from_candidates(
                        candidates,
                        screen_w=W,
                        screen_h=H,
                        rec_xy=rec_xy,
                        prefer_top_region=True,
                        top_region_y_norm=float(payload.get("ocr_top_region_y_norm") or 0.60),
                    )
                    if chosen:
                        click_norm = chosen["click_norm"]
                        cx, cy = self._to_pyautogui_xy_from_norm(click_norm)
                        logger.info("[execute] click ocr_xy=(%d,%d)", cx, cy)
                        logger.debug(
                            "click_norm=%s conf=%.3f screenshot_px=%s pool=%d top_region=%s",
                            click_norm,
                            chosen["confidence"],
                            chosen["screen_px"],
                            chosen["pool_size"],
                            chosen["used_top_region"],
                        )
                        self._click_screen_px(cx, cy)
                        time.sleep(self.post_click_sleep_s)
                        return self._ok(
                            "Clicked (ocr_candidates_chosen_scaled)",
                            {
                                "mode": "ocr_candidates_chosen_scaled",
                                "label": label,
                                "click_norm": click_norm,
                                "clicked_pyautogui_xy": [cx, cy],
                                "chosen_confidence": chosen["confidence"],
                                "used_top_region": chosen["used_top_region"],
                                "top_region_y_norm": chosen["top_region_y_norm"],
                                "pool_size": chosen["pool_size"],
                                "click_target_raw": ct,
                                "fallback_recorded_xy": list(rec_xy) if rec_xy else None,
                                "screenshot_size": [W, H],
                                "pyautogui_size": [sw, sh],
                            },
                            t0,
                        )
            except Exception as e:
                logger.debug("[afford] click_target failed: %s", e)

        # Vision fallback: returns screenshot-space px; convert to pyautogui space.
        if self.click_use_vision_fallback and vision_desc and vision_desc.lower() not in {"click in textedit", "click"}:
            if _normalize_text(label) == "new document":
                self._debug_save_sent_screenshot(img, step_id=step_id, label="New_Document", tag="vision")
            logger.info("[execute] click vision_fallback label=%r", label)
            v = self._claude_find_click_xy_on_image(img, element_description=vision_desc, min_conf=self.click_vision_min_conf)
            if v:
                sx, sy = int(v["x"]), int(v["y"])
                cx, cy = self._to_pyautogui_xy_from_screenshot_px((sx, sy), (W, H))
                logger.info("[execute] click vision_xy=(%d,%d)", cx, cy)
                logger.debug("screenshot_xy=(%d,%d) conf=%.2f bounds=%s", sx, sy, v["confidence"], v["bounds"])
                self._click_screen_px(cx, cy)
                time.sleep(self.post_click_sleep_s)
                return self._ok(
                    "Clicked (vision_fallback_scaled)",
                    {
                        "mode": "vision_fallback_scaled",
                        "vision_desc": vision_desc,
                        "screenshot_xy": [sx, sy],
                        "clicked_pyautogui_xy": [cx, cy],
                        "confidence": v["confidence"],
                        "bounds": v["bounds"],
                        "reasoning": v["reasoning"],
                        "fallback_recorded_xy": list(rec_xy) if rec_xy else None,
                        "screenshot_size": [W, H],
                        "pyautogui_size": [sw, sh],
                    },
                    t0,
                )

        # Final fallback: recorded click (scaled).
        if rec_xy:
            rx, ry = self._to_pyautogui_xy_from_screenshot_px(rec_xy, (W, H))
            logger.info("[execute] click fallback_recorded_xy=(%d,%d)", rx, ry)
            self._click_screen_px(rx, ry)
            time.sleep(self.post_click_sleep_s)
            return self._ok(
                "Clicked (recorded_final_fallback_scaled)",
                {"recorded_xy": list(rec_xy), "clicked_pyautogui_xy": [rx, ry], "label": label},
                t0,
            )

        return self._fail(
            ERROR_ELEMENT_NOT_FOUND,
            "Could not resolve click target (ocr+vision failed, no recorded coords)",
            {"label": label, "vision_desc": vision_desc},
            t0,
        )

    # =============================
    # Typing / hotkeys / scroll
    # =============================
    def primitive_type_text(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info("[execute] type chars=%d", len(text))
            logger.debug("preview=%r", text[:80])
            for ch in text:
                self.kb.type(ch)
                if self.type_delay_s > 0:
                    time.sleep(self.type_delay_s)
            return self._ok("Typed text", {"chars": len(text), "preview": text[:80]}, t0)
        except Exception as e:
            logger.error("[execute] type failed: %s", e)
            return self._fail(ERROR_TYPE, f"Type error: {e}", {"text_preview": text[:120]}, t0)

    def primitive_hotkey(self, keys: List[str]) -> Dict[str, Any]:
        t0 = time.time()
        if not keys:
            return self._ok("No hotkey (noop)", {}, t0)
        try:
            logger.info("[execute] hotkey keys=%s", keys)
            ks = [self._map_key(k) for k in keys]
            for k in ks[:-1]:
                self.kb.press(k)
            self.kb.press(ks[-1])
            self.kb.release(ks[-1])
            for k in reversed(ks[:-1]):
                self.kb.release(k)
            time.sleep(0.15)
            return self._ok("Hotkey executed", {"keys": keys}, t0)
        except Exception as e:
            logger.error("[execute] hotkey failed: %s", e)
            return self._fail(ERROR_HOTKEY, f"Hotkey error: {e}", {"keys": keys}, t0)

    def primitive_press_keys(self, keys_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info("[execute] keys count=%d", len(keys_payload))
            for kd in keys_payload:
                k = kd.get("key")
                if not k:
                    continue
                mapped = self._map_key(str(k))
                self.kb.press(mapped)
                self.kb.release(mapped)
                time.sleep(0.05)
            return self._ok("Keys pressed", {"count": len(keys_payload)}, t0)
        except Exception as e:
            logger.error("[execute] key_press failed: %s", e)
            return self._fail(ERROR_HOTKEY, f"Key press error: {e}", {"keys": keys_payload}, t0)

    def primitive_scroll(self, dx: float, dy: float) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info("[execute] scroll dx=%.1f dy=%.1f", dx, dy)
            pyautogui.hscroll(int(dx))
            pyautogui.scroll(int(dy))
            time.sleep(0.1)
            return self._ok("Scrolled", {"dx": dx, "dy": dy}, t0)
        except Exception as e:
            logger.error("[execute] scroll failed: %s", e)
            return self._fail(ERROR_UNKNOWN, f"Scroll error: {e}", {"dx": dx, "dy": dy}, t0)

    # =============================
    # Observer (/observe) full-screen
    # =============================
    def observe_screen(self) -> Dict[str, Any]:
        t0 = time.time()
        if not self.ocr_base_url:
            return {"ok": False, "error_type": ERROR_OBSERVE, "message": "DESKTOP_OCR_URL not set"}

        try:
            img_path, (w, h) = self._take_screenshot_file(suffix="observe")
            self._last_screenshot_size_px = (w, h)

            endpoint = self.ocr_base_url.rstrip("/") + "/observe"
            with open(img_path, "rb") as f:
                files = {"file": (Path(img_path).name, f, "image/png")}
                resp = requests.post(endpoint, files=files, timeout=self.ocr_timeout_s)

            resp.raise_for_status()
            obs = resp.json()
            if not obs.get("ok"):
                return self._fail(ERROR_OBSERVE, "observe returned ok=false", {"obs": obs}, t0)

            obs["_local"] = {"screenshot_path": str(img_path), "w": w, "h": h}
            out = self._ok("observe ok", {"observation": obs}, t0)

            if self.log_observe or self.log_observe_full:
                self._log_observe_payload(out)

            return out

        except Exception as e:
            logger.debug("[observe] failed: %s", e)
            return self._fail(ERROR_OBSERVE, f"observe error: {e}", {}, t0)

    def _log_observe_payload(self, observe_result: Dict[str, Any]) -> None:
        try:
            obs = (observe_result.get("extracted") or {}).get("observation") or {}
            elements = obs.get("elements") or []
            top = sorted(elements, key=lambda e: float(e.get("confidence", 0.0)), reverse=True)[:12]
            summary = {
                "ok": obs.get("ok"),
                "elements": len(elements),
                "top_conf": float(top[0].get("confidence", 0.0)) if top else 0.0,
                "top_texts": [{"text": (t.get("text") or "")[:80], "conf": float(t.get("confidence", 0.0))} for t in top],
            }
            if self.log_observe_full:
                logger.debug("[observe] full=%s", json.dumps(obs, indent=2)[:12000])
            else:
                logger.debug("[observe] summary=%s", json.dumps(summary, indent=2))
        except Exception:
            pass

    # =============================
    # Validator
    # =============================
    def validate_step(
        self,
        verify_hint: Dict[str, Any],
        *,
        step: Dict[str, Any],
        obs_before: Optional[Dict[str, Any]],
        obs_after: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not verify_hint:
            return {"ok": True, "message": "no verify_hint (skipped)", "validation": {"skipped": True}}

        timeout_ms = int(verify_hint.get("timeout_ms") or 5000)
        deadline = time.time() + (timeout_ms / 1000.0)

        def _get_obs_payload(o: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not o or not o.get("ok"):
                return None
            return (o.get("extracted") or {}).get("observation") or o.get("observation")

        last_obs = _get_obs_payload(obs_after)
        if last_obs is None and self.ocr_base_url:
            last_obs = _get_obs_payload(self.observe_screen())

        checks = verify_hint.get("checks") or []
        strategy = verify_hint.get("strategy") or "unknown"
        vtype = verify_hint.get("type") or "unknown"

        def _eval_checks(observation: Optional[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
            results: List[Dict[str, Any]] = []
            ok = True

            info = self.get_active_app_info()
            active_name = info.get("name") or ""
            active_bid = info.get("bundle_id") or ""

            for c in checks:
                op = str(c.get("op") or "")
                passed = True
                detail: Dict[str, Any] = {"op": op}

                if op == "active_app_is":
                    expected = str(c.get("value") or "").strip()
                    if "." in expected:
                        passed = (not expected) or (expected == active_bid)
                    else:
                        passed = (not expected) or (expected.lower() in active_name.lower())
                    detail.update({"expected": expected, "actual_name": active_name, "actual_bundle_id": active_bid})

                elif op == "ocr_contains":
                    val = str(c.get("value") or "")
                    passed = bool(observation) and _ocr_contains(observation, val)
                    detail.update({"value": val})

                elif op == "ocr_contains_all":
                    vals = c.get("values") or []
                    passed = bool(observation) and _ocr_contains_all(observation, [str(x) for x in vals])
                    detail.update({"values": vals})

                elif op == "ocr_contains_any":
                    vals = c.get("values") or []
                    passed = bool(observation) and _ocr_contains_any(observation, [str(x) for x in vals])
                    detail.update({"values": vals})

                elif op == "optional_ocr_contains":
                    val = str(c.get("value") or "")
                    hit = bool(observation) and _ocr_contains(observation, val)
                    passed = True
                    detail.update({"value": val, "hit": hit, "soft": True})

                else:
                    passed = True
                    detail.update({"note": "unknown_op_soft_pass"})

                detail["passed"] = bool(passed)
                results.append(detail)

                if op.startswith("optional_"):
                    continue
                if not passed:
                    ok = False

            return ok, results

        last_results: List[Dict[str, Any]] = []

        while time.time() <= deadline:
            ok, results = _eval_checks(last_obs)
            last_results = results
            if ok:
                return {"ok": True, "message": f"validated ({vtype}/{strategy})", "validation": {"type": vtype, "strategy": strategy, "results": results}}
            if self.ocr_base_url:
                obs = _get_obs_payload(self.observe_screen())
                if obs:
                    last_obs = obs
            time.sleep(0.25)

        # Claude fallback is a last resort when heuristic checks fail.
        if self.validator_use_claude_fallback and self.claude and self.ocr_base_url:
            try:
                screenshot_path = (((last_obs or {}).get("_local") or {}).get("screenshot_path")) if last_obs else None
                if screenshot_path and os.path.exists(screenshot_path):
                    b64 = self._image_file_to_b64(Path(screenshot_path))
                else:
                    b64 = self._take_screenshot_b64(suffix="validate_fallback")

                prompt = f"""You are validating whether a desktop step succeeded.

Step summary: {step.get("summary") or step.get("type")}
Verify hint JSON: {json.dumps(verify_hint, indent=2)}

Decide if the step likely succeeded based on what you see.
Return JSON only:
{{
  "success": true/false,
  "confidence": 0.0-1.0,
  "reason": "<short>",
  "suggested_fix": "<short or empty>"
}}"""

                resp = self.claude.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=350,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                raw = "".join([c.text for c in resp.content if hasattr(c, "text")])
                j = _extract_first_json(raw) or {}
                if j.get("success") and float(j.get("confidence") or 0) >= 0.7:
                    return {"ok": True, "message": "validated via Claude fallback", "validation": {"type": vtype, "strategy": "claude_fallback", "raw": j}}
            except Exception as e:
                logger.debug("[validate] claude_fallback error: %s", e)

        return {"ok": False, "message": f"verification failed ({vtype}/{strategy})", "validation": {"type": vtype, "strategy": strategy, "results": last_results}}

    def _default_verify_hint_for_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        kind = (step.get("type") or step.get("kind") or "").strip()
        app = step.get("app_name") or ""
        bundle = step.get("bundle_id") or ""
        expected = bundle or app

        if kind == "desktop_type_text":
            joined = ((step.get("payload") or {}).get("joined_text") or "").strip()
            tokens = re.findall(r"[A-Za-z0-9]{4,}", joined)[:2]
            checks = [{"op": "active_app_is", "value": expected}]
            if tokens:
                checks.append({"op": "ocr_contains_all", "values": tokens})
            return {"type": "desktop_type_verified", "strategy": "observe_text_present", "timeout_ms": 5000, "checks": checks}

        if kind == "desktop_hotkey":
            payload = step.get("payload") or {}
            hotkeys = payload.get("hotkeys") or []
            combo = str(hotkeys[0].get("combo") if hotkeys else "").lower().replace(" ", "")
            if combo in ("cmd+s", "command+s"):
                return {
                    "type": "desktop_save_initiated",
                    "strategy": "observe_ui_change",
                    "timeout_ms": 8000,
                    "checks": [
                        {"op": "active_app_is", "value": expected},
                        {"op": "ocr_contains_any", "values": ["Save", "Cancel", "Where:", "Format:", "Name:"]},
                    ],
                }
            return {"type": "desktop_hotkey_verified", "strategy": "observe_active_app_window", "timeout_ms": 3000, "checks": [{"op": "active_app_is", "value": expected}]}

        if kind == "desktop_click":
            return {"type": "desktop_click_verified", "strategy": "observe_active_app_window", "timeout_ms": 3000, "checks": [{"op": "active_app_is", "value": expected}]}

        return {}

    # =============================
    # Screenshot utils
    # =============================
    def _take_screenshot_file(self, suffix: str = "") -> Tuple[str, Tuple[int, int]]:
        screenshot = pyautogui.screenshot()
        w, h = screenshot.size

        if self.save_screenshots:
            timestamp = int(time.time() * 1000)
            filename = f"screenshot_{timestamp}_{suffix}.png" if suffix else f"screenshot_{timestamp}.png"
            screenshot_path = self.debug_dir / filename
            screenshot.save(screenshot_path, "PNG")
            return str(screenshot_path), (w, h)

        temp_path = Path("/tmp/desktop_agent_screenshot.png")
        screenshot.save(temp_path, "PNG")
        return str(temp_path), (w, h)

    def _take_screenshot_b64(self, suffix: str = "") -> str:
        img_path, _ = self._take_screenshot_file(suffix=suffix)
        return self._image_file_to_b64(Path(img_path))

    def _image_file_to_b64(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # =============================
    # Visual Debugging
    # =============================
    def _show_click_location(self, x: int, y: int, duration: float = 1.0):
        if not self.debug_show_clicks:
            return

        def _show_overlay():
            try:
                import tkinter as tk
                root = tk.Tk()
                root.attributes("-alpha", 0.7)
                root.attributes("-topmost", True)
                root.overrideredirect(True)
                size = 60
                root.geometry(f"{size}x{size}+{x-size//2}+{y-size//2}")

                canvas = tk.Canvas(root, width=size, height=size, bg="red", highlightthickness=0)
                canvas.pack()
                canvas.create_oval(10, 10, size - 10, size - 10, outline="white", width=4)
                canvas.create_line(size // 2, 10, size // 2, size - 10, fill="white", width=2)
                canvas.create_line(10, size // 2, size - 10, size // 2, fill="white", width=2)

                root.after(int(duration * 1000), root.destroy)
                root.mainloop()
            except Exception as e:
                logger.debug("[debug_clicks] overlay unavailable: %s", e)

        threading.Thread(target=_show_overlay, daemon=True).start()
        time.sleep(0.10)

    # =============================
    # Active app
    # =============================
    def get_active_app_info(self) -> Dict[str, str]:
        script = r'''
tell application "System Events"
  set p to first application process whose frontmost is true
  set appName to name of p
  try
    set bid to bundle identifier of p
  on error
    set bid to ""
  end try
end tell
return appName & "||" & bid
'''
        rc, out, _ = run_osascript(script, timeout=5)
        if rc != 0:
            return {"name": "", "bundle_id": ""}
        parts = (out or "").split("||")
        name = (parts[0] if len(parts) > 0 else "").strip()
        bid = (parts[1] if len(parts) > 1 else "").strip()
        return {"name": name, "bundle_id": bid}

    def _evidence(self) -> Dict[str, Any]:
        info = self.get_active_app_info()
        return {"active_app": info.get("name"), "active_bundle_id": info.get("bundle_id")}

    # =============================
    # Key mapping + combo parsing
    # =============================
    def _map_key(self, k: str):
        k = (k or "").strip().lower()
        if k in {"cmd", "command", "meta"}:
            return Key.cmd
        if k in {"ctrl", "control"}:
            return Key.ctrl
        if k in {"alt", "option"}:
            return Key.alt
        if k == "shift":
            return Key.shift
        if k in {"enter", "return"}:
            return Key.enter
        if k == "tab":
            return Key.tab
        if k in {"esc", "escape"}:
            return Key.esc
        if k == "space":
            return Key.space
        if len(k) == 1:
            return k
        raise ValueError(f"Unsupported key: {k}")

    def _parse_combo_to_keys(self, combo: str) -> List[str]:
        c = (combo or "").strip()
        if not c:
            return []
        parts = [p.strip() for p in c.split("+") if p.strip()]
        out: List[str] = []
        for p in parts:
            pl = p.lower()
            if pl in ("cmd", "command", "meta"):
                out.append("cmd")
            elif pl in ("ctrl", "control"):
                out.append("ctrl")
            elif pl in ("alt", "option"):
                out.append("alt")
            elif pl == "shift":
                out.append("shift")
            elif pl in ("enter", "return"):
                out.append("enter")
            elif len(pl) == 1:
                out.append(pl)
            else:
                out.append(pl)
        return out

    # =============================
    # Result helpers
    # =============================
    def _ok(self, message: str, extracted: Dict[str, Any], t0: float) -> Dict[str, Any]:
        return {
            "ok": True,
            "error_type": None,
            "message": message,
            "evidence": self._evidence(),
            "extracted": extracted or {},
            "timing_ms": int((time.time() - t0) * 1000),
        }

    def _fail(self, error_type: str, message: str, extracted: Dict[str, Any], t0: float) -> Dict[str, Any]:
        return {
            "ok": False,
            "error_type": error_type,
            "message": message,
            "evidence": self._evidence(),
            "extracted": extracted or {},
            "timing_ms": int((time.time() - t0) * 1000),
        }


# -----------------------------
# Example: run desktop steps from workflow_steps.v1.json
# -----------------------------
def load_workflow_steps(path: str) -> List[Dict[str, Any]]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    steps = data.get("steps") or []
    return [s for s in steps if (s.get("agent") == "desktop")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="recordings/20260107_163715/workflow_steps.v1.json")
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("DESKTOP_MAX_RETRIES", "2")))
    ap.add_argument("--observe-between", action="store_true")
    ap.add_argument("--ocr-url", default=os.getenv("DESKTOP_OCR_URL", ""))
    ap.add_argument("--ocr-timeout", type=int, default=int(os.getenv("DESKTOP_OCR_TIMEOUT_S", "120")))
    ap.add_argument("--ocr-min-conf", type=float, default=float(os.getenv("DESKTOP_OCR_MIN_CONF", "0.55")))
    ap.add_argument("--save-screenshots", action="store_true", default=_env_bool("DESKTOP_SAVE_SCREENSHOTS", True))
    ap.add_argument("--debug-clicks", action="store_true", default=_env_bool("DESKTOP_DEBUG_CLICKS", False))
    ap.add_argument("--debug-dir", default=os.getenv("DESKTOP_DEBUG_DIR", "/tmp/desktop_agent_debug"))
    ap.add_argument("--type-delay", type=float, default=_env_float("DESKTOP_TYPE_DELAY_S", 0.0))
    ap.add_argument("--apply-menubar-offset", action="store_true", default=_env_bool("DESKTOP_APPLY_MENUBAR_OFFSET", False))
    ap.add_argument("--claude-fallback", action="store_true", default=_env_bool("DESKTOP_VALIDATOR_CLAUDE_FALLBACK", True))
    ap.add_argument("--log-observe", action="store_true", default=_env_bool("DESKTOP_LOG_OBSERVE", False))
    ap.add_argument("--log-observe-full", action="store_true", default=_env_bool("DESKTOP_LOG_OBSERVE_FULL", False))
    ap.add_argument("--no-auto-focus", action="store_true")
    ap.add_argument("--ocr-click-strategy", default=_env_str("DESKTOP_OCR_CLICK_STRATEGY", "lowest"),
                    choices=["lowest", "highest", "first"])
    ap.add_argument("--no-observe-on-target", action="store_true")

    args = ap.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
    ocr_url = (args.ocr_url or "").strip() or None

    workflow_path = Path(args.workflow)
    workflow_dir = workflow_path.parent
    workflow_dir.mkdir(parents=True, exist_ok=True)

    agent = MacDesktopAgent(
        anthropic_api_key=api_key,
        ocr_base_url=ocr_url,
        ocr_timeout_s=args.ocr_timeout,
        ocr_min_conf=args.ocr_min_conf,
        save_screenshots=bool(args.save_screenshots),
        debug_show_clicks=bool(args.debug_clicks),
        apply_menubar_offset=bool(args.apply_menubar_offset),
        debug_dir=args.debug_dir,
        type_delay_s=float(args.type_delay),
        backoff=Backoff(max_retries=int(args.max_retries)),
        validator_use_claude_fallback=bool(args.claude_fallback),
        log_observe=bool(args.log_observe),
        log_observe_full=bool(args.log_observe_full),
        auto_focus_before_step=(not bool(args.no_auto_focus)),
        ocr_click_strategy=args.ocr_click_strategy,
        click_do_observe_on_target=(not bool(args.no_observe_on_target)),
        use_quartz_click=_env_bool("DESKTOP_USE_QUARTZ_CLICK", False),
        # Save “sent-to-ocr” screenshots alongside workflow_steps.v1.json
        workflow_artifacts_dir=str(workflow_dir),
    )

    steps = load_workflow_steps(args.workflow)
    if not steps:
        raise SystemExit(f"No desktop steps found in workflow: {args.workflow}")

    ctx: Dict[str, Any] = {"vars": {}}

    for s in steps:
        res = agent.run_primitive_step(s, ctx)
        print(json.dumps(res, indent=2))
        if not res.get("ok"):
            break

        if args.observe_between and ocr_url:
            obs = agent.observe_screen()
            o = (obs.get("extracted") or {}).get("observation") or {}
            elems = (o.get("elements") or [])
            print("OBS:", json.dumps({"ok": o.get("ok"), "elements": len(elems), "active": agent.get_active_app_info()}, indent=2))

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
