from __future__ import annotations

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

from pynput.keyboard import Controller, Key
import pyautogui
from PIL import ImageGrab
import anthropic
from dotenv import load_dotenv

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


# -----------------------------
# Setup logging
# -----------------------------
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
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


logger = setup_logger("MacDesktopAgent", level=logging.DEBUG)

# -----------------------------
# Error types
# -----------------------------
ERROR_FOCUS = "focus_error"
ERROR_CLICK = "click_error"
ERROR_TYPE = "type_error"
ERROR_HOTKEY = "hotkey_error"
ERROR_ELEMENT_NOT_FOUND = "element_not_found"
ERROR_VERIFICATION = "verification_failed"
ERROR_UNKNOWN = "unknown_error"


def run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    logger.debug(f"Running command: {' '.join(cmd)}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    logger.debug(
        f"Command result: rc={p.returncode}, stdout='{p.stdout.strip()[:100]}', stderr='{p.stderr.strip()[:100]}'"
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def run_osascript(script: str, timeout: int = 10) -> Tuple[int, str, str]:
    logger.debug(f"Running AppleScript: {script[:200]}...")
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
        total_sleep = delay + jitter
        logger.debug(f"Retry backoff: attempt={attempt}, sleeping={total_sleep:.2f}s")
        time.sleep(total_sleep)


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Balanced JSON object scan
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


class MacDesktopAgent:
    """
    Vision-first desktop agent with fullscreen app strategy.
    """

    # Correct: menu bar height ~25px (positive), we subtract it from screenshot coordinates when clicking.
    MACOS_MENU_BAR_HEIGHT_PX = -25

    def __init__(
        self,
        backoff: Backoff = Backoff(),
        type_delay_s: float = 0.0,
        post_launch_sleep_s: float = 1.0,
        post_click_sleep_s: float = 0.3,
        anthropic_api_key: Optional[str] = None,
        save_screenshots: bool = True,
        debug_show_clicks: bool = True,
        apply_menubar_offset: bool = True,
        debug_dir: Optional[str] = None,
    ):
        self.backoff = backoff
        self.type_delay_s = type_delay_s
        self.post_launch_sleep_s = post_launch_sleep_s
        self.post_click_sleep_s = post_click_sleep_s
        self.kb = Controller()
        self.save_screenshots = save_screenshots
        self.debug_show_clicks = debug_show_clicks
        self.apply_menubar_offset = apply_menubar_offset

        if anthropic_api_key:
            logger.info("Initializing Claude Vision API")
            self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            logger.warning("No Anthropic API key provided - vision features disabled")
            self.claude = None

        if self.save_screenshots:
            self.debug_dir = Path(debug_dir or "/tmp/desktop_agent_debug")
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"Debug screenshots will be saved to: {self.debug_dir}")

    def run_step(self, step: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        step_id = step.get("id", "unknown")
        action = (step.get("action") or "").strip()
        args = step.get("args") or {}
        max_retries = int(step.get("retries", self.backoff.max_retries))

        logger.info("=" * 80)
        logger.info(f"EXECUTING STEP: {step_id}")
        logger.info(f"Action: {action}")
        logger.info(f"Args: {args}")
        logger.info(f"Max retries: {max_retries}")
        logger.info("=" * 80)

        def _do_action() -> Dict[str, Any]:
            if action == "launch_fullscreen":
                return self.launch_fullscreen(args["app_name"])
            if action == "click_element":
                return self.click_element(
                    args["element_description"],
                    offset_x=args.get("click_offset_x", 0),
                    offset_y=args.get("click_offset_y", 0),
                )
            if action == "type_text":
                return self.type_text(args.get("text", ""))
            if action == "verify_visible":
                return self.verify_visible(args["element_description"])
            if action == "hotkey":
                return self.hotkey(args.get("keys", []))
            if action == "sleep":
                duration = float(args.get("duration", 1.0))
                logger.info(f"Sleeping for {duration}s...")
                time.sleep(duration)
                return self._ok(f"Slept for {duration}s", {"duration": duration}, t0)

            return self._fail(ERROR_UNKNOWN, f"unknown action: {action}", {}, t0)

        last = None
        for attempt in range(max_retries + 1):
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            res = _do_action()

            logger.info(f"Result: ok={res.get('ok')}, error_type={res.get('error_type')}, message={res.get('message')}")
            logger.debug(f"Full result: {res}")

            if res.get("ok"):
                res["step_id"] = step_id
                res["executor"] = "desktop"
                res["action"] = action
                res["attempt"] = attempt + 1
                return res

            last = res
            if attempt < max_retries:
                self.backoff.sleep(attempt)
                continue
            break

        last = last or self._fail(ERROR_UNKNOWN, "unknown failure", {}, t0)
        last["step_id"] = step_id
        last["executor"] = "desktop"
        last["action"] = action
        last["attempt"] = max_retries + 1
        return last

    # -----------------------------
    # Visual Debugging
    # -----------------------------
    def _show_click_location(self, x: int, y: int, duration: float = 1.5):
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
                logger.warning(f"Could not show click debug overlay: {e}")

        thread = threading.Thread(target=_show_overlay, daemon=True)
        thread.start()
        time.sleep(0.2)

    # -----------------------------
    # Core Actions
    # -----------------------------
    def launch_fullscreen(self, app_name: str) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"Launching app: {app_name}")

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
                logger.warning(f"Failed to maximize window: {err3 or out3} (continuing anyway)")

            time.sleep(self.post_launch_sleep_s)

            active_app = self.get_active_app()
            logger.info(f"Active app after launch: {active_app}")

            if active_app != app_name:
                return self._fail(ERROR_FOCUS, f"App not frontmost: expected {app_name}, got {active_app}", {"expected": app_name, "actual": active_app}, t0)

            return self._ok(f"Launched and maximized {app_name}", {"app_name": app_name, "active_app": active_app}, t0)

        except Exception as e:
            logger.exception(f"Exception during launch: {e}")
            return self._fail(ERROR_FOCUS, f"Launch error: {e}", {"app_name": app_name}, t0)

    def click_element(self, element_description: str, offset_x: int = 0, offset_y: int = 0) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"Looking for element: '{element_description}'")

        if not self.claude:
            return self._fail(ERROR_UNKNOWN, "Vision API not configured", {}, t0)

        try:
            screenshot_b64 = self._take_screenshot(suffix=f"click_{element_description.replace(' ', '_')[:30]}")

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
  "confidence": 0.0-1.0
}}"""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=450,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            raw_text = "".join([c.text for c in response.content if hasattr(c, "text")])
            result = _extract_first_json(raw_text) or {}
            logger.info(f"Vision result: {result}")

            if not result.get("found") or result.get("confidence", 0) < 0.7:
                return self._fail(
                    ERROR_ELEMENT_NOT_FOUND,
                    f"Element not found or low confidence: {element_description}",
                    {"element": element_description, "raw": raw_text[:800]},
                    t0,
                )

            click_x = int(result["x"]) + int(offset_x)
            click_y = int(result["y"]) + int(offset_y)

            if self.apply_menubar_offset:
                original_y = click_y
                click_y = click_y - self.MACOS_MENU_BAR_HEIGHT_PX
                logger.info(f"Menu bar compensation: y {original_y} -> {click_y} (subtract {self.MACOS_MENU_BAR_HEIGHT_PX}px)")

            self._show_click_location(click_x, click_y, duration=1.5)

            logger.info(f"Clicking at ({click_x}, {click_y})")
            pyautogui.click(click_x, click_y)
            time.sleep(self.post_click_sleep_s)

            return self._ok(
                f"Clicked: {element_description}",
                {
                    "element": element_description,
                    "coords": (click_x, click_y),
                    "bounds": {
                        "left": result.get("left"),
                        "top": result.get("top"),
                        "right": result.get("right"),
                        "bottom": result.get("bottom"),
                    },
                    "confidence": result.get("confidence"),
                },
                t0,
            )

        except Exception as e:
            logger.exception(f"Exception during click: {e}")
            return self._fail(ERROR_CLICK, f"Click error: {e}", {"element": element_description}, t0)

    def verify_visible(self, element_description: str) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"Verifying element visible: '{element_description}'")

        if not self.claude:
            return self._ok("Vision verification disabled", {}, t0)

        try:
            screenshot_b64 = self._take_screenshot(suffix=f"verify_{element_description.replace(' ', '_')[:30]}")

            prompt = f"""Is this element visible?

Element: {element_description}

Return JSON only:
{{
  "visible": true/false,
  "confidence": 0.0-1.0
}}"""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=250,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            raw_text = "".join([c.text for c in response.content if hasattr(c, "text")])
            result = _extract_first_json(raw_text) or {}
            logger.info(f"Verification result: {result}")

            if result.get("visible") and result.get("confidence", 0) > 0.7:
                return self._ok(f"Verified visible: {element_description}", {"verification": result}, t0)

            return self._fail(ERROR_VERIFICATION, f"Element not visible: {element_description}", {"verification": result}, t0)

        except Exception as e:
            logger.exception(f"Exception during verification: {e}")
            return self._fail(ERROR_VERIFICATION, f"Verification error: {e}", {"element": element_description}, t0)

    def type_text(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"Typing text (length={len(text)}): '{text[:50]}...'")

        try:
            for i, ch in enumerate(text):
                self.kb.type(ch)
                if self.type_delay_s > 0:
                    time.sleep(self.type_delay_s)
                if i > 0 and i % 50 == 0:
                    logger.debug(f"Typed {i}/{len(text)} characters...")

            return self._ok("Typed text", {"chars": len(text), "preview": text[:50]}, t0)
        except Exception as e:
            logger.exception(f"Exception during typing: {e}")
            return self._fail(ERROR_TYPE, f"Type error: {e}", {"text_preview": text[:40]}, t0)

    def hotkey(self, keys: List[str]) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"Executing hotkey: {' + '.join(keys)}")

        try:
            ks = [self._map_key(k) for k in keys]

            for k in ks[:-1]:
                self.kb.press(k)

            self.kb.press(ks[-1])
            self.kb.release(ks[-1])

            for k in reversed(ks[:-1]):
                self.kb.release(k)

            return self._ok("Hotkey executed", {"keys": keys}, t0)
        except Exception as e:
            logger.exception(f"Exception during hotkey: {e}")
            return self._fail(ERROR_HOTKEY, f"Hotkey error: {e}", {"keys": keys}, t0)

    # -----------------------------
    # Utility Methods
    # -----------------------------
    def _take_screenshot(self, suffix: str = "") -> str:
        screenshot = ImageGrab.grab()

        if self.save_screenshots:
            timestamp = int(time.time() * 1000)
            filename = f"screenshot_{timestamp}_{suffix}.png" if suffix else f"screenshot_{timestamp}.png"
            screenshot_path = self.debug_dir / filename
            screenshot.save(screenshot_path, "PNG")
            logger.debug(f"Screenshot saved to: {screenshot_path}")

        temp_path = Path("/tmp/desktop_agent_screenshot.png")
        screenshot.save(temp_path, "PNG")

        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        return encoded

    def get_active_app(self) -> str:
        script = (
            'tell application "System Events"\n'
            'set frontApp to name of first application process whose frontmost is true\n'
            'end tell\n'
            'return frontApp'
        )
        rc, out, _ = run_osascript(script, timeout=5)
        return out.strip() if rc == 0 else ""

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

    def _ok(self, message: str, extracted: Dict[str, Any], t0: float) -> Dict[str, Any]:
        return {
            "ok": True,
            "error_type": None,
            "message": message,
            "evidence": {"active_app": self.get_active_app()},
            "extracted": extracted or {},
            "timing_ms": int((time.time() - t0) * 1000),
        }

    def _fail(self, error_type: str, message: str, extracted: Dict[str, Any], t0: float) -> Dict[str, Any]:
        return {
            "ok": False,
            "error_type": error_type,
            "message": message,
            "evidence": {"active_app": self.get_active_app()},
            "extracted": extracted or {},
            "timing_ms": int((time.time() - t0) * 1000),
        }


# -----------------------------
# Example Usage (debug individually)
# -----------------------------
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment to run desktop_agent.py directly.")

    agent = MacDesktopAgent(
        anthropic_api_key=api_key,
        save_screenshots=_env_bool("DESKTOP_SAVE_SCREENSHOTS", True),
        debug_show_clicks=_env_bool("DESKTOP_DEBUG_CLICKS", True),
        apply_menubar_offset=_env_bool("DESKTOP_APPLY_MENUBAR_OFFSET", True),
        debug_dir=os.getenv("DESKTOP_DEBUG_DIR", "/tmp/desktop_agent_debug"),
        type_delay_s=_env_float("DESKTOP_TYPE_DELAY_S", 0.0),
    )

    steps = [
        {"id": "step_1", "action": "launch_fullscreen", "args": {"app_name": "TextEdit"}, "retries": 2},
        {"id": "step_2", "action": "click_element", "args": {"element_description": "New Document button"}, "retries": 3},
        {"id": "step_3", "action": "type_text", "args": {"text": "Hello from Desktop Agent!\n"}, "retries": 1},
    ]

    for s in steps:
        r = agent.run_step(s, {})
        print(json.dumps(r, indent=2))
        if not r.get("ok"):
            break
