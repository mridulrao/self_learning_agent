#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import platform
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Deque, Tuple

# Screen recording
import cv2
import numpy as np
from PIL import ImageGrab

# macOS window bounds (Quartz) — optional fallback window crop
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
    kCGWindowOwnerPID,
    kCGWindowLayer,
    kCGWindowBounds,
    kCGWindowNumber,
    kCGWindowName,
)

# Browser instrumentation
from playwright.async_api import async_playwright, Page

# OS-level event tracking
from pynput import mouse, keyboard

# Voice
import sounddevice as sd
import requests
import torch
from AppKit import NSWorkspace, NSScreen

from dotenv import load_dotenv

from dom_script import dom_script

load_dotenv()

# -----------------------------
# Voice config
# -----------------------------
@dataclass
class VoiceConfig:
    enabled: bool = True

    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 32  # ~512 samples @ 16k

    # ✅ Make trigger a bit more forgiving than 0.5/3 (still robust)
    vad_threshold: float = 0.35
    start_trigger_blocks: int = 2
    end_hangover_ms: int = 700
    min_note_ms: int = 400
    max_note_ms: int = 30_000
    merge_gap_ms: int = 300

    # ✅ Debug/diagnostics (writes to events.jsonl)
    log_audio_debug: bool = True
    audio_debug_every_ms: int = 2000  # heartbeat cadence

    # (Optional) force a specific device index if needed
    input_device: Optional[int] = None

    deepgram_model: str = "nova-3"
    deepgram_smart_format: bool = True
    deepgram_punctuate: bool = True
    deepgram_utterances: bool = False
    deepgram_language: Optional[str] = None



class VoiceNoteRecorder:
    """
    Always-on mic capture -> Silero VAD -> writes wav segments -> Deepgram transcribes
    Emits events back into the main recorder's event queue.
    """

    def __init__(self, session_dir: Path, event_log_fn, voice_cfg: VoiceConfig):
        self.session_dir = session_dir
        self.voice_cfg = voice_cfg
        self.log_event = event_log_fn

        self.voice_dir = self.session_dir / "voice_notes"
        self.voice_dir.mkdir(exist_ok=True)

        self._stop = threading.Event()
        self._audio_q: "queue.Queue[tuple[int, bytes]]" = queue.Queue(maxsize=400)  # a bit bigger
        self._segment_q: "queue.Queue[dict]" = queue.Queue()

        self._capture_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._stt_thread: Optional[threading.Thread] = None

        self._vad_model = None
        self._dg_key = os.getenv("DEEPGRAM_API_KEY")
        self._dropped_audio_blocks = 0

        # debug counters
        self._blocks_seen = 0
        self._silent_blocks = 0
        self._last_debug_ms = 0

    def start(self):
        if not self.voice_cfg.enabled:
            return

        # Log device info
        try:
            default_in = sd.default.device[0]  # (input, output)
            dev = sd.query_devices(default_in, "input") if default_in is not None else None
            self.log_event("voice_audio_device", {
                "default_input_device": default_in,
                "configured_input_device": self.voice_cfg.input_device,
                "default_input_device_info": dev,
            })
        except Exception as e:
            self.log_event("voice_error", {"stage": "device_info", "error": str(e)})

        if not self._dg_key:
            print("⚠ Voice enabled but DEEPGRAM_API_KEY is not set. Transcription will be skipped.")

        self._load_silero()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._stt_thread = threading.Thread(target=self._stt_loop, daemon=True)

        self._capture_thread.start()
        self._vad_thread.start()
        self._stt_thread.start()

        self.log_event("voice_recorder_started", {
            "sample_rate": self.voice_cfg.sample_rate,
            "block_ms": self.voice_cfg.block_ms,
            "vad_threshold": self.voice_cfg.vad_threshold,
            "start_trigger_blocks": self.voice_cfg.start_trigger_blocks,
        })

    def stop(self):
        if not self.voice_cfg.enabled:
            return

        self._stop.set()
        for t in [self._capture_thread, self._vad_thread, self._stt_thread]:
            if t:
                t.join(timeout=2)

        if self._dropped_audio_blocks:
            self.log_event("voice_audio_drop", {"dropped_blocks": self._dropped_audio_blocks})

        self.log_event("voice_recorder_stopped", {})

    def _load_silero(self):
        torch.set_num_threads(1)
        try:
            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            model.eval()
            self._vad_model = model
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD via torch.hub: {e}")

    def _capture_loop(self):
        cfg = self.voice_cfg
        blocksize = int(cfg.sample_rate * (cfg.block_ms / 1000.0))

        def callback(indata, frames, time_info, status):
            if self._stop.is_set():
                raise sd.CallbackStop

            ts_ms = int(time.time() * 1000)
            pcm16 = bytes(indata)

            # Debug RMS + silence ratio heartbeat
            try:
                a = np.frombuffer(pcm16, dtype=np.int16)
                rms = float(np.sqrt(np.mean((a.astype(np.float32) / 32768.0) ** 2))) if a.size else 0.0
                is_silent = (rms < 0.001)
            except Exception:
                rms = -1.0
                is_silent = False

            self._blocks_seen += 1
            if is_silent:
                self._silent_blocks += 1

            if cfg.log_audio_debug:
                if (ts_ms - self._last_debug_ms) >= int(cfg.audio_debug_every_ms):
                    self._last_debug_ms = ts_ms
                    self.log_event("voice_audio_debug", {
                        "blocks_seen": self._blocks_seen,
                        "silent_blocks": self._silent_blocks,
                        "silent_ratio": float(self._silent_blocks) / float(max(1, self._blocks_seen)),
                        "rms": rms,
                        "frames": frames,
                    })

            try:
                self._audio_q.put_nowait((ts_ms, pcm16))
            except queue.Full:
                self._dropped_audio_blocks += 1

        try:
            with sd.RawInputStream(
                samplerate=cfg.sample_rate,
                channels=cfg.channels,
                dtype="int16",
                blocksize=blocksize,
                callback=callback,
                device=cfg.input_device,  # None => default
            ):
                while not self._stop.is_set():
                    time.sleep(0.05)
        except Exception as e:
            self.log_event("voice_error", {"stage": "capture", "error": str(e)})

    def _vad_prob(self, pcm16: bytes) -> float:
        cfg = self.voice_cfg
        audio_i16 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        audio_f = audio_i16 / 32768.0
        wav = torch.from_numpy(audio_f)
        with torch.no_grad():
            p = self._vad_model(wav, cfg.sample_rate).item()
        return float(p)

    def _write_wav(self, path: Path, pcm16_concat: bytes):
        cfg = self.voice_cfg
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(cfg.channels)
            wf.setsampwidth(2)
            wf.setframerate(cfg.sample_rate)
            wf.writeframes(pcm16_concat)

    def _vad_loop(self):
        cfg = self.voice_cfg
        if self._vad_model is None:
            self.log_event("voice_error", {"stage": "vad", "error": "VAD model not loaded"})
            return

        speech_on = False
        speech_blocks = 0
        note_start_ms: Optional[int] = None
        last_speech_ms: Optional[int] = None
        buffered: List[bytes] = []

        def finalize_note(end_ms: int):
            nonlocal speech_on, note_start_ms, last_speech_ms, buffered, speech_blocks

            if note_start_ms is None:
                speech_on = False
                buffered = []
                speech_blocks = 0
                last_speech_ms = None
                return

            dur = end_ms - note_start_ms
            if dur < cfg.min_note_ms:
                speech_on = False
                buffered = []
                speech_blocks = 0
                last_speech_ms = None
                note_start_ms = None
                return

            fname = f"note_{note_start_ms}_{end_ms}.wav"
            wav_path = self.voice_dir / fname
            pcm = b"".join(buffered)
            self._write_wav(wav_path, pcm)

            self.log_event("voice_note_start", {
                "note_id": fname.replace(".wav", ""),
                "time_ms": note_start_ms,
                "audio_path": str(wav_path),
            })
            self.log_event("voice_note_end", {
                "note_id": fname.replace(".wav", ""),
                "time_ms": end_ms,
                "audio_path": str(wav_path),
                "duration_ms": dur,
            })
            self.log_event("voice_note_transcript_pending", {
                "note_id": fname.replace(".wav", ""),
                "time_ms": note_start_ms,
                "start_ms": note_start_ms,
                "end_ms": end_ms,
                "audio_path": str(wav_path),
            })

            self._segment_q.put({
                "note_id": fname.replace(".wav", ""),
                "start_ms": note_start_ms,
                "end_ms": end_ms,
                "audio_path": str(wav_path),
            })

            speech_on = False
            buffered = []
            speech_blocks = 0
            last_speech_ms = None
            note_start_ms = None

        while not self._stop.is_set():
            try:
                ts_ms, pcm16 = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                p = self._vad_prob(pcm16)
            except Exception as e:
                self.log_event("voice_error", {"stage": "vad_prob", "error": str(e)})
                continue

            is_speech = p >= cfg.vad_threshold
            if is_speech:
                last_speech_ms = ts_ms

            if not speech_on:
                if is_speech:
                    speech_blocks += 1
                    if speech_blocks >= cfg.start_trigger_blocks:
                        speech_on = True
                        # ✅ match your old recorder: start at trigger time
                        note_start_ms = ts_ms
                        buffered = [pcm16]
                else:
                    speech_blocks = 0
            else:
                buffered.append(pcm16)

                if note_start_ms is not None and (ts_ms - note_start_ms) >= cfg.max_note_ms:
                    finalize_note(ts_ms)
                    continue

                if last_speech_ms is not None:
                    silence_ms = ts_ms - last_speech_ms
                    if silence_ms >= cfg.end_hangover_ms:
                        finalize_note(last_speech_ms)

    def _deepgram_transcribe(self, wav_path: str) -> Dict[str, Any]:
        if not self._dg_key:
            return {"_error": "DEEPGRAM_API_KEY not set"}

        cfg = self.voice_cfg
        params = {
            "model": cfg.deepgram_model,
            "smart_format": str(cfg.deepgram_smart_format).lower(),
            "punctuate": str(cfg.deepgram_punctuate).lower(),
        }
        if cfg.deepgram_utterances:
            params["utterances"] = "true"
        if cfg.deepgram_language:
            params["language"] = cfg.deepgram_language

        headers = {"Authorization": f"Token {self._dg_key}", "Content-Type": "audio/wav"}
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        r = requests.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, data=audio_bytes, timeout=60)
        if r.status_code >= 400:
            return {"_error": f"Deepgram HTTP {r.status_code}", "_body": r.text[:2000]}
        return r.json()

    def _extract_transcript_and_words(self, dg_json: Dict[str, Any]) -> Dict[str, Any]:
        try:
            alt = dg_json["results"]["channels"][0]["alternatives"][0]
            transcript = alt.get("transcript", "") or ""
            words = alt.get("words", []) or []
            return {"transcript": transcript, "words": words}
        except Exception:
            return {"transcript": "", "words": [], "_raw": dg_json}

    def _stt_loop(self):
        while not self._stop.is_set():
            try:
                seg = self._segment_q.get(timeout=0.1)
            except queue.Empty:
                continue

            note_id = seg["note_id"]
            start_ms = int(seg["start_ms"])
            end_ms = int(seg["end_ms"])
            wav_path = seg["audio_path"]

            dg = self._deepgram_transcribe(wav_path)
            if "_error" in dg:
                self.log_event("voice_note_transcript", {
                    "note_id": note_id,
                    "time_ms": start_ms,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "audio_path": wav_path,
                    "ok": False,
                    "error": dg["_error"],
                    "deepgram_body": dg.get("_body", ""),
                })
                continue

            parsed = self._extract_transcript_and_words(dg)

            word_items = []
            for w in parsed.get("words", [])[:5000]:
                try:
                    ws = int(float(w.get("start", 0.0)) * 1000.0) + start_ms
                    we = int(float(w.get("end", 0.0)) * 1000.0) + start_ms
                    word_items.append({"word": w.get("word", ""), "start_ms": ws, "end_ms": we, "confidence": w.get("confidence", None)})
                except Exception:
                    continue

            self.log_event("voice_note_transcript", {
                "note_id": note_id,
                "time_ms": start_ms,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "audio_path": wav_path,
                "ok": True,
                "model": self.voice_cfg.deepgram_model,
                "text": parsed.get("transcript", ""),
                "words": word_items,
            })


class ActivityRecorder:
    def __init__(
        self,
        output_dir: str = "./recordings",
        browser_type: str = "chromium",
        browser_channel: str = "chrome",
        log_browser_events: bool = False,
        voice_cfg: Optional[VoiceConfig] = None,

        # DOM logger controls
        log_text_input_on_each_keystroke: bool = False,
        capture_web_screenshots: bool = True,
        capture_web_screenshots_on: Optional[List[str]] = None,

        # Correlation settings
        correlate_click_window_ms: int = 250,

        # Screen recording
        screen_fps: float = 10.0,
        frame_checkpoint_every_n: int = 30,

        # Desktop keyboard logging
        log_raw_keystrokes: bool = True,
        key_buffer_idle_flush_ms: int = 800,
        key_buffer_max_len: int = 2000,

        # Active app change polling
        active_app_poll_ms: int = 200,

        # Click evidence timing
        pre_click_delay_ms: int = 20,
        post_click_delay_ms: int = 220,

        # Click evidence crop sizes
        pre_click_half_size_px: int = 160,
        post_click_half_size_px: int = 350,

        # Optional: also capture active window crop as fallback (post only)
        also_capture_window_fallback: bool = True,

        # -----------------------------
        # NEW: web nav noise controls
        # -----------------------------
        log_iframe_navigations: bool = False,     # default: ignore iframe navigations
        capture_nav_screenshots: bool = False,    # default: do NOT screenshot on nav
        nav_dedupe_ms: int = 1200,               # ignore repeated same-url nav logs quickly

        # -----------------------------
        # NEW: web screenshot throttle
        # -----------------------------
        web_screenshot_min_interval_ms: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.browser_type = browser_type
        self.browser_channel = browser_channel
        self.log_browser_events = log_browser_events

        self.log_text_input_on_each_keystroke = log_text_input_on_each_keystroke
        self.capture_web_screenshots = capture_web_screenshots

        # IMPORTANT: Default no "change" screenshots (Google Forms is noisy).
        self.capture_web_screenshots_on = capture_web_screenshots_on or ["click", "submit"]

        self.correlate_click_window_ms = int(correlate_click_window_ms)

        self.screen_fps = float(screen_fps)
        self.frame_checkpoint_every_n = int(frame_checkpoint_every_n)

        self.log_raw_keystrokes = bool(log_raw_keystrokes)
        self.key_buffer_idle_flush_ms = int(key_buffer_idle_flush_ms)
        self.key_buffer_max_len = int(key_buffer_max_len)

        self.active_app_poll_ms = int(active_app_poll_ms)

        self.pre_click_delay_ms = int(pre_click_delay_ms)
        self.post_click_delay_ms = int(post_click_delay_ms)
        self.pre_click_half_size_px = int(pre_click_half_size_px)
        self.post_click_half_size_px = int(post_click_half_size_px)
        self.also_capture_window_fallback = bool(also_capture_window_fallback)

        # Web nav noise controls
        self.log_iframe_navigations = bool(log_iframe_navigations)
        self.capture_nav_screenshots = bool(capture_nav_screenshots)
        self.nav_dedupe_ms = int(nav_dedupe_ms)
        self._last_nav_url: Optional[str] = None
        self._last_nav_ms: int = 0

        # Web screenshot throttle
        self.web_screenshot_min_interval_ms = int(web_screenshot_min_interval_ms)
        self._last_web_shot_ms: int = 0

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)

        self.is_recording = False
        self.event_queue: "queue.Queue[dict]" = queue.Queue()

        self.video_path = self.session_dir / "screen_recording.avi"
        self.events_path = self.session_dir / "events.jsonl"
        self.browser_trace_path = self.session_dir / "browser_trace.zip"

        # Web screenshots
        self.web_screens_dir = self.session_dir / "web_screenshots"
        self.web_screens_dir.mkdir(exist_ok=True)
        self._web_screenshot_lock = asyncio.Lock()
        self._web_screenshot_seq = 0

        # Desktop screenshots
        self.desktop_screens_dir = self.session_dir / "desktop_screenshots"
        self.desktop_screens_dir.mkdir(exist_ok=True)
        self._desktop_screenshot_lock = threading.Lock()
        self._desktop_screenshot_seq = 0

        # Components
        self.screen_recorder = None
        self.mouse_listener = None
        self.keyboard_listener = None
        self.browser_context = None
        self.browser = None
        self.playwright = None
        self.page: Optional[Page] = None

        # Voice
        self.voice_cfg = voice_cfg or VoiceConfig(enabled=True)
        self.voice_recorder: Optional[VoiceNoteRecorder] = None

        # Event IDs + click correlation buffer
        self._event_seq = 0
        self._event_seq_lock = threading.Lock()
        self._recent_os_clicks: Deque[Dict[str, Any]] = deque(maxlen=80)

        # Active app watcher
        self._active_app_thread: Optional[threading.Thread] = None
        self._active_app_stop = threading.Event()
        self._last_active_app: Optional[Dict[str, Any]] = None

        # Raw key buffer
        self._keybuf_lock = threading.Lock()
        self._keybuf_text: str = ""
        self._keybuf_last_ms: Optional[int] = None
        self._keybuf_last_win: Optional[Dict[str, Any]] = None

        # macOS scale cache + screen points size
        self._macos_scale_xy: Optional[Tuple[float, float]] = None
        self._macos_scale_last_ms: int = 0
        self._macos_screen_pts_wh: Optional[Tuple[float, float]] = None

        # pending mouse-down storage
        self._pending_click_lock = threading.Lock()
        self._pending_click: Optional[Dict[str, Any]] = None

    # -----------------------------
    # Utility: event IDs
    # -----------------------------
    def _next_event_id(self) -> str:
        with self._event_seq_lock:
            self._event_seq += 1
            seq = self._event_seq
        return f"{self.session_id}_{seq:08d}"

    def log_event(self, event_type: str, data: dict):
        now_ms = int(time.time() * 1000)
        event = {
            "event_id": self._next_event_id(),
            "timestamp": datetime.now().isoformat(),
            "time_ms": now_ms,
            "event_type": event_type,
            **data,
        }
        self.event_queue.put(event)

    def _sleep_ms(self, ms: int):
        if ms > 0:
            time.sleep(ms / 1000.0)

    # -----------------------------
    # Active window info
    # -----------------------------
    def get_active_window_info(self) -> dict:
        try:
            if platform.system() == "Windows":
                return {"window_title": "WindowsActiveWindow", "app_name": "WindowsActiveApp", "pid": None, "bundle_id": None}

            if platform.system() == "Darwin":
                workspace = NSWorkspace.sharedWorkspace()
                active_app = workspace.activeApplication() or {}
                name = active_app.get("NSApplicationName", "Unknown")
                pid = active_app.get("NSApplicationProcessIdentifier", None)
                bundle_id = active_app.get("NSApplicationBundleIdentifier", None)
                return {"window_title": name, "app_name": name, "pid": pid, "bundle_id": bundle_id}

            if platform.system() == "Linux":
                return {"window_title": "LinuxActiveWindow", "app_name": "LinuxActiveApp", "pid": None, "bundle_id": None}

        except Exception as e:
            return {"window_title": "Error", "app_name": "Error", "pid": None, "bundle_id": None, "error": str(e)}

        return {"window_title": "Unknown", "app_name": "Unknown", "pid": None, "bundle_id": None}

    # -----------------------------
    # macOS: points->pixels mapping
    # -----------------------------
    def _macos_get_scale_xy(self) -> Tuple[float, float]:
        now = int(time.time() * 1000)
        if self._macos_scale_xy and (now - self._macos_scale_last_ms) < 2000:
            return self._macos_scale_xy

        try:
            screen = NSScreen.mainScreen()
            if screen is None:
                self._macos_scale_xy = (1.0, 1.0)
                self._macos_screen_pts_wh = None
                self._macos_scale_last_ms = now
                return self._macos_scale_xy

            frame = screen.frame()
            pts_w = float(frame.size.width) if float(frame.size.width) > 0 else 1.0
            pts_h = float(frame.size.height) if float(frame.size.height) > 0 else 1.0
            self._macos_screen_pts_wh = (pts_w, pts_h)

            img = ImageGrab.grab()
            px_w, px_h = img.size

            sx = float(px_w) / pts_w
            sy = float(px_h) / pts_h

            if sx < 0.5 or sx > 4.0:
                sx = 1.0
            if sy < 0.5 or sy > 4.0:
                sy = 1.0

            self._macos_scale_xy = (sx, sy)
            self._macos_scale_last_ms = now
            return self._macos_scale_xy
        except Exception:
            self._macos_scale_xy = (1.0, 1.0)
            self._macos_screen_pts_wh = None
            self._macos_scale_last_ms = now
            return self._macos_scale_xy

    def _to_screen_px(self, x: float, y: float, screen_px_wh: Tuple[int, int]) -> Tuple[int, int, Dict[str, Any]]:
        sw, sh = int(screen_px_wh[0]), int(screen_px_wh[1])
        meta: Dict[str, Any] = {"coord_in": "unknown", "coord_out": "screen_px"}

        if platform.system() != "Darwin":
            cx = int(round(x))
            cy = int(round(y))
            cx = max(0, min(cx, sw - 1))
            cy = max(0, min(cy, sh - 1))
            meta["coord_in"] = "screen_px_assumed"
            return cx, cy, meta

        sx, sy = self._macos_get_scale_xy()
        pts_wh = self._macos_screen_pts_wh

        treat_as_points = False
        if pts_wh:
            pts_w, pts_h = pts_wh
            if (x <= pts_w + 2) and (y <= pts_h + 2) and (sw >= int(pts_w * 1.15)):
                treat_as_points = True

        if treat_as_points:
            cx = int(round(float(x) * sx))
            cy = int(round(float(y) * sy))
            meta["coord_in"] = "screen_points_detected"
            meta["scale_xy"] = [sx, sy]
            meta["screen_points_wh"] = [pts_wh[0], pts_wh[1]] if pts_wh else None
        else:
            cx = int(round(x))
            cy = int(round(y))
            meta["coord_in"] = "screen_px_detected_or_assumed"
            meta["scale_xy"] = [sx, sy]
            meta["screen_points_wh"] = [pts_wh[0], pts_wh[1]] if pts_wh else None

        cx = max(0, min(cx, sw - 1))
        cy = max(0, min(cy, sh - 1))
        return cx, cy, meta

    # -----------------------------
    # Crop math
    # -----------------------------
    @staticmethod
    def _centered_bbox(cx: int, cy: int, half: int, sw: int, sh: int) -> Tuple[int, int, int, int]:
        half = max(1, int(half))
        w = 2 * half
        h = 2 * half

        left = cx - half
        top = cy - half
        right = left + w
        bottom = top + h

        if left < 0:
            right += -left
            left = 0
        if right > sw:
            shift = right - sw
            left -= shift
            right = sw
        if left < 0:
            left = 0
        if right <= left + 1:
            right = min(sw, left + 2)

        if top < 0:
            bottom += -top
            top = 0
        if bottom > sh:
            shift = bottom - sh
            top -= shift
            bottom = sh
        if top < 0:
            top = 0
        if bottom <= top + 1:
            bottom = min(sh, top + 2)

        return (int(left), int(top), int(right), int(bottom))

    # -----------------------------
    # macOS window fallback crop (optional)
    # -----------------------------
    def _macos_pick_front_window_for_pid(self, pid: int) -> Optional[Dict[str, Any]]:
        try:
            wins = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []
        except Exception:
            return None

        candidates: List[Dict[str, Any]] = []
        for w in wins:
            try:
                if int(w.get(kCGWindowOwnerPID, -1)) != int(pid):
                    continue
                if int(w.get(kCGWindowLayer, 999)) != 0:
                    continue

                b = w.get(kCGWindowBounds) or {}
                x = float(b.get("X", 0))
                y = float(b.get("Y", 0))
                ww = float(b.get("Width", 0))
                hh = float(b.get("Height", 0))
                if ww < 80 or hh < 80:
                    continue

                wid = int(w.get(kCGWindowNumber, 0))
                name = str(w.get(kCGWindowName, "") or "")
                candidates.append({
                    "window_id": wid,
                    "bounds_points": {"x": x, "y": y, "w": ww, "h": hh},
                    "name": name,
                    "area": ww * hh,
                })
            except Exception:
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda d: float(d.get("area", 0.0)), reverse=True)
        return candidates[0]

    # -----------------------------
    # Desktop screenshot capture around click
    # -----------------------------
    def _save_crop(self, full_img, bbox: Tuple[int, int, int, int], suffix: str) -> Dict[str, Any]:
        with self._desktop_screenshot_lock:
            self._desktop_screenshot_seq += 1
            seq = self._desktop_screenshot_seq

        ts = int(time.time() * 1000)
        fname = f"{ts}_{seq:06d}_{suffix}.png"
        path = self.desktop_screens_dir / fname

        crop = full_img.crop(bbox)
        crop.save(str(path), format="PNG")

        return {
            "screenshot_path": str(path.relative_to(self.session_dir)),
            "image_size": {"w": int(crop.size[0]), "h": int(crop.size[1])},
            "bbox_px": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "coord_space": "screen_px_cropped_region",
        }

    def _capture_click_evidence_bundle(
        self,
        *,
        click_id: str,
        raw_x: float,
        raw_y: float,
        click_xy_px: Tuple[int, int],
        window_info: Dict[str, Any],
        when: str,
        half_size_px: int,
        full_img=None,
        include_window_fallback: bool = False,
    ) -> Dict[str, Any]:
        if full_img is None:
            full_img = ImageGrab.grab()
        sw, sh = full_img.size

        cx, cy = click_xy_px
        bbox = self._centered_bbox(cx, cy, half_size_px, sw, sh)

        shot = self._save_crop(
            full_img,
            bbox,
            suffix=f"{click_id}_{when}_clickcrop",
        )

        out: Dict[str, Any] = {
            **shot,
            "crop": {
                "kind": "around_click",
                "center_px": [int(cx), int(cy)],
                "half_size_px": int(half_size_px),
            },
        }

        if include_window_fallback and platform.system() == "Darwin":
            pid = window_info.get("pid")
            if pid:
                win = self._macos_pick_front_window_for_pid(int(pid))
                if win:
                    sx, sy = self._macos_get_scale_xy()
                    b = win["bounds_points"]

                    left = int(round(b["x"] * sx))
                    top = int(round(b["y"] * sy))
                    right = int(round((b["x"] + b["w"]) * sx))
                    bottom = int(round((b["y"] + b["h"]) * sy))

                    left = max(0, min(left, sw - 1))
                    top = max(0, min(top, sh - 1))
                    right = max(left + 1, min(right, sw))
                    bottom = max(top + 1, min(bottom, sh))

                    bbox_w = (left, top, right, bottom)
                    fb = self._save_crop(full_img, bbox_w, suffix=f"{click_id}_{when}_window")

                    out["window_fallback"] = {
                        **fb,
                        "coord_space": "screen_px_cropped_window",
                        "window_crop": {
                            "window_id": win.get("window_id"),
                            "window_name": win.get("name", ""),
                            "bounds_points": b,
                            "scale_xy": [sx, sy],
                            "bbox_px": [left, top, right, bottom],
                        },
                    }

        return out

    # -----------------------------
    # Active app watcher
    # -----------------------------
    def start_active_app_watcher(self):
        self._active_app_stop.clear()
        self._last_active_app = None

        def _same_app(a: Optional[dict], b: Optional[dict]) -> bool:
            if not a or not b:
                return False
            return (
                (a.get("bundle_id") == b.get("bundle_id"))
                and (a.get("pid") == b.get("pid"))
                and (a.get("app_name") == b.get("app_name"))
            )

        def loop():
            prev = self.get_active_window_info()
            self._last_active_app = prev
            self.log_event("active_app_snapshot", {"active": prev})

            poll_s = max(0.05, self.active_app_poll_ms / 1000.0)

            while self.is_recording and not self._active_app_stop.is_set():
                cur = self.get_active_window_info()
                if not _same_app(prev, cur):
                    self.log_event("active_app_changed", {"from": prev, "to": cur})
                    prev = cur
                    self._last_active_app = cur
                time.sleep(poll_s)

        self._active_app_thread = threading.Thread(target=loop, daemon=True)
        self._active_app_thread.start()

    # -----------------------------
    # Screen recording
    # -----------------------------
    def start_screen_recording(self):
        def record_screen():
            screen = ImageGrab.grab()
            width, height = screen.size
            start_ms = int(time.time() * 1000)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(str(self.video_path), fourcc, self.screen_fps, (width, height))

            self.log_event("screen_recording_started", {
                "width": width,
                "height": height,
                "fps": self.screen_fps,
                "start_time_ms": start_ms,
                "video_path": str(self.video_path.relative_to(self.session_dir)),
            })

            frame_idx = 0
            frame_period = 1.0 / max(self.screen_fps, 0.1)

            while self.is_recording:
                t0 = time.time()
                img = ImageGrab.grab()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)

                frame_idx += 1
                if self.frame_checkpoint_every_n > 0 and (frame_idx % self.frame_checkpoint_every_n) == 0:
                    self.log_event("screen_frame_checkpoint", {
                        "frame_idx": frame_idx,
                        "approx_video_ms": int(time.time() * 1000 - start_ms),
                        "width": width,
                        "height": height,
                    })

                dt = time.time() - t0
                time.sleep(max(0.0, frame_period - dt))

            out.release()
            self.log_event("screen_recording_stopped", {"end_time_ms": int(time.time() * 1000)})

        self.screen_recorder = threading.Thread(target=record_screen, daemon=True)
        self.screen_recorder.start()

    # -----------------------------
    # Key buffer
    # -----------------------------
    def _flush_key_buffer(self, reason: str = "idle"):
        if not self.log_raw_keystrokes:
            return

        with self._keybuf_lock:
            txt = self._keybuf_text
            last_ms = self._keybuf_last_ms
            win = self._keybuf_last_win
            self._keybuf_text = ""
            self._keybuf_last_ms = None
            self._keybuf_last_win = None

        if txt and win and last_ms is not None:
            self.log_event("key_text_input", {"text": txt, "reason": reason, "length": len(txt), **win})

    def _append_key_char(self, ch: str, window_info: dict):
        if not self.log_raw_keystrokes:
            return

        now_ms = int(time.time() * 1000)

        with self._keybuf_lock:
            prev_win = self._keybuf_last_win
            if prev_win is not None and (
                prev_win.get("bundle_id") != window_info.get("bundle_id")
                or prev_win.get("pid") != window_info.get("pid")
            ):
                txt = self._keybuf_text
                last_ms = self._keybuf_last_ms
                win = self._keybuf_last_win
                self._keybuf_text = ""
                self._keybuf_last_ms = None
                self._keybuf_last_win = None
            else:
                txt = ""
                last_ms = None
                win = None

        if txt and win and last_ms is not None:
            self.log_event("key_text_input", {"text": txt, "reason": "app_change", "length": len(txt), **win})

        with self._keybuf_lock:
            if self._keybuf_last_ms is not None:
                if (now_ms - self._keybuf_last_ms) >= self.key_buffer_idle_flush_ms and self._keybuf_text:
                    txt2 = self._keybuf_text
                    win2 = self._keybuf_last_win
                    self._keybuf_text = ""
                    self._keybuf_last_ms = None
                    self._keybuf_last_win = None
                else:
                    txt2 = ""
                    win2 = None
            else:
                txt2 = ""
                win2 = None

        if txt2 and win2:
            self.log_event("key_text_input", {"text": txt2, "reason": "idle", "length": len(txt2), **win2})

        with self._keybuf_lock:
            if len(self._keybuf_text) < self.key_buffer_max_len:
                self._keybuf_text += ch
            self._keybuf_last_ms = now_ms
            self._keybuf_last_win = window_info

    def _backspace_key_buffer(self, window_info: dict):
        if not self.log_raw_keystrokes:
            return
        with self._keybuf_lock:
            if self._keybuf_last_win and (
                self._keybuf_last_win.get("bundle_id") != window_info.get("bundle_id")
                or self._keybuf_last_win.get("pid") != window_info.get("pid")
            ):
                return
            if self._keybuf_text:
                self._keybuf_text = self._keybuf_text[:-1]
            self._keybuf_last_ms = int(time.time() * 1000)
            self._keybuf_last_win = window_info

    # -----------------------------
    # Input tracking (mouse + keyboard)
    # -----------------------------
    def start_input_tracking(self):
        def on_mouse_move(x, y):
            window_info = self.get_active_window_info()
            self.log_event("mouse_move", {"x": x, "y": y, **window_info})

        def on_mouse_click(x, y, button, pressed):
            window_info = self.get_active_window_info()

            # flush any pending text BEFORE click boundary
            self._flush_key_buffer(reason="before_click")

            if pressed is True:
                down_ms = int(time.time() * 1000)

                self._sleep_ms(self.pre_click_delay_ms)
                full = ImageGrab.grab()
                sw, sh = full.size
                cx, cy, coord_meta = self._to_screen_px(x, y, (sw, sh))

                click_id = self._next_event_id()

                pre = self._capture_click_evidence_bundle(
                    click_id=click_id,
                    raw_x=x,
                    raw_y=y,
                    click_xy_px=(cx, cy),
                    window_info=window_info,
                    when="pre",
                    half_size_px=self.pre_click_half_size_px,
                    full_img=full,
                    include_window_fallback=False,
                )

                with self._pending_click_lock:
                    self._pending_click = {
                        "click_id": click_id,
                        "raw_x": x,
                        "raw_y": y,
                        "click_px": (cx, cy),
                        "coord_meta": coord_meta,
                        "window_info": window_info,
                        "time_ms_down": down_ms,
                        "pre": pre,
                    }

                evt = {
                    "event_id": click_id,
                    "timestamp": datetime.now().isoformat(),
                    "time_ms": down_ms,
                    "event_type": "mouse_click",
                    "phase": "down",
                    "x": x,
                    "y": y,
                    "x_px": cx,
                    "y_px": cy,
                    "button": str(button),
                    "pressed": True,
                    "coord_meta": coord_meta,
                    **window_info,
                    "artifact": {
                        "artifact_schema": "desktop_click_evidence_v1",
                        "click_id": click_id,
                        "coord_space": "screen_px",
                        "raw_click_xy": {"x": x, "y": y},
                        "click_xy_px": {"x": cx, "y": cy},
                        "pre": pre,
                    }
                }
                self.event_queue.put(evt)
                return

            # mouse up
            up_ms = int(time.time() * 1000)

            pending = None
            with self._pending_click_lock:
                pending = self._pending_click
                self._pending_click = None

            self._sleep_ms(self.post_click_delay_ms)

            full = ImageGrab.grab()
            sw, sh = full.size
            cx, cy, coord_meta = self._to_screen_px(x, y, (sw, sh))

            stable_click_id = (pending["click_id"] if pending else self._next_event_id())
            up_event_id = self._next_event_id()

            post = self._capture_click_evidence_bundle(
                click_id=stable_click_id,
                raw_x=(pending["raw_x"] if pending else x),
                raw_y=(pending["raw_y"] if pending else y),
                click_xy_px=(pending["click_px"] if pending else (cx, cy)),
                window_info=(pending["window_info"] if pending else window_info),
                when="post",
                half_size_px=self.post_click_half_size_px,
                full_img=full,
                include_window_fallback=self.also_capture_window_fallback,
            )

            bundle = {
                "artifact_schema": "desktop_click_evidence_v1",
                "click_id": stable_click_id,
                "coord_space": "screen_px",
                "raw_click_xy": {"x": (pending["raw_x"] if pending else x), "y": (pending["raw_y"] if pending else y)},
                "click_xy_px": {"x": (pending["click_px"][0] if pending else cx), "y": (pending["click_px"][1] if pending else cy)},
                "pre": (pending["pre"] if pending and pending.get("pre") else None),
                "post": post,
            }
            if post.get("window_fallback"):
                bundle["fallback_window"] = post["window_fallback"]

            evt = {
                "event_id": up_event_id,
                "timestamp": datetime.now().isoformat(),
                "time_ms": up_ms,
                "event_type": "mouse_click",
                "phase": "up",
                "x": x,
                "y": y,
                "x_px": cx,
                "y_px": cy,
                "button": str(button),
                "pressed": False,
                "coord_meta": (pending["coord_meta"] if pending else coord_meta),
                **window_info,
                "artifact": bundle,
                "screenshot_pre_delay_ms": self.pre_click_delay_ms,
                "screenshot_post_delay_ms": self.post_click_delay_ms,
            }

            self.event_queue.put(evt)

            self._recent_os_clicks.append({
                "input_event_id": up_event_id,
                "time_ms": up_ms,
                "x": x,
                "y": y,
                "x_px": cx,
                "y_px": cy,
                "button": str(button),
                **window_info,
                "artifact": bundle,
            })

        def on_mouse_scroll(x, y, dx, dy):
            window_info = self.get_active_window_info()
            self._flush_key_buffer(reason="before_scroll")
            self.log_event("mouse_scroll", {"x": x, "y": y, "dx": dx, "dy": dy, **window_info})

        SPECIAL_KEYS = {
            keyboard.Key.enter: "Enter",
            keyboard.Key.esc: "Escape",
            keyboard.Key.tab: "Tab",
            keyboard.Key.space: "Space",
            keyboard.Key.backspace: "Backspace",
            keyboard.Key.delete: "Delete",
            keyboard.Key.up: "ArrowUp",
            keyboard.Key.down: "ArrowDown",
            keyboard.Key.left: "ArrowLeft",
            keyboard.Key.right: "ArrowRight",
            keyboard.Key.home: "Home",
            keyboard.Key.end: "End",
            keyboard.Key.page_up: "PageUp",
            keyboard.Key.page_down: "PageDown",
        }

        held_modifiers = {"ctrl": False, "alt": False, "shift": False, "meta": False}
        MOD_KEYS = {
            keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
            keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
            keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
            keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r,
        }

        def _set_modifier(k, down: bool):
            if k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                held_modifiers["ctrl"] = down
            elif k in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                held_modifiers["alt"] = down
            elif k in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                held_modifiers["shift"] = down
            elif k in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                held_modifiers["meta"] = down

        def _hotkey_string(ch: str) -> str:
            parts = []
            if held_modifiers["ctrl"]:
                parts.append("Ctrl")
            if held_modifiers["alt"]:
                parts.append("Alt")
            if held_modifiers["shift"]:
                parts.append("Shift")
            if held_modifiers["meta"]:
                parts.append("Cmd")
            parts.append(ch.upper() if len(ch) == 1 else ch)
            return "+".join(parts)

        def on_key_press(key):
            window_info = self.get_active_window_info()

            if key in MOD_KEYS:
                _set_modifier(key, True)
                return

            if isinstance(key, keyboard.Key) and key in SPECIAL_KEYS:
                kname = SPECIAL_KEYS[key]

                if kname == "Backspace":
                    self._backspace_key_buffer(window_info)
                elif kname == "Space":
                    if self.log_raw_keystrokes:
                        self._append_key_char(" ", window_info)
                elif kname == "Tab":
                    if self.log_raw_keystrokes:
                        self._append_key_char("\t", window_info)
                elif kname == "Enter":
                    if self.log_raw_keystrokes:
                        self._append_key_char("\n", window_info)
                    self._flush_key_buffer(reason="enter")

                self.log_event("key_press_special", {
                    "key": kname,
                    "ctrl": held_modifiers["ctrl"],
                    "alt": held_modifiers["alt"],
                    "shift": held_modifiers["shift"],
                    "meta": held_modifiers["meta"],
                    **window_info,
                })
                return

            ch: Optional[str] = None
            try:
                ch = key.char  # type: ignore[attr-defined]
            except Exception:
                ch = None

            if ch is not None:
                if (held_modifiers["meta"] or held_modifiers["ctrl"]) and ch.strip() != "":
                    self._flush_key_buffer(reason="before_hotkey")

                    hk = _hotkey_string(ch)
                    self.log_event("hotkey", {
                        "combo": hk,
                        "key": ch,
                        "ctrl": held_modifiers["ctrl"],
                        "alt": held_modifiers["alt"],
                        "shift": held_modifiers["shift"],
                        "meta": held_modifiers["meta"],
                        **window_info,
                    })
                    return

                if self.log_raw_keystrokes:
                    self._append_key_char(ch, window_info)
                return

            self.log_event("key_press_special", {
                "key": str(key),
                "ctrl": held_modifiers["ctrl"],
                "alt": held_modifiers["alt"],
                "shift": held_modifiers["shift"],
                "meta": held_modifiers["meta"],
                **window_info,
            })

        def on_key_release(key):
            if key in MOD_KEYS:
                _set_modifier(key, False)

        self.mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click, on_scroll=on_mouse_scroll)
        self.mouse_listener.start()

        self.keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.keyboard_listener.start()

        print("Input tracking started:")
        print("  - mouse: move/click/scroll")
        print("  - keyboard: special keys (Space/Tab/Enter/Backspace tracked), hotkeys (Cmd/Ctrl+Key), and raw text buffers (key_text_input)")

    # -----------------------------
    # Web screenshots
    # -----------------------------
    async def _take_web_screenshot(self, page: Page, reason: str, payload: Dict[str, Any]) -> Optional[str]:
        if not self.capture_web_screenshots:
            return None

        async with self._web_screenshot_lock:
            try:
                self._web_screenshot_seq += 1
                ts = int(time.time() * 1000)
                seq = self._web_screenshot_seq
                safe_reason = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in reason])[:60]
                fname = f"{ts}_{seq:06d}_{safe_reason}.png"
                path = self.web_screens_dir / fname
                await page.screenshot(path=str(path), full_page=False)
                return str(path.relative_to(self.session_dir))
            except Exception as e:
                self.log_event("web_screenshot_error", {"reason": reason, "error": str(e)})
                return None

    # -----------------------------
    # Correlation: nearest OS click
    # -----------------------------
    def _nearest_os_click_id(self, target_time_ms: int) -> Optional[str]:
        if not self._recent_os_clicks:
            return None
        best: Optional[Tuple[int, str]] = None
        for c in list(self._recent_os_clicks)[-40:]:
            dt = abs(int(c.get("time_ms", 0)) - int(target_time_ms))
            if dt <= self.correlate_click_window_ms:
                ieid = c.get("input_event_id")
                if not ieid:
                    continue
                if best is None or dt < best[0]:
                    best = (dt, ieid)
        return best[1] if best else None

    def _nearest_os_click_dt(self, input_event_id: str, target_time_ms: int) -> Optional[int]:
        for c in list(self._recent_os_clicks)[-80:]:
            if c.get("input_event_id") == input_event_id:
                return abs(int(c.get("time_ms", 0)) - int(target_time_ms))
        return None

    # -----------------------------
    # Browser instrumentation
    # -----------------------------
    @staticmethod
    def _frame_depth(frame) -> int:
        d = 0
        cur = frame
        while True:
            try:
                parent = cur.parent_frame
            except Exception:
                parent = None
            if not parent:
                break
            d += 1
            cur = parent
            if d > 20:
                break
        return d

    async def setup_browser_instrumentation(self, start_url: Optional[str] = None) -> Page:
        self.playwright = await async_playwright().start()

        launch_args = {"headless": False, "args": ["--disable-blink-features=AutomationControlled"]}
        if self.browser_channel:
            launch_args["channel"] = self.browser_channel

        self.browser = await self.playwright.chromium.launch(**launch_args)
        self.browser_context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )

        # ✅ Reduce overhead: tracing screenshots OFF (snapshots still useful)
        await self.browser_context.tracing.start(screenshots=False, snapshots=True, sources=True)

        page = await self.browser_context.new_page()
        self.page = page

        async def _ui_event_binding(source, payload):
            if not isinstance(payload, dict):
                payload = {"raw": payload}

            kind = (payload.get("kind") or "").lower()
            recorder_now_ms = int(time.time() * 1000)

            try:
                frame_url = source.frame.url if source and source.frame else ""
            except Exception:
                frame_url = ""
            payload.setdefault("frame_url", frame_url)
            payload.setdefault("url", page.url)

            linked = self._nearest_os_click_id(recorder_now_ms)
            if linked:
                payload["linked_input_event_id"] = linked
                dt = self._nearest_os_click_dt(linked, recorder_now_ms)
                if dt is not None:
                    payload["linked_input_dt_ms"] = dt

            # ✅ Screenshot only for configured kinds, AND throttled
            if kind in set(self.capture_web_screenshots_on or []):
                now_ms = int(time.time() * 1000)
                if (now_ms - self._last_web_shot_ms) >= self.web_screenshot_min_interval_ms:
                    rel = await self._take_web_screenshot(page, reason=f"ui_{kind}", payload=payload)
                    if rel:
                        payload["screenshot_path"] = rel
                    self._last_web_shot_ms = now_ms

            self.log_event("ui_event", {"env": "web", **payload})

        await page.expose_binding("logUiEvent", _ui_event_binding)

        await page.add_init_script(
            f"window.__FP_LOG_EACH_KEYSTROKE__ = {str(self.log_text_input_on_each_keystroke).lower()};"
        )
        await page.add_init_script(dom_script)

        loop = asyncio.get_running_loop()

        async def _log_web_navigation(kind: str, frame, url: str):
            try:
                is_main = (frame == page.main_frame)
            except Exception:
                is_main = False

            # ✅ Ignore iframe navigations unless enabled
            if (not is_main) and (not self.log_iframe_navigations):
                return

            try:
                depth = 0 if is_main else self._frame_depth(frame)
            except Exception:
                depth = 0 if is_main else 1

            try:
                frame_url = getattr(frame, "url", "") or ""
            except Exception:
                frame_url = ""

            main_url = page.url or ""

            # ✅ Dedup main navigations quickly (prevents spam "reload-like" logs)
            now_ms = int(time.time() * 1000)
            if is_main:
                if self._last_nav_url == main_url and (now_ms - self._last_nav_ms) < self.nav_dedupe_ms:
                    return
                self._last_nav_url = main_url
                self._last_nav_ms = now_ms

            data: Dict[str, Any] = {
                "env": "web",
                "kind": kind,
                "url": url,
                "main_url": main_url,
                "frame_url": frame_url,
                "frame_name": (getattr(frame, "name", "") or ""),
                "frame_is_main": bool(is_main),
                "is_iframe": bool(not is_main),
                "frame_depth": int(depth),
            }

            # ✅ Don't screenshot on navigation unless explicitly enabled
            if self.capture_web_screenshots and self.capture_nav_screenshots:
                rel = await self._take_web_screenshot(page, reason=f"nav_{kind}", payload=data)
                if rel:
                    data["screenshot_path"] = rel

            self.log_event("web_navigation", data)

        def on_frame_navigated(frame):
            try:
                loop.create_task(_log_web_navigation("framenavigated", frame, getattr(frame, "url", "") or ""))
            except Exception:
                pass

        # (Optional) Load events are noisy; keep only if you want them.
        # def on_load():
        #     try:
        #         loop.create_task(_log_web_navigation("load", page.main_frame, page.url))
        #     except Exception:
        #         pass

        page.on("framenavigated", on_frame_navigated)
        # page.on("load", on_load)

        if self.log_browser_events:
            page.on("console", lambda msg: self.log_event("console", {"env": "web", "type": msg.type, "text": msg.text, "url": page.url}))

        if start_url:
            await page.goto(start_url, wait_until="domcontentloaded")

        print(f"Browser instrumentation started ({self.browser_type}" + (f" via {self.browser_channel}" if self.browser_channel else "") + ")")
        return page

    # -----------------------------
    # Event writer
    # -----------------------------
    def start_event_writer(self):
        def write_events():
            with open(self.events_path, "w", encoding="utf-8") as f:
                while True:
                    event = self.event_queue.get()
                    if isinstance(event, dict) and event.get("event_type") == "__STOP__":
                        break
                    f.write(json.dumps(event) + "\n")
                    f.flush()

        writer_thread = threading.Thread(target=write_events, daemon=True)
        writer_thread.start()
        return writer_thread

    async def start_recording(self, enable_browser: bool = True, start_url: Optional[str] = None) -> Optional[Page]:
        print(f"\n=== Starting Activity Recording ===")
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.session_dir}")

        self.is_recording = True

        self.event_writer = self.start_event_writer()
        self.start_active_app_watcher()
        self.start_screen_recording()
        self.start_input_tracking()

        if self.voice_cfg.enabled:
            self.voice_recorder = VoiceNoteRecorder(self.session_dir, self.log_event, self.voice_cfg)
            self.voice_recorder.start()

        page = None
        if enable_browser:
            try:
                page = await self.setup_browser_instrumentation(start_url=start_url)
            except Exception as e:
                print(f"Warning: Browser setup failed: {e}")
                print("Continuing without browser instrumentation...")

        self.log_event("recording_started", {
            "session_id": self.session_id,
            "platform": platform.system(),
            "browser_enabled": enable_browser and page is not None,
            "voice_enabled": self.voice_cfg.enabled,
            "web_trace_path": str(self.browser_trace_path.relative_to(self.session_dir)),
            "web_screenshots_dir": str(self.web_screens_dir.relative_to(self.session_dir)),
            "desktop_screenshots_dir": str(self.desktop_screens_dir.relative_to(self.session_dir)),
            "correlate_click_window_ms": self.correlate_click_window_ms,
            "screen_fps": self.screen_fps,
            "frame_checkpoint_every_n": self.frame_checkpoint_every_n,
            "log_raw_keystrokes": self.log_raw_keystrokes,
            "key_buffer_idle_flush_ms": self.key_buffer_idle_flush_ms,
            "active_app_poll_ms": self.active_app_poll_ms,
            "pre_click_delay_ms": self.pre_click_delay_ms,
            "post_click_delay_ms": self.post_click_delay_ms,
            "pre_click_half_size_px": self.pre_click_half_size_px,
            "post_click_half_size_px": self.post_click_half_size_px,
            "also_capture_window_fallback": self.also_capture_window_fallback,

            # new
            "log_iframe_navigations": self.log_iframe_navigations,
            "capture_nav_screenshots": self.capture_nav_screenshots,
            "nav_dedupe_ms": self.nav_dedupe_ms,
            "web_screenshot_min_interval_ms": self.web_screenshot_min_interval_ms,
            "capture_web_screenshots_on": self.capture_web_screenshots_on,
        })

        print("\n✓ All recording components started")
        print("  - Active app change watcher")
        print("  - Screen recording (+ checkpoints)")
        print("  - Input tracking (mouse + keyboard text/hotkeys)")
        if self.voice_cfg.enabled:
            print("  - Voice notes (Silero VAD) + Deepgram transcription")
        if page:
            print("  - Browser instrumentation + DOM UI events + tracing")
        print("\nPress Ctrl+C or type STOP marker to stop\n")

        return page

    async def stop_recording(self):
        print("\n=== Stopping Activity Recording ===")
        self.is_recording = False

        self._flush_key_buffer(reason="manual")

        self._active_app_stop.set()
        if self._active_app_thread:
            self._active_app_thread.join(timeout=1)

        if self.voice_recorder:
            self.voice_recorder.stop()

        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        if self.browser_context:
            try:
                await self.browser_context.tracing.stop(path=str(self.browser_trace_path))
                await self.browser_context.close()
            except Exception as e:
                print(f"Warning: Error closing browser context: {e}")

        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                print(f"Warning: Error closing browser: {e}")

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                print(f"Warning: Error stopping playwright: {e}")

        if self.screen_recorder:
            self.screen_recorder.join(timeout=2)

        self.log_event("recording_stopped", {"session_id": self.session_id})
        self.event_queue.put({"event_type": "__STOP__"})

        if hasattr(self, "event_writer"):
            self.event_writer.join(timeout=3)

        print(f"\n✓ Recording saved to: {self.session_dir}")
        print(f"  - Video: {self.video_path.name}")
        print(f"  - Events: {self.events_path.name}")
        print(f"  - Browser trace: {self.browser_trace_path.name}")
        print(f"  - Web screenshots: web_screenshots/")
        print(f"  - Desktop screenshots: desktop_screenshots/")
        if self.voice_cfg.enabled:
            print(f"  - Voice notes: voice_notes/")


async def main():
    START_MARKER = "###WORKFLOW_RECORDING_START###"
    STOP_MARKER = "###WORKFLOW_RECORDING_STOP###"

    recorder = ActivityRecorder(
        log_browser_events=False,
        log_text_input_on_each_keystroke=False,

        # Web screenshots: default only on click/submit, throttled
        capture_web_screenshots=True,
        capture_web_screenshots_on=["click", "submit"],

        # Correlate OS click <-> DOM events
        correlate_click_window_ms=250,

        # Screen recording
        screen_fps=10.0,
        frame_checkpoint_every_n=30,

        # Keyboard
        log_raw_keystrokes=True,
        key_buffer_idle_flush_ms=800,

        # Active app polling
        active_app_poll_ms=200,

        # Click evidence
        pre_click_delay_ms=20,
        post_click_delay_ms=250,
        pre_click_half_size_px=160,
        post_click_half_size_px=380,
        also_capture_window_fallback=True,

        # New: nav noise controls
        log_iframe_navigations=False,
        capture_nav_screenshots=False,
        nav_dedupe_ms=1200,

        # New: screenshot throttle
        web_screenshot_min_interval_ms=500,

        voice_cfg=VoiceConfig(
            enabled=True,
            sample_rate=16000,
            channels=1,
            block_ms=32,

            vad_threshold=0.5,
            start_trigger_blocks=3,
            end_hangover_ms=700,
            min_note_ms=400,
            max_note_ms=30_000,
            merge_gap_ms=300,

            deepgram_model="nova-3",
            deepgram_smart_format=True,
            deepgram_punctuate=True,
            deepgram_utterances=False,
            deepgram_language=None,
        ),
    )

    print("\n" + "=" * 60)
    print("ACTIVITY RECORDER - MANUAL MODE")
    print("=" * 60)
    print("\nTo START recording, type:")
    print(f"  {START_MARKER}")
    print("\nTo STOP recording, type:")
    print(f"  {STOP_MARKER}")
    print("\nThese markers will be logged and can be filtered out later.")
    print("=" * 60 + "\n")

    while True:
        user_input = input("Waiting for start command: ").strip()
        if user_input == START_MARKER:
            print("\n✓ Start command received!\n")
            break
        else:
            print(f"Invalid command. Please type: {START_MARKER}")

    try:
        page = await recorder.start_recording(enable_browser=True, start_url=None)

        recorder.log_event("workflow_control", {
            "control_type": "START",
            "marker": START_MARKER,
            "description": "Recording started by user command",
        })

        if page:
            print("✓ Browser page available for manual browsing / automation")
        else:
            print("✓ Recording screen and input only")

        print(f"\nWhen done, type: {STOP_MARKER}\n")

        stop_event = asyncio.Event()

        def check_stop_command():
            while not stop_event.is_set():
                try:
                    user_input2 = input().strip()
                    if user_input2 == STOP_MARKER:
                        print("\n✓ Stop command received!\n")
                        stop_event.set()
                        break
                    else:
                        print(f"Invalid command. To stop, type: {STOP_MARKER}")
                except EOFError:
                    break

        input_thread = threading.Thread(target=check_stop_command, daemon=True)
        input_thread.start()

        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        recorder.log_event("workflow_control", {
            "control_type": "STOP",
            "marker": STOP_MARKER,
            "description": "Recording stopped by user command",
        })

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by Ctrl+C")
        recorder.log_event("workflow_control", {"control_type": "INTERRUPT", "description": "Recording interrupted by user (Ctrl+C)"})
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        recorder.log_event("workflow_control", {"control_type": "ERROR", "description": f"Recording stopped due to error: {str(e)}"})
    finally:
        await recorder.stop_recording()

        print("\n" + "=" * 60)
        print("WORKFLOW FILTERING")
        print("=" * 60)
        print("\nTo filter out control commands from events.jsonl:")
        print(f'  grep -v "workflow_control" {recorder.events_path}')
        print("\nOr programmatically filter events where:")
        print('  event["event_type"] != "workflow_control"')
        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
