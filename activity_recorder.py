"""
Comprehensive Activity Recorder
Captures screen recording, browser instrumentation, OS-level events,
AND voice notes (Silero VAD segmentation + Deepgram transcription).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading
import queue

# Screen recording
import cv2
import numpy as np
from PIL import ImageGrab

# Browser instrumentation
from playwright.async_api import async_playwright, Page

# OS-level event tracking
from pynput import mouse, keyboard
import psutil
import platform

# Voice
import sounddevice as sd
import requests
import torch
from AppKit import NSWorkspace

from dotenv import load_dotenv

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

    # Silero VAD params
    vad_threshold: float = 0.5
    start_trigger_blocks: int = 3        # require N consecutive "speech" blocks
    end_hangover_ms: int = 700           # end after this much silence
    min_note_ms: int = 400               # discard tiny notes
    max_note_ms: int = 30_000            # force split long notes
    merge_gap_ms: int = 300              # if speech resumes quickly, merge

    # Deepgram
    deepgram_model: str = "nova-3"
    deepgram_smart_format: bool = True
    deepgram_punctuate: bool = True
    deepgram_utterances: bool = False   # set True if you want utterance segmentation
    deepgram_language: Optional[str] = None  # e.g., "en-US"


class VoiceNoteRecorder:
    """
    Always-on mic capture -> Silero VAD -> writes wav segments -> Deepgram transcribes
    Emits events back into the main recorder's event queue with correct time_ms boundaries.
    """

    def __init__(
        self,
        session_dir: Path,
        event_log_fn,  # callable(event_type: str, data: dict)
        voice_cfg: VoiceConfig,
    ):
        self.session_dir = session_dir
        self.voice_cfg = voice_cfg
        self.log_event = event_log_fn

        self.voice_dir = self.session_dir / "voice_notes"
        self.voice_dir.mkdir(exist_ok=True)

        self._stop = threading.Event()
        self._audio_q: "queue.Queue[tuple[int, bytes]]" = queue.Queue(maxsize=200)  # (time_ms, pcm16bytes)
        self._segment_q: "queue.Queue[dict]" = queue.Queue()

        self._capture_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._stt_thread: Optional[threading.Thread] = None

        # Silero VAD model
        self._vad_model = None

        # Deepgram
        self._dg_key = os.getenv("DEEPGRAM_API_KEY")

    def start(self):
        if not self.voice_cfg.enabled:
            return

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
        })

    def stop(self):
        if not self.voice_cfg.enabled:
            return

        self._stop.set()

        # Let threads drain queues briefly
        for t in [self._capture_thread, self._vad_thread, self._stt_thread]:
            if t:
                t.join(timeout=2)

        self.log_event("voice_recorder_stopped", {})

    # -----------------------------
    # Silero loading
    # -----------------------------
    def _load_silero(self):
        """
        Uses torch.hub to load Silero VAD.
        """
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

    # -----------------------------
    # Audio capture
    # -----------------------------
    def _capture_loop(self):
        cfg = self.voice_cfg
        blocksize = int(cfg.sample_rate * (cfg.block_ms / 1000.0))  # samples

        def callback(indata, frames, time_info, status):
            if self._stop.is_set():
                raise sd.CallbackStop
            if status:
                print(f"[voice] sounddevice status: {status}")

            # RawInputStream gives a cffi buffer, not numpy
            pcm16 = bytes(indata)
            ts_ms = int(time.time() * 1000)

            try:
                self._audio_q.put_nowait((ts_ms, pcm16))
            except queue.Full:
                pass

        try:
            with sd.RawInputStream(
                samplerate=cfg.sample_rate,
                channels=cfg.channels,
                dtype="int16",
                blocksize=blocksize,
                callback=callback,
            ):
                while not self._stop.is_set():
                    time.sleep(0.05)
        except Exception as e:
            self.log_event("voice_error", {"stage": "capture", "error": str(e)})

    # -----------------------------
    # VAD + segmentation
    # -----------------------------
    def _vad_prob(self, pcm16: bytes) -> float:
        """
        Returns speech probability for a frame.
        Silero expects float tensor in [-1, 1].
        """
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
            wf.setsampwidth(2)  # int16
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

        end_hangover_ms = cfg.end_hangover_ms
        min_note_ms = cfg.min_note_ms
        max_note_ms = cfg.max_note_ms

        def finalize_note(end_ms: int):
            nonlocal speech_on, note_start_ms, last_speech_ms, buffered, speech_blocks

            if note_start_ms is None:
                speech_on = False
                buffered = []
                speech_blocks = 0
                last_speech_ms = None
                return

            dur = end_ms - note_start_ms
            if dur < min_note_ms:
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
                        note_start_ms = ts_ms
                        buffered = [pcm16]
                else:
                    speech_blocks = 0
            else:
                buffered.append(pcm16)

                if note_start_ms is not None and (ts_ms - note_start_ms) >= max_note_ms:
                    finalize_note(ts_ms)
                    continue

                if last_speech_ms is not None:
                    silence_ms = ts_ms - last_speech_ms
                    if silence_ms >= end_hangover_ms:
                        finalize_note(last_speech_ms)

    # -----------------------------
    # Deepgram transcription
    # -----------------------------
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

        headers = {
            "Authorization": f"Token {self._dg_key}",
            "Content-Type": "audio/wav",
        }

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        r = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )
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
                    word_items.append({
                        "word": w.get("word", ""),
                        "start_ms": ws,
                        "end_ms": we,
                        "confidence": w.get("confidence", None),
                    })
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
        # NEW: controls how chatty the DOM logger is
        log_text_input_on_each_keystroke: bool = False,  # False => only log on change/blur
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.browser_type = browser_type
        self.browser_channel = browser_channel
        self.log_browser_events = log_browser_events
        self.log_text_input_on_each_keystroke = log_text_input_on_each_keystroke

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)

        self.is_recording = False
        self.event_queue = queue.Queue()

        self.video_path = self.session_dir / "screen_recording.avi"
        self.events_path = self.session_dir / "events.jsonl"
        self.browser_trace_path = self.session_dir / "browser_trace.zip"

        # Components
        self.screen_recorder = None
        self.mouse_listener = None
        self.keyboard_listener = None
        self.browser_context = None
        self.browser = None
        self.playwright = None
        self.page = None

        # Voice
        self.voice_cfg = voice_cfg or VoiceConfig(enabled=True)
        self.voice_recorder: Optional[VoiceNoteRecorder] = None

    def get_active_window_info(self) -> dict:
        try:
            if platform.system() == "Windows":
                hwnd = win32gui.GetForegroundWindow()
                window_title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    app_name = process.name()
                except Exception:
                    app_name = "Unknown"
                return {"window_title": window_title, "app_name": app_name}

            elif platform.system() == "Darwin":
                workspace = NSWorkspace.sharedWorkspace()
                active_app = workspace.activeApplication()
                return {
                    "window_title": active_app.get("NSApplicationName", "Unknown"),
                    "app_name": active_app.get("NSApplicationName", "Unknown"),
                }

            elif platform.system() == "Linux":
                d = display.Display()
                window = d.get_input_focus().focus
                wmname = window.get_wm_name()
                wmclass = window.get_wm_class()
                return {
                    "window_title": wmname if wmname else "Unknown",
                    "app_name": wmclass[1] if wmclass else "Unknown",
                }
        except Exception as e:
            return {"window_title": "Error", "app_name": str(e)}

        return {"window_title": "Unknown", "app_name": "Unknown"}

    def log_event(self, event_type: str, data: dict):
        event = {
            "timestamp": datetime.now().isoformat(),
            "time_ms": int(time.time() * 1000),
            "event_type": event_type,
            **data,
        }
        self.event_queue.put(event)

    def start_screen_recording(self):
        def record_screen():
            screen = ImageGrab.grab()
            width, height = screen.size

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(str(self.video_path), fourcc, 10.0, (width, height))

            print(f"Screen recording started: {width}x{height}")

            while self.is_recording:
                img = ImageGrab.grab()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                time.sleep(0.1)

            out.release()
            print("Screen recording stopped")

        self.screen_recorder = threading.Thread(target=record_screen, daemon=True)
        self.screen_recorder.start()

    def start_input_tracking(self):
        def on_mouse_move(x, y):
            window_info = self.get_active_window_info()
            self.log_event("mouse_move", {"x": x, "y": y, **window_info})

        def on_mouse_click(x, y, button, pressed):
            window_info = self.get_active_window_info()
            self.log_event(
                "mouse_click",
                {"x": x, "y": y, "button": str(button), "pressed": pressed, **window_info},
            )

        def on_mouse_scroll(x, y, dx, dy):
            window_info = self.get_active_window_info()
            self.log_event("mouse_scroll", {"x": x, "y": y, "dx": dx, "dy": dy, **window_info})

        def on_key_press(key):
            window_info = self.get_active_window_info()
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key)
            self.log_event("key_press", {"key": key_char, **window_info})

        def on_key_release(key):
            window_info = self.get_active_window_info()
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key)
            self.log_event("key_release", {"key": key_char, **window_info})

        self.mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click, on_scroll=on_mouse_scroll)
        self.mouse_listener.start()

        self.keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.keyboard_listener.start()

        print("Input tracking started")

    async def setup_browser_instrumentation(self):
        """
        UPDATED:
        - Exposes a Python binding (logFormEvent) for DOM-level form events
        - Injects init script that captures:
          - radio/checkbox selections (label text)
          - dropdown selections
          - text field values
        """
        try:
            self.playwright = await async_playwright().start()
            browser_launcher = getattr(self.playwright, self.browser_type)

            launch_args = {"headless": False, "args": ["--disable-blink-features=AutomationControlled"]}
            if self.browser_channel:
                launch_args["channel"] = self.browser_channel

            self.browser = await browser_launcher.launch(**launch_args)
            self.browser_context = await self.browser.new_context(
                viewport={"width": 1280, "height": 720},
                ignore_https_errors=True,
            )

            await self.browser_context.tracing.start(screenshots=True, snapshots=True, sources=True)

            page = await self.browser_context.new_page()
            self.page = page

            # -----------------------------
            # NEW: Expose a binding to receive DOM form events
            # -----------------------------
            async def _form_event_binding(source, payload):
                # payload is a JSON-serializable dict created in the page
                if not isinstance(payload, dict):
                    payload = {"raw": payload}
                payload.setdefault("url", page.url)
                self.log_event("form_interaction", payload)

            await page.expose_binding("logFormEvent", _form_event_binding)

            # -----------------------------
            # NEW: Inject JS that captures form interactions
            # -----------------------------
            dom_logger_script = r"""
(() => {
  const SHOULD_LOG_INPUT = window.__FP_LOG_EACH_KEYSTROKE__ === true;

  function nowMs() { return Date.now(); }
  function safeText(t) {
    if (!t) return "";
    return String(t).replace(/\s+/g, " ").trim();
  }
  function takeLines(text) {
    return String(text || "").split("\n").map(safeText).filter(Boolean);
  }

  function elementDescriptor(el) {
    if (!el) return {};
    const tag = (el.tagName || "").toLowerCase();
    const type = el.getAttribute && el.getAttribute("type");
    const role = el.getAttribute && el.getAttribute("role");
    const name = el.getAttribute && (el.getAttribute("aria-label") || el.getAttribute("name") || el.getAttribute("id")) || "";
    return { tag, type: type || "", role: role || "", name: safeText(name) };
  }

  function resolveIdRefs(idList) {
    const ids = String(idList || "").trim().split(/\s+/).filter(Boolean);
    if (!ids.length) return "";
    const parts = [];
    for (const id of ids) {
      const n = document.getElementById(id);
      if (!n) continue;
      const t = safeText(n.textContent || n.innerText || "");
      if (t) parts.push(t);
    }
    return safeText(parts.join(" "));
  }

  function labelTextForInput(el) {
    if (!el) return "";
    const id = el.getAttribute && el.getAttribute("id");
    if (id) {
      const lbl = document.querySelector(`label[for="${CSS.escape(id)}"]`);
      if (lbl) return safeText(lbl.innerText);
    }
    const wrap = el.closest && el.closest("label");
    if (wrap) return safeText(wrap.innerText);
    const aria = el.getAttribute && el.getAttribute("aria-label");
    if (aria) return safeText(aria);
    return "";
  }

  function choiceTextForOption(el) {
    if (!el) return "";

    const aria = el.getAttribute && el.getAttribute("aria-label");
    if (aria) {
      const t = safeText(aria);
      if (t) return t.slice(0, 200);
    }

    const by = el.getAttribute && el.getAttribute("aria-labelledby");
    if (by) {
      const t = resolveIdRefs(by);
      if (t) return t.slice(0, 200);
    }

    const txt = safeText(el.innerText || el.textContent || "");
    if (!txt) return "";
    const lines = takeLines(txt);
    if (!lines.length) return txt.slice(0, 200);
    lines.sort((a,b)=>a.length-b.length);
    return lines[0].slice(0, 200);
  }

  function collectOptionLabels(container) {
    if (!container) return new Set();
    const els = Array.from(container.querySelectorAll(
      '[role="checkbox"],[role="radio"],[role="option"],input[type="checkbox"],input[type="radio"]'
    ));
    const set = new Set();
    for (const el of els) {
      const d = elementDescriptor(el);
      const t = (d.tag === "input") ? labelTextForInput(el) : choiceTextForOption(el);
      const norm = safeText(t).toLowerCase();
      if (norm) set.add(norm);
    }
    return set;
  }

  // NEW: stable title extraction for *any* question, anchored at listitem
  function titleFromListItem(li) {
    if (!li) return "";

    // 1) Best: role=heading inside listitem
    const h = li.querySelector('[role="heading"]');
    if (h) {
      const t = safeText(h.textContent || h.innerText || "");
      if (t) return t.slice(0, 220);
    }

    // 2) aria-labelledby / aria-describedby on the listitem itself
    if (li.getAttribute) {
      const lb = li.getAttribute("aria-labelledby");
      const db = li.getAttribute("aria-describedby");
      if (lb) {
        const t = resolveIdRefs(lb);
        if (t) return t.slice(0, 220);
      }
      if (db) {
        const t = resolveIdRefs(db);
        if (t) return t.slice(0, 220);
      }
    }

    // 3) fallback: first non-option line from listitem.innerText
    const optionLabels = collectOptionLabels(li);
    const lines = takeLines(li.innerText || "");
    for (const l of lines.slice(0, 15)) {
      const norm = l.toLowerCase();
      if (optionLabels.has(norm)) continue;
      if (/^\d+$/.test(norm)) continue;
      // avoid generic UI noise
      if (norm === "required" || norm.includes("clear selection")) continue;
      if (l) return l.slice(0, 220);
    }

    return "";
  }

  // For option clicks: prefer ancestor ARIA refs; for inputs: prefer listitem title
  function findQuestionTitle(target) {
    if (!target) return "";

    const d = elementDescriptor(target);
    const li = target.closest && target.closest('[role="listitem"]');

    // Text-like inputs: anchor directly on listitem
    if (d.tag === "input" || d.tag === "textarea" || d.tag === "select") {
      const t = titleFromListItem(li);
      if (t) return t;
      // as last resort, try ARIA walk
      // (kept below)
    }

    // Option-like targets: walk up for aria-labelledby/aria-describedby and filter out option labels
    const optionLabels = collectOptionLabels(li || target.closest('form') || document.body);

    const clickedChoice = safeText(choiceTextForOption(target)).toLowerCase();
    if (clickedChoice) optionLabels.add(clickedChoice);

    const candidates = [];
    let el = target;
    for (let depth = 0; el && depth < 12; depth++) {
      if (el.getAttribute) {
        const lb = el.getAttribute("aria-labelledby");
        const db = el.getAttribute("aria-describedby");
        if (lb) {
          const t = resolveIdRefs(lb);
          if (t) candidates.push(t);
        }
        if (db) {
          const t = resolveIdRefs(db);
          if (t) candidates.push(t);
        }
      }
      el = el.parentElement;
    }

    // Also include listitem title as a candidate
    if (li) {
      const t = titleFromListItem(li);
      if (t) candidates.push(t);
    }

    const cleaned = candidates
      .map(safeText)
      .filter(Boolean)
      .filter(t => !optionLabels.has(t.toLowerCase()))
      .filter(t => t.length >= 2);

    if (!cleaned.length) return "";

    function score(t) {
      const s = t.toLowerCase();
      let sc = 0;
      if (s.includes("?")) sc += 5;
      sc += Math.min(10, Math.floor(t.length / 20));
      if (s.includes("clear selection") || s === "submit" || s === "clear form") sc -= 10;
      return sc;
    }

    cleaned.sort((a,b) => score(b) - score(a));
    return cleaned[0].slice(0, 220);
  }

  function getValueSnapshot(target) {
    const d = elementDescriptor(target);

    if (d.tag === "input" || d.tag === "textarea") {
      const inputType = (d.type || "").toLowerCase();
      if (inputType === "checkbox" || inputType === "radio") {
        return {
          field_kind: inputType,
          checked: !!target.checked,
          value: target.value ?? "",
          label: labelTextForInput(target)
        };
      }
      return {
        field_kind: d.tag,
        value: target.value ?? "",
        label: labelTextForInput(target)
      };
    }

    if (d.tag === "select") {
      const idx = target.selectedIndex;
      const opt = idx >= 0 ? target.options[idx] : null;
      return {
        field_kind: "select",
        value: target.value ?? "",
        choice: opt ? safeText(opt.textContent) : ""
      };
    }

    const role = (d.role || "").toLowerCase();
    if (role === "radio" || role === "checkbox") {
      const checked = target.getAttribute("aria-checked");
      return {
        field_kind: role,
        checked: checked === "true",
        choice: choiceTextForOption(target)
      };
    }
    if (role === "option") {
      const selected = target.getAttribute("aria-selected");
      return {
        field_kind: "option",
        selected: selected === "true",
        choice: choiceTextForOption(target)
      };
    }

    return { field_kind: d.tag || role || "unknown" };
  }

  const lastByKey = new Map();
  function dedupeKey(target) {
    if (!target) return "null";
    const d = elementDescriptor(target);
    const role = d.role || "";
    const name = d.name || "";
    const tag = d.tag || "";
    return `${tag}|${role}|${name}|${target.className || ""}`.slice(0, 300);
  }

  function emit(kind, target, extra = {}) {
    try {
      if (!window.logFormEvent) return;

      const snap = getValueSnapshot(target);
      const qtxt = findQuestionTitle(target);

      const payload = {
        kind,
        time_ms: nowMs(),
        question: qtxt,
        ...elementDescriptor(target),
        ...snap,
        ...extra
      };

      const key = `${kind}::${dedupeKey(target)}`;
      const sval = JSON.stringify(payload);
      const prev = lastByKey.get(key);
      if (prev === sval) return;
      lastByKey.set(key, sval);

      window.logFormEvent(payload);
    } catch (e) {}
  }

  function onInput(e) {
    const t = e.target;
    if (!t) return;
    const tag = (t.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") {
      const type = (t.getAttribute("type") || "").toLowerCase();
      if (type === "radio" || type === "checkbox") return;
      if (!SHOULD_LOG_INPUT) return;
      emit("input", t);
    }
  }

  function onChange(e) {
    const t = e.target;
    if (!t) return;
    emit("change", t);
  }

  function onClick(e) {
    const t = e.target;
    if (!t) return;

    const el = t.closest && (
      t.closest('[role="checkbox"]') ||
      t.closest('[role="radio"]') ||
      t.closest('[role="option"]')
    );
    if (el) {
      setTimeout(() => emit("click", el), 0);
      return;
    }

    const tag = (t.tagName || "").toLowerCase();
    if (tag === "input") {
      const type = (t.getAttribute("type") || "").toLowerCase();
      if (type === "radio" || type === "checkbox") {
        setTimeout(() => emit("click", t), 0);
      }
    }
  }

  document.addEventListener("input", onInput, true);
  document.addEventListener("change", onChange, true);
  document.addEventListener("click", onClick, true);

  window.__FP_FORM_LOGGER_READY__ = true;
})();
"""

            # Pass one flag into the page so you can control chattiness (keystrokes vs only final)
            await page.add_init_script(
                f"window.__FP_LOG_EACH_KEYSTROKE__ = {str(self.log_text_input_on_each_keystroke).lower()};"
            )
            await page.add_init_script(dom_logger_script)

            # Navigate AFTER we inject the init scripts so it applies to the form page
            await self.page.goto("https://forms.gle/QtA2djT82q62M2Dz8", wait_until="domcontentloaded")

            # Basic navigation instrumentation
            def sync_log_page_load():
                self.log_event("page_load", {"url": page.url, "title": "Loading..."})

            page.on("load", sync_log_page_load)
            page.on("framenavigated", lambda frame: self.log_event("navigation", {"url": frame.url, "name": frame.name}))

            if self.log_browser_events:
                page.on("console", lambda msg: self.log_event("console", {"type": msg.type, "text": msg.text, "url": page.url}))

                def log_request(request):
                    if request.resource_type in ["document", "xhr", "fetch"]:
                        self.log_event("network_request", {
                            "url": request.url,
                            "method": request.method,
                            "resource_type": request.resource_type,
                            "page_url": page.url,
                        })

                def log_response(response):
                    request = response.request
                    if request.resource_type in ["document", "xhr", "fetch"]:
                        self.log_event("network_response", {"url": response.url, "status": response.status, "page_url": page.url})

                page.on("request", log_request)
                page.on("response", log_response)

            print(
                f"Browser instrumentation started ({self.browser_type}"
                + (f" via {self.browser_channel}" if self.browser_channel else "")
                + ")"
            )
            print("✓ DOM form interaction logger enabled (event_type=form_interaction)")
            return page

        except Exception as e:
            print(f"Error in browser setup: {e}")
            import traceback
            traceback.print_exc()
            raise

    def start_event_writer(self):
        def write_events():
            with open(self.events_path, "w") as f:
                while self.is_recording:
                    try:
                        event = self.event_queue.get(timeout=0.1)
                        f.write(json.dumps(event) + "\n")
                        f.flush()
                    except queue.Empty:
                        continue

                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        f.write(json.dumps(event) + "\n")
                    except queue.Empty:
                        break
                f.flush()

        writer_thread = threading.Thread(target=write_events, daemon=True)
        writer_thread.start()
        return writer_thread

    async def start_recording(self, enable_browser: bool = True) -> Optional[Page]:
        print(f"\n=== Starting Activity Recording ===")
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.session_dir}")

        self.is_recording = True

        # Start event writer
        self.event_writer = self.start_event_writer()

        # Start screen recording
        self.start_screen_recording()

        # Start input tracking
        self.start_input_tracking()

        # Start voice recorder (VAD + Deepgram)
        if self.voice_cfg.enabled:
            self.voice_recorder = VoiceNoteRecorder(self.session_dir, self.log_event, self.voice_cfg)
            self.voice_recorder.start()

        # Setup browser (optional)
        page = None
        if enable_browser:
            try:
                page = await self.setup_browser_instrumentation()
            except Exception as e:
                print(f"Warning: Browser setup failed: {e}")
                print("Continuing without browser instrumentation...")

        self.log_event("recording_started", {
            "session_id": self.session_id,
            "platform": platform.system(),
            "browser_enabled": enable_browser and page is not None,
            "voice_enabled": self.voice_cfg.enabled,
        })

        print("\n✓ All recording components started")
        print("  - Screen recording")
        print("  - Input tracking (mouse/keyboard)")
        if self.voice_cfg.enabled:
            print("  - Voice notes (Silero VAD) + Deepgram transcription")
        if page:
            print("  - Browser instrumentation + DOM form events")
        print("\nPress Ctrl+C or call stop_recording() to stop\n")

        return page

    async def stop_recording(self):
        print("\n=== Stopping Activity Recording ===")
        self.is_recording = False

        # Stop voice
        if self.voice_recorder:
            self.voice_recorder.stop()

        # Stop input listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        # Save browser trace and close browser
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

        # Wait for screen recorder to finish
        if self.screen_recorder:
            self.screen_recorder.join(timeout=2)

        # Wait for event writer
        if hasattr(self, "event_writer"):
            self.event_writer.join(timeout=2)

        self.log_event("recording_stopped", {"session_id": self.session_id})

        print(f"\n✓ Recording saved to: {self.session_dir}")
        print(f"  - Video: {self.video_path.name}")
        print(f"  - Events: {self.events_path.name}")
        print(f"  - Browser trace: {self.browser_trace_path.name}")
        if self.voice_cfg.enabled:
            print(f"  - Voice notes: voice_notes/")


async def main():
    START_MARKER = "###WORKFLOW_RECORDING_START###"
    STOP_MARKER = "###WORKFLOW_RECORDING_STOP###"

    recorder = ActivityRecorder()

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
        page = await recorder.start_recording(enable_browser=True)

        recorder.log_event("workflow_control", {
            "control_type": "START",
            "marker": START_MARKER,
            "description": "Recording started by user command",
        })

        if page:
            print("✓ Browser page available for automation")
        else:
            print("✓ Recording screen and input only")

        print(f"\nWhen done, type: {STOP_MARKER}\n")

        stop_event = asyncio.Event()

        def check_stop_command():
            while not stop_event.is_set():
                try:
                    user_input = input().strip()
                    if user_input == STOP_MARKER:
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
