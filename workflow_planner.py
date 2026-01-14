"""
Unified Workflow Planner
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Step dataclass
# -----------------------------
@dataclass
class Step:
    """Single automation step - compatible with WorkflowOrchestrator"""
    id: str
    action: str
    args: Dict[str, Any]
    retries: int = 2
    description: Optional[str] = None
    agent: str = "browser"  # "browser" or "desktop"
    delay_after: float = 0.5
    timestamp: Optional[str] = None
    time_ms: Optional[int] = None

    # metadata container
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


# -----------------------------
# Voice note indexing + Deepgram
# -----------------------------
class VoiceNoteIndex:
    """
    Builds an index of voice notes from events.
    Backfills missing transcripts via Deepgram (internally uses audio_path).
    """
    def __init__(self, events: List[Dict[str, Any]]):
        self.events = events
        # note_id -> {start_ms,end_ms,audio_path,text,ok}
        self.notes: Dict[str, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self):
        for e in self.events:
            et = e.get("event_type")
            if et not in ("voice_note_start", "voice_note_end", "voice_note_transcript"):
                continue
            note_id = e.get("note_id")
            if not note_id:
                continue

            note = self.notes.setdefault(note_id, {
                "note_id": note_id,
                "start_ms": None,
                "end_ms": None,
                "audio_path": None,
                "ok": None,
                "text": "",
            })

            if et == "voice_note_start":
                note["start_ms"] = e.get("time_ms") or e.get("start_ms") or note["start_ms"]
                note["audio_path"] = note["audio_path"] or e.get("audio_path")

            elif et == "voice_note_end":
                note["end_ms"] = e.get("time_ms") or e.get("end_ms") or note["end_ms"]
                note["audio_path"] = note["audio_path"] or e.get("audio_path")

            elif et == "voice_note_transcript":
                note["start_ms"] = note["start_ms"] or e.get("start_ms") or e.get("time_ms")
                note["end_ms"] = note["end_ms"] or e.get("end_ms")
                note["audio_path"] = note["audio_path"] or e.get("audio_path")
                note["ok"] = e.get("ok", True)
                note["text"] = (e.get("text") or e.get("transcript") or "").strip()

        # Normalize
        for n in self.notes.values():
            if n["start_ms"] is None and n["end_ms"] is not None:
                n["start_ms"] = n["end_ms"]
            if n["end_ms"] is None and n["start_ms"] is not None:
                n["end_ms"] = n["start_ms"]

    def missing_transcripts(self) -> List[Dict[str, Any]]:
        missing = []
        for n in self.notes.values():
            audio_path = n.get("audio_path")
            text = (n.get("text") or "").strip()
            ok = n.get("ok")
            if audio_path and (text == "" or ok is False or ok is None):
                missing.append(n)
        return missing

    @staticmethod
    def _deepgram_transcribe_file(
        wav_path: str,
        api_key: str,
        model: str = "nova-3",
        smart_format: bool = True,
        punctuate: bool = True,
        language: Optional[str] = None,
        timeout_s: int = 60,
    ) -> Dict[str, Any]:
        params = {
            "model": model,
            "smart_format": str(smart_format).lower(),
            "punctuate": str(punctuate).lower(),
        }
        if language:
            params["language"] = language

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        }

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        r = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=timeout_s,
        )
        if r.status_code >= 400:
            return {"_error": f"Deepgram HTTP {r.status_code}", "_body": r.text[:2000]}
        return r.json()

    @staticmethod
    def _extract_transcript(dg_json: Dict[str, Any]) -> str:
        try:
            alt = dg_json["results"]["channels"][0]["alternatives"][0]
            return (alt.get("transcript") or "").strip()
        except Exception:
            return ""

    def backfill_missing_transcripts(
        self,
        verbose: bool = True,
        model: str = "nova-3",
        smart_format: bool = True,
        punctuate: bool = True,
        language: Optional[str] = None,
    ) -> int:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            if verbose:
                print("⚠ DEEPGRAM_API_KEY not set; cannot backfill transcripts.")
            return 0

        filled = 0
        for note in self.missing_transcripts():
            audio_path = note.get("audio_path")
            if not audio_path:
                continue
            if not Path(audio_path).exists():
                if verbose:
                    print(f"⚠ Audio file missing for note {note['note_id']}: {audio_path}")
                continue

            if verbose:
                print(f"Deepgram backfill: {note['note_id']}")

            dg = self._deepgram_transcribe_file(
                wav_path=audio_path,
                api_key=api_key,
                model=model,
                smart_format=smart_format,
                punctuate=punctuate,
                language=language,
            )
            if "_error" in dg:
                note["ok"] = False
                note["text"] = ""
                if verbose:
                    print(f"  ❌ Deepgram error: {dg['_error']}")
                continue

            text = self._extract_transcript(dg)
            note["ok"] = True
            note["text"] = text
            filled += 1

        return filled

    def notes_overlapping_range(self, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
        hits = []
        for n in self.notes.values():
            s = n.get("start_ms")
            e = n.get("end_ms")
            if s is None or e is None:
                continue
            if e >= start_ms and s <= end_ms:
                hits.append(n)
        hits.sort(key=lambda x: x.get("start_ms") or 0)
        return hits

    def closest_note_to_time(self, t_ms: int) -> Optional[Dict[str, Any]]:
        best = None
        best_dist = None
        for n in self.notes.values():
            s = n.get("start_ms")
            e = n.get("end_ms")
            if s is None or e is None:
                continue
            if t_ms < s:
                dist = s - t_ms
            elif t_ms > e:
                dist = t_ms - e
            else:
                dist = 0
            if best_dist is None or dist < best_dist:
                best = n
                best_dist = dist
        return best


# -----------------------------
# Form filled details extraction
# -----------------------------
class FormFillIndex:
    """
    Extracts final filled details from recorder events of type: event_type="form_interaction"
    """

    def __init__(self, events: List[Dict[str, Any]]):
        self.events = events
        self.filled_details: Dict[str, Any] = {}
        self._build()

    @staticmethod
    def _norm_q(q: str) -> str:
        q = (q or "").strip()
        q = " ".join(q.split())
        return q

    def _build(self):
        interactions = [e for e in self.events if e.get("event_type") == "form_interaction"]
        interactions.sort(key=lambda x: (x.get("time_ms") if isinstance(x.get("time_ms"), int) else 10**18))

        checkbox_sets: Dict[str, set] = {}

        for e in interactions:
            q = self._norm_q(e.get("question") or "")
            if not q:
                fallback = self._norm_q(e.get("name") or e.get("aria_label") or "")
                q = fallback or ""
            if not q:
                continue

            kind = (e.get("field_kind") or e.get("tag") or e.get("role") or "").lower()
            choice = (e.get("choice") or e.get("label") or "").strip()
            value = e.get("value")
            checked = e.get("checked")
            selected = e.get("selected")

            if isinstance(checked, str):
                checked = checked.lower() == "true"
            if isinstance(selected, str):
                selected = selected.lower() == "true"

            if kind == "radio":
                if checked is True or checked is None:
                    if choice:
                        self.filled_details[q] = choice
                    elif value not in (None, ""):
                        self.filled_details[q] = value
                continue

            if kind == "checkbox":
                s = checkbox_sets.setdefault(q, set())
                if not choice and value not in (None, ""):
                    choice = str(value)

                if checked is True:
                    if choice:
                        s.add(choice)
                elif checked is False:
                    if choice and choice in s:
                        s.remove(choice)

                self.filled_details[q] = sorted(list(s))
                continue

            if kind == "select":
                if choice:
                    self.filled_details[q] = choice
                elif value not in (None, ""):
                    self.filled_details[q] = value
                continue

            if kind == "option":
                if selected is True or selected is None:
                    if choice:
                        self.filled_details[q] = choice
                continue

            if kind in ("input", "textarea"):
                if value is None:
                    value = ""
                self.filled_details[q] = str(value)
                continue

            if choice:
                self.filled_details[q] = choice
            elif value not in (None, ""):
                self.filled_details[q] = str(value)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.filled_details)


# -----------------------------
# Analyzer
# -----------------------------
class UnifiedEventAnalyzer:
    BROWSER_WINDOWS = ['Google Chrome', 'Chrome']
    BROWSER_EVENT_TYPES = ['navigation', 'page_load']

    def __init__(self, events: List[Dict]):
        def _sort_key(e: Dict[str, Any]):
            tm = e.get("time_ms")
            if isinstance(tm, int):
                return (0, tm)
            return (1, e.get("timestamp", ""))

        self.events = sorted(events, key=_sort_key)
        self.timeline = self._build_timeline()
        self.voice_index = VoiceNoteIndex(self.events)
        self.form_index = FormFillIndex(self.events)

    def _is_browser_event(self, event: Dict) -> bool:
        window_title = event.get('window_title', '')
        event_type = event.get('event_type', '')
        url = event.get('url', '')

        if window_title and any(browser in window_title for browser in self.BROWSER_WINDOWS):
            return True
        if event_type in self.BROWSER_EVENT_TYPES:
            return True
        if url and url != '':
            return True
        return False

    def _build_timeline(self) -> List[Dict]:
        timeline = []
        for event in self.events:
            agent = 'browser' if self._is_browser_event(event) else 'desktop'
            timeline.append({
                'event': event,
                'agent': agent,
                'timestamp': event.get('timestamp', ''),
                'time_ms': event.get('time_ms', None),
                'app_name': event.get('app_name', ''),
                'window_title': event.get('window_title', '')
            })
        return timeline

    def get_search_query(self, url: str) -> Optional[str]:
        try:
            if 'google.com/search' in url and 'q=' in url:
                query = url.split('q=')[1].split('&')[0]
                return query.replace('+', ' ')
        except Exception:
            pass
        return None

    def is_google_form_url(self, url: str) -> bool:
        u = (url or "").lower()
        return ("forms.gle/" in u) or ("docs.google.com/forms" in u)

    def extract_form_id_from_url(self, url: str) -> Optional[str]:
        u = (url or "").strip()
        if "docs.google.com/forms/d/e/" in u:
            try:
                part = u.split("docs.google.com/forms/d/e/")[1]
                form_id = part.split("/")[0]
                return form_id or None
            except Exception:
                return None
        return None


# -----------------------------
# Planner
# -----------------------------
class UnifiedWorkflowPlanner:
    def __init__(self, events: List[Dict], deepgram_backfill: bool = True):
        self.analyzer = UnifiedEventAnalyzer(events)
        self.step_counter = 0
        self.steps: List[Dict[str, Any]] = []

        # extracted form state
        self.form_filled_details: Dict[str, Any] = self.analyzer.form_index.as_dict()
        self.form_id: Optional[str] = None
        self.form_url: Optional[str] = None

        # Keep last planned step time (for fallback anchoring)
        self._last_time_ms: Optional[int] = None

        if deepgram_backfill:
            filled = self.analyzer.voice_index.backfill_missing_transcripts(verbose=True)
            if filled:
                print(f"✓ Backfilled {filled} missing voice transcript(s) via Deepgram")

        self._form_session_bounds: Optional[Tuple[int, int]] = None  # (start_ms, end_ms)

    def _next_step_id(self) -> str:
        self.step_counter += 1
        return f"step_{self.step_counter}"

    # -------- Voice metadata (transcript-only) --------
    def _voice_transcript_for_span(self, start_ms: int, end_ms: int) -> str:
        notes = self.analyzer.voice_index.notes_overlapping_range(start_ms, end_ms)
        texts = [(n.get("text") or "").strip() for n in notes if (n.get("text") or "").strip()]

        if texts:
            return " ".join(texts).strip()

        mid = int((start_ms + end_ms) / 2)
        n = self.analyzer.voice_index.closest_note_to_time(mid)
        if n:
            return (n.get("text") or "").strip()

        return ""

    def _attach_voice_metadata(self, step: Step, span_ms: Optional[Tuple[int, int]] = None):
        # If we have neither a step time nor explicit span, we cannot align to any voice note.
        if not step.time_ms and not span_ms:
            return

        if span_ms:
            s, e = span_ms
        else:
            s = int(step.time_ms) - 1500
            e = int(step.time_ms) + 1500

        transcript = self._voice_transcript_for_span(s, e)

        step.metadata = step.metadata or {}
        step.metadata["voice_note"] = {
            "start_ms": s,
            "end_ms": e,
            "transcript": transcript
        }

    # -----------------------------
    # Step adding
    # -----------------------------
    def _add_step(
        self,
        action: str,
        args: Dict,
        agent: str,
        description: str = None,
        retries: int = 2,
        delay_after: float = 0.5,
        timestamp: str = None,
        time_ms: Optional[int] = None,
        voice_span_ms: Optional[Tuple[int, int]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        step = Step(
            id=self._next_step_id(),
            action=action,
            args=args,
            retries=retries,
            description=description,
            agent=agent,
            delay_after=delay_after,
            timestamp=timestamp,
            time_ms=time_ms,
            metadata=None
        )

        # update last time anchor if present
        if isinstance(time_ms, int):
            self._last_time_ms = time_ms

        # Attach transcript-only voice metadata
        self._attach_voice_metadata(step, span_ms=voice_span_ms)

        # Attach any extra metadata
        if extra_metadata:
            step.metadata = step.metadata or {}
            for k, v in extra_metadata.items():
                step.metadata[k] = v

        self.steps.append(step.to_dict())

    # -----------------------------
    # Helpers for Google Form anchors
    # -----------------------------
    def _infer_form_session_bounds(self, form_url: str) -> Tuple[int, int]:
        start_ms = None
        end_ms = None

        for item in self.analyzer.timeline:
            e = item["event"]
            if item["agent"] != "browser":
                continue
            if e.get("event_type") == "navigation":
                url = (e.get("url") or "").strip()
                if url and url.startswith(form_url):
                    start_ms = item.get("time_ms") or e.get("time_ms")
                    break

        if start_ms is None:
            for item in self.analyzer.timeline:
                e = item["event"]
                if item["agent"] != "browser":
                    continue
                url = (e.get("url") or "").strip()
                if url and self.analyzer.is_google_form_url(url):
                    start_ms = item.get("time_ms") or e.get("time_ms")
                    break

        last_tm = None
        for item in self.analyzer.timeline:
            tm = item.get("time_ms") or item["event"].get("time_ms")
            if not isinstance(tm, int):
                continue
            last_tm = tm

        if start_ms is None:
            start_ms = next((it.get("time_ms") for it in self.analyzer.timeline if isinstance(it.get("time_ms"), int)), 0)

        for item in self.analyzer.timeline:
            if item["agent"] != "browser":
                continue
            tm = item.get("time_ms") or item["event"].get("time_ms")
            if isinstance(tm, int) and tm >= start_ms:
                end_ms = tm

        if end_ms is None:
            end_ms = last_tm if isinstance(last_tm, int) else start_ms + 10_000

        if end_ms < start_ms + 5000:
            end_ms = start_ms + 5000

        return int(start_ms), int(end_ms)

    def _form_step_anchors(self, form_url: str) -> Dict[str, Tuple[int, int, int]]:
        if not self._form_session_bounds:
            self._form_session_bounds = self._infer_form_session_bounds(form_url)

        start_ms, end_ms = self._form_session_bounds
        span = end_ms - start_ms

        goto_t = start_ms
        extract_t = start_ms + int(0.08 * span)
        fill_t = start_ms + int(0.45 * span)
        submit_t = start_ms + int(0.90 * span)

        return {
            "goto": (goto_t, goto_t - 1000, goto_t + 6000),
            "extract_form_id": (extract_t, extract_t - 1000, extract_t + 5000),
            "fill_google_form": (fill_t, fill_t - 3000, fill_t + 20000),
            "submit_form": (submit_t, submit_t - 5000, submit_t + 8000),
        }

    def _coerce_form_data_for_agent(self, filled: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (filled or {}).items():
            kk = " ".join((k or "").split()).strip()
            if not kk:
                continue
            if isinstance(v, list):
                out[kk] = [str(x) for x in v if str(x).strip()]
            elif v is None:
                out[kk] = ""
            else:
                out[kk] = str(v)
        return out

    # -----------------------------
    # NEW: Format form details for desktop notes
    # -----------------------------
    def _format_form_filled_for_notes(self) -> str:
        lines: List[str] = []
        lines.append("Google Form submission notes")
        if self.form_url:
            lines.append(f"Form URL: {self.form_url}")
        if self.form_id:
            lines.append(f"Form ID (best-effort): {self.form_id}")
        lines.append("")
        lines.append("Filled details:")
        if not self.form_filled_details:
            lines.append("- (no filled_details captured)")
        else:
            for k, v in self.form_filled_details.items():
                if isinstance(v, list):
                    vv = ", ".join([str(x) for x in v])
                else:
                    vv = str(v)
                lines.append(f"- {k}: {vv}")
        lines.append("")
        return "\n".join(lines)

    # -----------------------------
    # NEW: Desktop anchoring using real logs
    # -----------------------------
    def _find_desktop_anchor_times_for_app(
        self,
        app_name: str,
        after_ms: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (launch_time_ms, type_time_ms) best-effort from real desktop logs.

        launch_time_ms:
          - first desktop event where app_name matches OR window_title contains app_name (case-insensitive)
          - optionally constrained to >= after_ms

        type_time_ms:
          - first likely keyboard/text event after launch (best-effort) while app still matches
          - if can't find, returns None
        """
        app_l = (app_name or "").lower()

        def is_app_event(item: Dict[str, Any]) -> bool:
            if item.get("agent") != "desktop":
                return False
            e = item.get("event") or {}
            wn = (e.get("window_title") or "").lower()
            an = (e.get("app_name") or "").lower()
            return (app_l in an) or (app_l in wn)

        def time_ok(tm: Any) -> bool:
            return isinstance(tm, int) and (after_ms is None or tm >= after_ms)

        # 1) find launch-ish time: first time we see the app active
        launch_tm: Optional[int] = None
        launch_idx: Optional[int] = None
        for idx, item in enumerate(self.analyzer.timeline):
            tm = item.get("time_ms") or (item.get("event") or {}).get("time_ms")
            if not time_ok(tm):
                continue
            if is_app_event(item):
                launch_tm = int(tm)
                launch_idx = idx
                break

        if launch_tm is None:
            return None, None

        # 2) find first typing-ish event after that
        type_tm: Optional[int] = None
        KEYWORD_TYPES = {
            "key_down", "key_up", "key_press", "keyboard",
            "text_input", "type", "keystroke"
        }

        for item in self.analyzer.timeline[launch_idx:]:
            if item.get("agent") != "desktop":
                continue
            e = item.get("event") or {}
            tm = item.get("time_ms") or e.get("time_ms")
            if not isinstance(tm, int):
                continue

            # Prefer events while the same app is active
            wn = (e.get("window_title") or "").lower()
            an = (e.get("app_name") or "").lower()
            in_app = (app_l in an) or (app_l in wn)
            if not in_app:
                continue

            et = (e.get("event_type") or "").lower()
            if et in KEYWORD_TYPES:
                type_tm = int(tm)
                break

            # Also treat "key" substring as typing-ish
            if "key" in et or "text" in et:
                type_tm = int(tm)
                break

        return launch_tm, type_tm

    def _fallback_anchor_ms(self) -> int:
        """
        If no desktop evidence exists, anchor to the latest planned time or latest timeline time.
        """
        if isinstance(self._last_time_ms, int) and self._last_time_ms > 0:
            return int(self._last_time_ms)

        tms = [it.get("time_ms") for it in self.analyzer.timeline if isinstance(it.get("time_ms"), int)]
        if tms:
            return int(max(tms))
        return 0

    # -----------------------------
    # Main planning
    # -----------------------------
    def analyze_and_plan(self) -> Dict[str, Any]:
        browser_events = sum(1 for item in self.analyzer.timeline if item['agent'] == 'browser')
        desktop_events = sum(1 for item in self.analyzer.timeline if item['agent'] == 'desktop')

        print(f"\nTimeline Analysis:")
        print(f"  Browser events: {browser_events}")
        print(f"  Desktop events: {desktop_events}")

        nav_urls: List[str] = []
        first_form_url: Optional[str] = None
        all_searches: List[str] = []

        for item in self.analyzer.timeline:
            e = item["event"]
            if item["agent"] != "browser":
                continue
            if e.get("event_type") != "navigation":
                continue
            url = (e.get("url") or "").strip()
            if not url or "about:blank" in url or "about:srcdoc" in url:
                continue
            if url in nav_urls:
                continue
            nav_urls.append(url)

            if not first_form_url and self.analyzer.is_google_form_url(url):
                first_form_url = url

            if 'google.com/search' in url and 'q=' in url:
                q = self.analyzer.get_search_query(url)
                if q and q not in all_searches:
                    all_searches.append(q)

        # Prefer the exact form URL if present
        if first_form_url:
            self.form_url = first_form_url
            self.form_id = self.analyzer.extract_form_id_from_url(first_form_url)

            print(f"\nDetected Google Form URL: {first_form_url}")
            if self.form_id:
                print(f"Best-effort Form ID (from URL): {self.form_id}")

            if self.form_filled_details:
                print(f"✓ Extracted {len(self.form_filled_details)} filled field(s) from form_interaction events")
            else:
                print("⚠ No form_interaction events found; filled_details will be empty (check recorder DOM logger).")

            self._plan_google_form_flow(first_form_url)

            # ✅ Desktop steps (TextEdit) + voice metadata anchored from real logs if possible
            notes_text = self._format_form_filled_for_notes()
            self._plan_desktop_typing_and_save(search_query=None, text_override=notes_text)

            return self._build_workflow_output(
                f"Fill Google Form and submit: {first_form_url}",
                form_url=first_form_url,
                form_id=self.form_id,
                filled_details=self.form_filled_details
            )

        # Generic browsing flow
        print("\nPlanning generic browsing workflow...")

        last_url = None
        for item in self.analyzer.timeline:
            event = item["event"]
            agent = item["agent"]
            timestamp = item.get("timestamp")
            time_ms = item.get("time_ms") or event.get("time_ms")

            if agent == "browser" and event.get("event_type") == "navigation":
                url = (event.get("url") or "").strip()
                if not url or "about:blank" in url or "about:srcdoc" in url:
                    continue
                if url == last_url:
                    continue
                last_url = url

                self._add_step(
                    action="goto",
                    args={"url": url},
                    agent="browser",
                    description=f"Goto {url}",
                    retries=2,
                    delay_after=1.0,
                    timestamp=timestamp,
                    time_ms=time_ms
                )

                self._add_step(
                    action="scroll",
                    args={"direction": "down", "amount": 300},
                    agent="browser",
                    description="Scroll to load content",
                    retries=1,
                    delay_after=0.6,
                    timestamp=timestamp,
                    time_ms=time_ms
                )

        self._add_step(
            action="extract_text",
            args={"locator": "body", "description": "Main page content"},
            agent="browser",
            description="Extract page text into page_content",
            retries=2,
            delay_after=0.8,
        )

        self._plan_desktop_typing_and_save(None, text_override=None)

        return self._build_workflow_output(None)

    # -----------------------------
    # Google Forms flow planning
    # -----------------------------
    def _plan_google_form_flow(self, form_url: str) -> None:
        anchors = self._form_step_anchors(form_url)

        # 1) goto form
        t, s, e = anchors["goto"]
        self._add_step(
            action="goto",
            args={"url": form_url},
            agent="browser",
            description="Open Google Form",
            retries=1,
            delay_after=1.2,
            time_ms=t,
            voice_span_ms=(s, e)
        )

        # 2) extract_form_id
        t, s, e = anchors["extract_form_id"]
        self._add_step(
            action="extract_form_id",
            args={},
            agent="browser",
            description="Extract Form ID from header (best-effort)",
            retries=1,
            delay_after=0.3,
            time_ms=t,
            voice_span_ms=(s, e)
        )

        # Build agent-ready form_data from extracted DOM events
        form_data_for_agent = self._coerce_form_data_for_agent(self.form_filled_details)

        # 3) fill_google_form
        t, s, e = anchors["fill_google_form"]
        self._add_step(
            action="fill_google_form",
            args={
                "passes": 2,
                "form_data": form_data_for_agent
            },
            agent="browser",
            description="Fill Google Form fields",
            retries=0,
            delay_after=0.6,
            time_ms=t,
            voice_span_ms=(s, e)
        )

        # 4) submit_form
        t, s, e = anchors["submit_form"]
        self._add_step(
            action="submit_form",
            args={
                "require_no_missing_required": True,
                "return_filled_details": True,
                "return_form_id": True,
            },
            agent="browser",
            description="Submit form and return (form_id, filled_details, fill_results)",
            retries=2,
            delay_after=0.8,
            time_ms=t,
            voice_span_ms=(s, e),
            extra_metadata={
                "form_filled": {
                    "form_url": self.form_url,
                    "form_id_best_effort": self.form_id,
                    "filled_details": self.form_filled_details
                }
            }
        )

    # -----------------------------
    # Desktop planning
    # -----------------------------
    def _plan_desktop_typing_and_save(self, search_query: Optional[str], text_override: Optional[str] = None) -> None:
        app_name = "TextEdit"

        # Prefer anchoring from real desktop logs (TextEdit + typing)
        after_ms = self._fallback_anchor_ms()
        launch_tm, type_tm = self._find_desktop_anchor_times_for_app(app_name=app_name, after_ms=after_ms)

        # If logs didn't capture TextEdit, fallback to synthetic near the anchor
        if not isinstance(launch_tm, int):
            launch_tm = after_ms + 1500
        if not isinstance(type_tm, int):
            type_tm = int(launch_tm) + 1200

        self._add_step(
            action="launch_fullscreen",
            args={"app_name": app_name},
            agent="desktop",
            description=f"Launch {app_name}",
            retries=2,
            delay_after=1.2,
            time_ms=int(launch_tm),
            voice_span_ms=(int(launch_tm) - 1500, int(launch_tm) + 4000),
        )

        text_to_type = text_override if text_override is not None else "{{page_content}}"

        self._add_step(
            action="type_text",
            args={"text": text_to_type},
            agent="desktop",
            description="Type notes/content into TextEdit",
            retries=1,
            delay_after=0.5,
            time_ms=int(type_tm),
            voice_span_ms=(int(type_tm) - 2000, int(type_tm) + 6000),
        )

    # -----------------------------
    # Output + metadata
    # -----------------------------
    def _build_workflow_output(
        self,
        search_query: Optional[str],
        form_url: Optional[str] = None,
        form_id: Optional[str] = None,
        filled_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        intent = f"Automated workflow: {search_query}" if search_query else "Perform automated task"

        out: Dict[str, Any] = {
            "description": intent,
            "steps": self.steps
        }

        if form_url or form_id or filled_details is not None:
            out["form_filled"] = {
                "form_url": form_url,
                "form_id_best_effort": form_id,
                "filled_details": filled_details or {}
            }

        return out

    def get_metadata(self, search_query: Optional[str] = None) -> Dict[str, Any]:
        browser_count = sum(1 for s in self.steps if s['agent'] == 'browser')
        desktop_count = sum(1 for s in self.steps if s['agent'] == 'desktop')

        voice_notes_total = len(self.analyzer.voice_index.notes)
        voice_notes_with_text = sum(1 for n in self.analyzer.voice_index.notes.values() if (n.get("text") or "").strip())

        form_interactions = sum(1 for e in self.analyzer.events if e.get("event_type") == "form_interaction")
        filled_fields = len(self.form_filled_details or {})

        return {
            "generated_at": datetime.now().isoformat(),
            "user_intent": search_query or "Perform task",
            "total_steps": len(self.steps),
            "browser_steps": browser_count,
            "desktop_steps": desktop_count,
            "total_events_analyzed": len(self.analyzer.events),
            "sequential": True,
            "mixed_agents": True,
            "voice_notes_total": voice_notes_total,
            "voice_notes_with_transcripts": voice_notes_with_text,
            "form_interaction_events": form_interactions,
            "form_filled_fields_extracted": filled_fields,
            "improvements": [
                "Voice notes attached to steps as metadata.voice_note.transcript (transcript-only)",
                "Deepgram backfill used if transcript missing in logs",
                "Google Form synthetic steps get inferred time_ms anchors so metadata is present",
                "Extracted Google Form filled_details from event_type=form_interaction and attached to workflow + submit_form step metadata",
                "FIX: Desktop steps (TextEdit) are now appended for Google Form workflows too",
                "FIX: Desktop steps now get time_ms + voice metadata (prefer real desktop logs, fallback to anchors)"
            ]
        }


# -----------------------------
# File loading / CLI utilities
# -----------------------------
def load_events_from_file(events_file: str, filter_control_events: bool = True) -> List[Dict]:
    if not os.path.exists(events_file):
        uploads_path = f"/mnt/user-data/uploads/{events_file}"
        if os.path.exists(uploads_path):
            events_file = uploads_path
        else:
            raise FileNotFoundError(f"Events file not found: {events_file}")

    events = []
    control_events = []

    print(f"Reading events from: {events_file}")

    with open(events_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if filter_control_events and event.get('event_type') == 'workflow_control':
                    control_events.append(event)
                    continue
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

    if not events:
        raise ValueError("No valid events found in file")

    print(f"✓ Loaded {len(events)} events for workflow generation")
    return events


def generate_workflow_from_events(
    events_file: str,
    output_file: str = None,
    filter_control_events: bool = True
) -> Dict[str, Any]:
    events = load_events_from_file(events_file, filter_control_events=filter_control_events)

    print("\n" + "=" * 80)
    print("UNIFIED WORKFLOW PLANNER V3.3 (VOICE TRANSCRIPT-ONLY + FORM FILLED DETAILS)")
    print("=" * 80)

    planner = UnifiedWorkflowPlanner(events, deepgram_backfill=True)
    workflow = planner.analyze_and_plan()
    metadata = planner.get_metadata(workflow.get("description"))

    if output_file:
        full_output = {"metadata": metadata, "workflow": workflow}
        with open(output_file, 'w') as f:
            json.dump(full_output, f, indent=2)
        print(f"\nFull workflow saved to: {output_file}")

    return workflow


def get_latest_recording_folder(base_dir="recordings"):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Recordings directory '{base_dir}' not found")

    folders = [f for f in Path(base_dir).iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No recording folders found in '{base_dir}'")

    latest_folder = max(folders, key=lambda x: x.stat().st_mtime)
    return latest_folder


def main():
    if len(sys.argv) > 1:
        events_file = sys.argv[1]
    else:
        latest_folder = get_latest_recording_folder()
        events_file = latest_folder / "events.jsonl"
        print(f"Using latest recording: {latest_folder.name}")

    if not Path(events_file).exists():
        print(f"Error: events.jsonl not found in {Path(events_file).parent}")
        return None

    events_path = Path(events_file)
    output_file = events_path.parent / f"{events_path.stem}_workflow_voice_form.json"

    try:
        workflow = generate_workflow_from_events(str(events_file), str(output_file))
        print("Workflow generated successfully.")
        return workflow
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
