#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import os
import math
import requests
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# -----------------------------
# URL canonicalization (stable compare)
# -----------------------------
NOISE_QUERY_PARAMS = {
    "zx", "sei", "ved", "usg", "ei",
    "fbzx", "pli", "sourceid",
    "no_sw_cr", "rlz", "oq", "gs_lcrp", "sclient",
}

def canonicalize_url(url: str) -> str:
    """Remove tracking params and fragments so “same page” compares equal."""
    u = (url or "").strip()
    if not u:
        return ""
    try:
        parts = urlsplit(u)
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in NOISE_QUERY_PARAMS]
        q_sorted = urlencode(sorted(q))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, q_sorted, ""))
    except Exception:
        return u

# -----------------------------
# Desktop OCR (/observe)
# -----------------------------
def call_observe(base_url: str, image_path: Path, timeout: int = 120) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/observe"
    if not image_path.exists():
        return {"ok": False, "error": f"image_not_found: {str(image_path)}"}

    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "image/png")}
        try:
            resp = requests.post(endpoint, files=files, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

def _norm_bbox_to_px(bn: List[float], w: int, h: int) -> List[int]:
    return [
        int(round(bn[0] * w)),
        int(round(bn[1] * h)),
        int(round(bn[2] * w)),
        int(round(bn[3] * h)),
    ]

def _contains_point(b: List[int], x: int, y: int) -> bool:
    x1, y1, x2, y2 = b
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

def _bbox_center(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

def resolve_click_to_ocr_element(
    observe_json: Dict[str, Any],
    click_x: int,
    click_y: int,
    img_w: int,
    img_h: int,
    *,
    min_conf: float = 0.55,
    near_radius_px: int = 60,
) -> Dict[str, Any]:
    """
    Map a click (image-local px) onto OCR elements.

    Strategy:
      1) hit-test: click inside bbox
      2) else nearest bbox center within radius
    """
    if not observe_json.get("ok"):
        return {"ok": False, "error": observe_json.get("error", "observe_ok_false")}

    elems = observe_json.get("elements") or []
    best_hit = None
    best_near = None

    for el in elems:
        try:
            conf = float(el.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf < min_conf:
            continue

        bn = el.get("bbox_norm")
        if not bn or len(bn) != 4:
            continue

        bbox_px = _norm_bbox_to_px(bn, img_w, img_h)

        if _contains_point(bbox_px, click_x, click_y):
            area = max(1, (bbox_px[2] - bbox_px[0]) * (bbox_px[3] - bbox_px[1]))
            score = (conf, -area)  # prefer higher conf, then smaller box
            if (best_hit is None) or (score > best_hit["score"]):
                best_hit = {
                    "score": score,
                    "text": (el.get("text") or "").strip(),
                    "confidence": conf,
                    "bbox_px": bbox_px,
                    "element_id": el.get("id", None),
                }
            continue

        cx, cy = _bbox_center(bbox_px)
        d = _dist(cx, cy, float(click_x), float(click_y))
        if d <= near_radius_px:
            score = conf - (d / max(near_radius_px, 1))
            if (best_near is None) or (score > best_near["score"]):
                best_near = {
                    "score": score,
                    "text": (el.get("text") or "").strip(),
                    "confidence": conf,
                    "bbox_px": bbox_px,
                    "element_id": el.get("id", None),
                    "distance_px": float(d),
                }

    if best_hit:
        out = dict(best_hit)
        out.pop("score", None)
        return {"ok": True, "method": "hit_test", **out}

    if best_near:
        out = dict(best_near)
        out.pop("score", None)
        return {"ok": True, "method": "nearest", **out}

    return {"ok": False, "error": "no_match"}

# -----------------------------
# Shared utils
# -----------------------------
def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def ev_id(ev: Dict[str, Any]) -> str:
    return str(ev.get("event_id") or "")

def is_blank(url: str) -> bool:
    u = (url or "").strip().lower()
    return u in ("", "about:blank")

def load_events(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if isinstance(ev, dict) and "event_type" in ev and "time_ms" in ev:
                    out.append(ev)
            except Exception:
                pass
    out.sort(key=lambda e: safe_int(e.get("time_ms")))
    return out

def looks_like_browser_window(ev: Dict[str, Any]) -> bool:
    title = (ev.get("window_title") or "").lower()
    app = (ev.get("app_name") or "").lower()
    bundle = (ev.get("bundle_id") or "").lower()
    return (
        any(x in title for x in ("chrome", "chromium", "firefox", "safari", "edge"))
        or any(x in app for x in ("chrome", "chromium", "firefox", "safari", "edge"))
        or any(x in bundle for x in ("chrome", "chromium", "firefox", "safari", "edge"))
    )

def looks_like_terminal_window(ev: Dict[str, Any]) -> bool:
    app = (ev.get("app_name") or "").lower()
    title = (ev.get("window_title") or "").lower()
    bundle = (ev.get("bundle_id") or "").lower()

    if "terminal" in app or "terminal" in title:
        return True
    if "iterm" in app or "iterm" in title:
        return True
    if "com.apple.terminal" in bundle:
        return True
    if "com.googlecode.iterm2" in bundle:
        return True
    return False

def _dedupe_chain(urls: List[str]) -> List[str]:
    out: List[str] = []
    for u in urls:
        if is_blank(u):
            continue
        if not out or out[-1] != u:
            out.append(u)
    return out

def _safe_str(x: Any, limit: int = 500) -> str:
    s = str(x or "")
    return s if len(s) <= limit else s[:limit]

# -----------------------------
# Browser UI intent helpers
# -----------------------------
def _is_submitish_click(ui_ev: Dict[str, Any]) -> bool:
    if (ui_ev.get("kind") or "").lower() != "click":
        return False
    txt = " ".join([
        _safe_str(ui_ev.get("a11y_name"), 200),
        _safe_str(ui_ev.get("text_snippet"), 200),
        _safe_str(ui_ev.get("name_attr"), 200),
        _safe_str(ui_ev.get("id"), 200),
        _safe_str(ui_ev.get("classes"), 200),
    ]).lower()
    return bool(re.search(r"\b(submit|send|next|continue|finish|done|confirm|ok)\b", txt))

# -----------------------------
# Browser UI steps
# -----------------------------
@dataclass
class BrowserUIStep:
    step_id: str
    t_ms: int
    t_end_ms: int
    kind: str
    url: str = ""
    frame_url: str = ""
    is_iframe: bool = False
    target: Dict[str, Any] = field(default_factory=dict)
    value: Dict[str, Any] = field(default_factory=dict)
    evidence_event_ids: List[str] = field(default_factory=list)
    summary: str = ""
    verify_hint: Dict[str, Any] = field(default_factory=dict)

def _target_spec_from_ui(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_ref": ev.get("target_ref"),
        "element_id": ev.get("element_id"),
        "element_id_tier": ev.get("element_id_tier"),
        "origin": ev.get("origin"),
        "selector_candidates": ev.get("selector_candidates") or [],
        "a11y_name": ev.get("a11y_name"),
        "tag": ev.get("tag"),
        "role": ev.get("role"),
        "type": ev.get("type"),
        "id": ev.get("id"),
        "name_attr": ev.get("name_attr"),
        "element_fingerprint": ev.get("element_fingerprint"),
        "css_path": ev.get("css_path"),
        "xpath": ev.get("xpath"),
        "bbox_norm": ev.get("bbox_norm"),
        "click_norm": ev.get("click_norm"),
        "text_snippet": ev.get("text_snippet"),
        "classes": ev.get("classes"),
    }

def build_browser_ui_steps(
    events: List[Dict[str, Any]],
    *,
    dedupe_within_ms: int = 450,
    merge_type_within_ms: int = 1200,
    emit_focus_steps: bool = False,
) -> List[BrowserUIStep]:
    ui = [e for e in events if e.get("event_type") == "ui_event" and e.get("env") == "web"]
    ui.sort(key=lambda e: safe_int(e.get("time_ms")))

    steps: List[BrowserUIStep] = []
    last_key: Optional[Tuple[str, str, str]] = None
    last_t: Optional[int] = None

    def _target_key(ev: Dict[str, Any]) -> str:
        return (
            str(ev.get("target_ref") or "").strip()
            or str(ev.get("element_id") or "").strip()
            or str(ev.get("element_fingerprint") or "").strip()
            or ""
        )

    def _mk_step(kind: str, ev: Dict[str, Any], value: Dict[str, Any], summary: str, verify_hint: Dict[str, Any]) -> BrowserUIStep:
        t = safe_int(ev.get("time_ms"))
        return BrowserUIStep(
            step_id=f"browser_ui_{len(steps)+1}",
            t_ms=t,
            t_end_ms=t,
            kind=kind,
            url=str(ev.get("url") or ""),
            frame_url=str(ev.get("frame_url") or ""),
            is_iframe=bool(ev.get("is_iframe") is True),
            target=_target_spec_from_ui(ev),
            value=value,
            evidence_event_ids=[ev_id(ev)],
            summary=summary,
            verify_hint=verify_hint,
        )

    for ev in ui:
        kind = (ev.get("kind") or "").lower()
        t = safe_int(ev.get("time_ms"))

        tk = _target_key(ev)
        a11y = _safe_str(ev.get("a11y_name"), 200)

        semantic_kind: Optional[str] = None
        value: Dict[str, Any] = {}
        summary = ""
        verify_hint: Dict[str, Any] = {}

        field_kind = (ev.get("field_kind") or "").lower()
        tag = (ev.get("tag") or "").lower()
        role = (ev.get("role") or "").lower()
        typ = (ev.get("type") or "").lower()

        if kind == "focusin" and emit_focus_steps:
            if role in ("dialog", "textbox") or tag in ("input", "textarea") or field_kind in ("input", "textarea", "contenteditable", "div"):
                semantic_kind = "browser_focus"
                summary = f"Focus {a11y or role or tag or 'element'}"
                verify_hint = {
                    "type": "browser_focus_verified",
                    "strategy": "element_focused",
                    "checks": [{"op": "element_is_focused", "target": "same_target"}],
                }

        if semantic_kind is None and (kind == "submit" or _is_submitish_click(ev)):
            semantic_kind = "browser_submit"
            summary = f"Submit ({a11y or 'form/action'})"
            verify_hint = {
                "type": "browser_submit_verified",
                "strategy": "url_changed_or_confirmation",
                "checks": [
                    {"op": "url_changed"},
                    {"op": "url_contains_any", "values": ["response", "success", "thank", "complete", "formresponse"]},
                ],
            }

        elif semantic_kind is None and kind == "click":
            semantic_kind = "browser_click"
            summary = f"Click {a11y or ev.get('text_snippet') or ev.get('role') or ev.get('tag') or 'element'}"
            verify_hint = {
                "type": "browser_click_verified",
                "strategy": "state_change_or_focus",
                "checks": [{"op": "element_exists", "target": "same_target"}, {"op": "optional_url_change"}],
            }

        elif semantic_kind is None and kind == "change":
            if (
                field_kind in ("checkbox", "radio", "select")
                or role in ("checkbox", "radio", "option")
                or typ in ("checkbox", "radio")
                or tag == "select"
            ):
                semantic_kind = "browser_select"
                value = {
                    "field_kind": field_kind or role or typ or tag,
                    "checked": ev.get("checked", None),
                    "value": ev.get("value", None),
                    "selected_text": ev.get("selected_text", None),
                }
                summary = f"Select {a11y or 'option'}"
                verify_hint = {
                    "type": "browser_select_verified",
                    "strategy": "element_state",
                    "checks": [{"op": "element_state_matches", "target": "same_target", "value": value}],
                }
            else:
                if field_kind in ("input", "textarea", "contenteditable") or tag in ("input", "textarea") or ev.get("field_kind") == "contenteditable":
                    semantic_kind = "browser_type"
                    value = {"field_kind": field_kind or tag, "text": ev.get("value", None)}
                    typed = _safe_str(value.get("text"), 80)
                    summary = f'Type into {a11y or "field"}: "{typed}"'
                    verify_hint = {
                        "type": "browser_type_verified",
                        "strategy": "element_value",
                        "checks": [{"op": "element_value_equals", "target": "same_target", "value": value.get("text")}],
                    }

        if not semantic_kind:
            continue

        try:
            sval = json.dumps(value, sort_keys=True)
        except Exception:
            sval = str(value)

        k = (semantic_kind, tk, sval)

        if last_key == k and last_t is not None and (t - last_t) <= dedupe_within_ms:
            if steps:
                steps[-1].t_end_ms = max(steps[-1].t_end_ms, t)
                steps[-1].evidence_event_ids = list(dict.fromkeys(steps[-1].evidence_event_ids + [ev_id(ev)]))
            last_t = t
            continue

        if semantic_kind == "browser_type" and steps:
            prev = steps[-1]
            prev_tk = (
                str(prev.target.get("target_ref") or "").strip()
                or str(prev.target.get("element_id") or "").strip()
                or str(prev.target.get("element_fingerprint") or "").strip()
            )
            if prev.kind == "browser_type" and prev_tk == tk and (t - prev.t_end_ms) <= merge_type_within_ms:
                prev.t_end_ms = max(prev.t_end_ms, t)
                prev.evidence_event_ids = list(dict.fromkeys(prev.evidence_event_ids + [ev_id(ev)]))
                if value.get("text") is not None:
                    prev.value["text"] = value.get("text")
                    typed2 = _safe_str(value.get("text"), 80)
                    prev.summary = f'Type into {prev.target.get("a11y_name") or "field"}: "{typed2}"'
                last_key = k
                last_t = t
                continue

        steps.append(_mk_step(semantic_kind, ev, value, summary, verify_hint))
        last_key = k
        last_t = t

    for i, s in enumerate(steps, start=1):
        s.step_id = f"browser_ui_{i}"

    return steps

# -----------------------------
# Browser navigation steps
# -----------------------------
@dataclass
class BrowserNavStep:
    step_id: str
    t_anchor_ms: int
    t_start_ms: int
    t_end_ms: int
    anchor_type: str
    anchor: Dict[str, Any] = field(default_factory=dict)
    final_url: str = ""
    final_url_canon: str = ""
    main_frame_chain: List[str] = field(default_factory=list)
    kept_nav_events: List[str] = field(default_factory=list)
    summary: str = ""
    verify_hint: Dict[str, Any] = field(default_factory=dict)
    caused_by: Dict[str, Any] = field(default_factory=dict)

def is_main_frame_nav(ev: Dict[str, Any]) -> bool:
    if ev.get("event_type") != "web_navigation":
        return False
    if ev.get("env") != "web":
        return False
    if ev.get("frame_is_main") is True:
        return True
    if ev.get("is_iframe") is False:
        return True
    return False

def _collapse_typed_query_anchors(
    anchors: List[Tuple[int, str, Dict[str, Any]]],
    *,
    collapse_window_ms: int = 1200,
) -> List[Tuple[int, str, Dict[str, Any]]]:
    out: List[Tuple[int, str, Dict[str, Any]]] = []
    for (t, typ, payload) in anchors:
        if typ != "typed_query_os":
            out.append((t, typ, payload))
            continue

        if not out or out[-1][1] != "typed_query_os":
            out.append((t, typ, payload))
            continue

        prev_t, _, _prev_payload = out[-1]
        if (t - prev_t) <= collapse_window_ms:
            out[-1] = (t, typ, payload)
        else:
            out.append((t, typ, payload))
    return out

def collect_web_anchors(
    events: List[Dict[str, Any]],
    browser_ui_steps: Optional[List[BrowserUIStep]] = None,
) -> List[Tuple[int, str, Dict[str, Any]]]:
    anchors: List[Tuple[int, str, Dict[str, Any]]] = []

    if browser_ui_steps:
        for s in browser_ui_steps:
            if s.kind in ("browser_click", "browser_submit", "browser_select", "browser_type"):
                anchors.append((
                    int(s.t_ms),
                    f"ui_step:{s.kind}",
                    {
                        "step_id": s.step_id,
                        "kind": s.kind,
                        "url": s.url,
                        "frame_url": s.frame_url,
                        "target_ref": s.target.get("target_ref"),
                        "element_id": s.target.get("element_id"),
                        "element_id_tier": s.target.get("element_id_tier"),
                        "origin": s.target.get("origin"),
                        "element_fingerprint": s.target.get("element_fingerprint"),
                        "a11y_name": s.target.get("a11y_name"),
                        "css_path": s.target.get("css_path"),
                        "xpath": s.target.get("xpath"),
                        "selector_candidates": s.target.get("selector_candidates") or [],
                        "event_ids": list(s.evidence_event_ids),
                    },
                ))

    for ev in events:
        et = ev.get("event_type")
        t = safe_int(ev.get("time_ms"))

        if et == "mouse_click" and ev.get("pressed") is False and looks_like_browser_window(ev):
            anchors.append((
                t,
                "os_click",
                {
                    "x": ev.get("x"),
                    "y": ev.get("y"),
                    "button": ev.get("button"),
                    "app_name": ev.get("app_name"),
                    "window_title": ev.get("window_title"),
                    "pid": ev.get("pid"),
                    "bundle_id": ev.get("bundle_id"),
                    "event_id": ev_id(ev),
                },
            ))
            continue

        if et == "key_press_special" and ev.get("key") == "Enter" and looks_like_browser_window(ev):
            anchors.append((
                t,
                "enter_os",
                {
                    "key": "Enter",
                    "app_name": ev.get("app_name"),
                    "window_title": ev.get("window_title"),
                    "pid": ev.get("pid"),
                    "bundle_id": ev.get("bundle_id"),
                    "event_id": ev_id(ev),
                },
            ))
            continue

        if et == "key_text_input" and looks_like_browser_window(ev):
            txt = str(ev.get("text") or "")
            if txt.strip():
                anchors.append((
                    t,
                    "typed_query_os",
                    {
                        "text": txt,
                        "reason": ev.get("reason"),
                        "length": ev.get("length"),
                        "app_name": ev.get("app_name"),
                        "window_title": ev.get("window_title"),
                        "pid": ev.get("pid"),
                        "bundle_id": ev.get("bundle_id"),
                        "event_id": ev_id(ev),
                    },
                ))

    anchors.sort(key=lambda x: x[0])
    anchors = _collapse_typed_query_anchors(anchors, collapse_window_ms=1200)
    return anchors

def build_browser_nav_steps(
    events: List[Dict[str, Any]],
    anchors: List[Tuple[int, str, Dict[str, Any]]],
    ui_steps: Optional[List[BrowserUIStep]] = None,
    *,
    pre_ms: int,
    post_ms: int,
    min_gap_ms: int = 600,
    merge_same_url_within_ms: int = 1500,
) -> List[BrowserNavStep]:
    main_nav = [e for e in events if is_main_frame_nav(e)]
    main_nav.sort(key=lambda e: safe_int(e.get("time_ms")))

    if not main_nav:
        return []

    bursts: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []

    def t(ev): return safe_int(ev.get("time_ms"))

    for ev in main_nav:
        if not cur:
            cur = [ev]
            continue
        if (t(ev) - t(cur[-1])) <= min_gap_ms:
            cur.append(ev)
        else:
            bursts.append(cur)
            cur = [ev]
    if cur:
        bursts.append(cur)

    anchors_sorted = sorted(anchors, key=lambda x: x[0])

    def nav_url(ev: Dict[str, Any]) -> str:
        return str(ev.get("url") or ev.get("main_url") or "")

    def anchor_rank(a_type: str, payload: Dict[str, Any]) -> int:
        if a_type.startswith("ui_step:"):
            kind = str(payload.get("kind") or "")
            if kind == "browser_submit":
                return 300
            if kind == "browser_click":
                return 290
            if kind == "browser_select":
                return 200
            if kind == "browser_type":
                return 150
            return 120
        if a_type == "enter_os":
            return 220
        if a_type == "os_click":
            return 180
        if a_type == "typed_query_os":
            reason = str(payload.get("reason") or "").lower()
            return 140 if reason == "enter" else 20
        return 0

    def pick_best_anchor(burst_start: int) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        lookback_start = burst_start - max(1500, pre_ms)
        candidates = [a for a in anchors_sorted if lookback_start <= a[0] <= burst_start]
        if not candidates:
            return None
        best = None
        for a in candidates:
            at, a_type, payload = a
            r = anchor_rank(a_type, payload)
            score = (r, -abs(burst_start - at))
            if best is None or score > best[0]:
                best = (score, a)
        return best[1] if best else None

    steps: List[BrowserNavStep] = []
    used_ids: set[str] = set()

    for burst in bursts:
        burst = [e for e in burst if ev_id(e) and ev_id(e) not in used_ids]
        if not burst:
            continue

        t_start = min(t(e) for e in burst)
        t_end = max(t(e) for e in burst)

        raw_urls = [nav_url(e) for e in burst]
        urls = _dedupe_chain(raw_urls)
        if not urls:
            for e in burst:
                used_ids.add(ev_id(e))
            continue

        final_url = urls[-1]
        final_canon = canonicalize_url(final_url)

        if steps and canonicalize_url(steps[-1].final_url) == final_canon:
            for e in burst:
                used_ids.add(ev_id(e))
            continue

        picked = pick_best_anchor(t_start)

        if picked:
            t_anchor, a_type, a_payload = picked
        else:
            t_anchor, a_type, a_payload = (t_start, "nav_event", {"event_id": ev_id(burst[0])})

        caused_by: Dict[str, Any] = {"anchor_type": a_type}
        if a_type.startswith("ui_step:") and a_payload.get("step_id"):
            caused_by.update({"step_id": a_payload.get("step_id"), "kind": a_payload.get("kind")})
        else:
            caused_by.update({"anchor_event_id": a_payload.get("event_id")})

        new_step = BrowserNavStep(
            step_id=f"browser_nav_{len(steps)+1}",
            t_anchor_ms=int(t_anchor),
            t_start_ms=int(t_start),
            t_end_ms=int(t_end),
            anchor_type=a_type,
            anchor=a_payload,
            final_url=final_url,
            final_url_canon=final_canon,
            main_frame_chain=urls,
            kept_nav_events=[ev_id(e) for e in burst],
            summary=f"Navigate → {final_url}",
            verify_hint={
                "type": "browser_nav_verified",
                "strategy": "url_contains_canonical",
                "checks": [{"op": "url_contains", "value": final_canon}],
            },
            caused_by=caused_by,
        )

        if steps:
            prev = steps[-1]
            if canonicalize_url(prev.final_url) == final_canon and abs(new_step.t_anchor_ms - prev.t_anchor_ms) <= merge_same_url_within_ms:
                prev.t_start_ms = min(prev.t_start_ms, new_step.t_start_ms)
                prev.t_end_ms = max(prev.t_end_ms, new_step.t_end_ms)
                prev.kept_nav_events = list(dict.fromkeys(prev.kept_nav_events + new_step.kept_nav_events))
                prev.main_frame_chain = _dedupe_chain(prev.main_frame_chain + new_step.main_frame_chain)
                prev.anchor.setdefault("merged_anchors", [])
                prev.anchor["merged_anchors"].append(
                    {"t_anchor_ms": new_step.t_anchor_ms, "anchor_type": new_step.anchor_type, "anchor": new_step.anchor}
                )
                for e in burst:
                    used_ids.add(ev_id(e))
                continue

        steps.append(new_step)
        for e in burst:
            used_ids.add(ev_id(e))

    for i, s in enumerate(steps, start=1):
        s.step_id = f"browser_nav_{i}"

    return steps

# -----------------------------
# Desktop steps
# -----------------------------
@dataclass
class DesktopStep:
    step_id: str
    t_ms: int
    t_end_ms: int
    kind: str
    app_name: str = ""
    window_title: str = ""
    pid: Optional[int] = None
    bundle_id: Optional[str] = None
    evidence_event_ids: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    verify_hint: Dict[str, Any] = field(default_factory=dict)

def is_desktop_event(ev: Dict[str, Any]) -> bool:
    if looks_like_terminal_window(ev):
        return False
    et = ev.get("event_type")
    if et in ("mouse_click", "mouse_scroll", "typing_burst", "key_press_special", "key_text_input", "hotkey"):
        return not looks_like_browser_window(ev)
    return False

def _app_sig(ev: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[str]]:
    return (str(ev.get("app_name") or ""), ev.get("pid", None), ev.get("bundle_id", None))

def _extract_type_verification_tokens(text: str, max_tokens: int = 2) -> List[str]:
    """Pick a couple of stable tokens to validate typing via OCR later."""
    t = (text or "").strip()
    if not t:
        return []
    parts = re.findall(r"[A-Za-z0-9]{4,}", t)
    out: List[str] = []
    for p in parts:
        if p.lower() in ("http", "https", "www"):
            continue
        if p not in out:
            out.append(p)
        if len(out) >= max_tokens:
            break
    if not out and len(t) >= 6:
        out = [t[:12].strip()]
    return out

def _default_desktop_verify_hint(kind: str, app_name: str, bundle_id: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    bid = bundle_id or ""
    app = app_name or ""

    if kind == "desktop_click":
        return {
            "type": "desktop_click_verified",
            "strategy": "observe_active_app_window",
            "timeout_ms": 3000,
            "checks": [{"op": "active_app_is", "value": bid or app}],
        }

    if kind == "desktop_type_text":
        joined = str((payload or {}).get("joined_text") or "")
        tokens = _extract_type_verification_tokens(joined, max_tokens=2)
        checks = [{"op": "active_app_is", "value": bid or app}]
        if tokens:
            checks.append({"op": "ocr_contains_all", "values": tokens, "region_hint": "center"})
        return {
            "type": "desktop_type_verified",
            "strategy": "observe_text_present",
            "timeout_ms": 5000,
            "checks": checks,
        }

    if kind == "desktop_hotkey":
        hotkeys = ((payload or {}).get("hotkeys") or [])
        combo = str(hotkeys[0].get("combo") if hotkeys else "")
        if combo.lower().replace(" ", "") in ("cmd+s", "command+s"):
            return {
                "type": "desktop_save_initiated",
                "strategy": "observe_ui_change",
                "timeout_ms": 8000,
                "checks": [
                    {"op": "active_app_is", "value": bid or app},
                    {"op": "ocr_contains_any", "values": ["Save", "Cancel", "Where:", "Format:", "Name:"], "region_hint": "any"},
                ],
            }
        return {
            "type": "desktop_hotkey_verified",
            "strategy": "observe_active_app_window",
            "timeout_ms": 3000,
            "checks": [{"op": "active_app_is", "value": bid or app}],
        }

    if kind == "desktop_key":
        return {
            "type": "desktop_key_verified",
            "strategy": "observe_active_app_window",
            "timeout_ms": 3000,
            "checks": [{"op": "active_app_is", "value": bid or app}],
        }

    if kind == "desktop_scroll":
        return {
            "type": "desktop_scroll_verified",
            "strategy": "observe_active_app_window",
            "timeout_ms": 3000,
            "checks": [{"op": "active_app_is", "value": bid or app}],
        }

    return {}

def build_desktop_steps(
    events: List[Dict[str, Any]],
    *,
    merge_within_ms: int = 900,
    text_merge_within_ms: int = 1800,
) -> List[DesktopStep]:
    steps: List[DesktopStep] = []
    evs = [e for e in events if e.get("event_type") != "workflow_control"]
    evs.sort(key=lambda e: safe_int(e.get("time_ms")))

    def _same_app(prev: DesktopStep, ev: Dict[str, Any]) -> bool:
        return (prev.app_name, prev.pid, prev.bundle_id) == _app_sig(ev)

    def _push_step(step: DesktopStep) -> None:
        steps.append(step)
        step.step_id = f"desktop_{len(steps)}"

    def _mk_step(
        *,
        t: int,
        kind: str,
        ev: Dict[str, Any],
        payload: Dict[str, Any],
        evidence: List[str],
        t_end: Optional[int] = None,
        summary: str,
    ) -> DesktopStep:
        s = DesktopStep(
            step_id=f"desktop_{len(steps)+1}",
            t_ms=t,
            t_end_ms=t_end if t_end is not None else t,
            kind=kind,
            app_name=str(ev.get("app_name") or ""),
            window_title=str(ev.get("window_title") or ""),
            pid=ev.get("pid", None),
            bundle_id=ev.get("bundle_id", None),
            evidence_event_ids=evidence,
            payload=payload,
            summary=summary,
            verify_hint={},
        )
        s.verify_hint = _default_desktop_verify_hint(kind, s.app_name, s.bundle_id, payload)
        return s

    # typing aggregation
    typing_active = False
    typing_sig: Optional[Tuple[str, Optional[int], Optional[str]]] = None
    typing_start_t = 0
    typing_end_t = 0
    typing_evidence: List[str] = []
    typing_chunks: List[Dict[str, Any]] = []
    typing_buf: List[str] = []
    typing_first_ev: Optional[Dict[str, Any]] = None
    typing_last_event_was_text_input = False

    def _flush_typing(reason: str):
        nonlocal typing_active, typing_sig, typing_start_t, typing_end_t
        nonlocal typing_evidence, typing_chunks, typing_buf, typing_first_ev, typing_last_event_was_text_input

        if not typing_active:
            return

        joined = "".join(typing_buf)
        if joined:
            ev0 = typing_first_ev or {"app_name": "", "window_title": "", "pid": None, "bundle_id": None}
            _push_step(_mk_step(
                t=typing_start_t,
                kind="desktop_type_text",
                ev=ev0,
                payload={"chunks": typing_chunks, "joined_text": joined, "reason": reason},
                evidence=typing_evidence,
                t_end=typing_end_t,
                summary=f'Type text in {str(ev0.get("app_name") or "")}: "{joined[:60]}"',
            ))

        typing_active = False
        typing_sig = None
        typing_start_t = 0
        typing_end_t = 0
        typing_evidence = []
        typing_chunks = []
        typing_buf = []
        typing_first_ev = None
        typing_last_event_was_text_input = False

    def _ensure_typing(ev: Dict[str, Any], t: int):
        nonlocal typing_active, typing_sig, typing_start_t, typing_end_t, typing_first_ev
        sig = _app_sig(ev)
        if not typing_active:
            typing_active = True
            typing_sig = sig
            typing_start_t = t
            typing_end_t = t
            typing_first_ev = ev
            return
        if typing_sig != sig:
            _flush_typing("app_changed")
            typing_active = True
            typing_sig = sig
            typing_start_t = t
            typing_end_t = t
            typing_first_ev = ev
            return
        if (t - typing_end_t) > text_merge_within_ms:
            _flush_typing("gap")
            typing_active = True
            typing_sig = sig
            typing_start_t = t
            typing_end_t = t
            typing_first_ev = ev

    def _append_text(ev: Dict[str, Any], t: int, text: str, reason: str):
        nonlocal typing_end_t, typing_evidence, typing_chunks, typing_buf, typing_last_event_was_text_input
        _ensure_typing(ev, t)
        typing_end_t = max(typing_end_t, t)
        typing_evidence = list(dict.fromkeys(typing_evidence + [ev_id(ev)]))
        typing_chunks.append({"text": text, "length": len(text), "reason": reason})
        typing_buf.append(text)
        typing_last_event_was_text_input = True

    def _backspace(ev: Dict[str, Any], t: int):
        nonlocal typing_end_t, typing_evidence, typing_buf, typing_chunks, typing_last_event_was_text_input
        _ensure_typing(ev, t)
        typing_end_t = max(typing_end_t, t)
        typing_evidence = list(dict.fromkeys(typing_evidence + [ev_id(ev)]))
        if typing_buf:
            last = typing_buf[-1]
            if last:
                typing_buf[-1] = last[:-1]
            else:
                typing_buf.pop()
        typing_chunks.append({"text": "", "length": 0, "reason": "backspace"})
        typing_last_event_was_text_input = False

    def flush_before_non_typing():
        if typing_active:
            _flush_typing("before_non_typing")

    for ev in evs:
        if not is_desktop_event(ev):
            continue

        et = ev.get("event_type")
        t = safe_int(ev.get("time_ms"))
        evid = [ev_id(ev)]

        if et == "mouse_click" and ev.get("pressed") is False:
            flush_before_non_typing()

            artifact = ev.get("artifact") or {}
            click_xy_px = (artifact.get("click_xy_px") or {})
            raw_xy = (artifact.get("raw_click_xy") or {})

            # IMPORTANT:
            # click_xy_px is expected to be in the same pixel space as the recorder screenshots
            # referenced by evidence.pre/post/window_fallback.bbox_px.
            # The desktop agent will later scale this into pyautogui coordinates at execution time.
            x_px = ev.get("x_px", None)
            y_px = ev.get("y_px", None)
            if x_px is None or y_px is None:
                x_px = click_xy_px.get("x", None)
                y_px = click_xy_px.get("y", None)

            pre = artifact.get("pre") or {}
            post = artifact.get("post") or {}
            fb = artifact.get("fallback_window") or artifact.get("window_fallback") or {}

            payload = {
                "button": ev.get("button"),
                "pressed": ev.get("pressed"),
                "click_id": artifact.get("click_id"),
                "raw_xy": {"x": raw_xy.get("x", ev.get("x")), "y": raw_xy.get("y", ev.get("y"))},
                "click_xy_px": {"x": x_px, "y": y_px},
                "evidence": {
                    "pre": {
                        "screenshot_path": pre.get("screenshot_path"),
                        "bbox_px": pre.get("bbox_px"),
                        "image_size": pre.get("image_size"),
                        "coord_space": pre.get("coord_space"),
                    } if pre else None,
                    "post": {
                        "screenshot_path": post.get("screenshot_path"),
                        "bbox_px": post.get("bbox_px"),
                        "image_size": post.get("image_size"),
                        "coord_space": post.get("coord_space"),
                    } if post else None,
                    "window_fallback": {
                        "screenshot_path": fb.get("screenshot_path"),
                        "bbox_px": fb.get("bbox_px"),
                        "image_size": fb.get("image_size"),
                        "coord_space": fb.get("coord_space"),
                        "window_crop": (fb.get("window_crop") or {}),
                    } if fb else None,
                },
            }

            if steps and steps[-1].kind == "desktop_click" and _same_app(steps[-1], ev) and (t - steps[-1].t_end_ms) <= merge_within_ms:
                steps[-1].t_end_ms = max(steps[-1].t_end_ms, t)
                steps[-1].evidence_event_ids = list(dict.fromkeys(steps[-1].evidence_event_ids + evid))
                steps[-1].payload.setdefault("clicks", [])
                steps[-1].payload["clicks"].append(payload)
                steps[-1].summary = f"Click in {steps[-1].app_name}"
            else:
                _push_step(_mk_step(
                    t=t,
                    kind="desktop_click",
                    ev=ev,
                    payload={"clicks": [payload]},
                    evidence=evid,
                    summary=f"Click in {str(ev.get('app_name') or '')}",
                ))
            continue

        if et == "key_text_input":
            txt = str(ev.get("text") or "")
            if txt:
                _append_text(ev, t, txt, str(ev.get("reason") or "idle"))
            continue

        if et == "key_press_special":
            k = str(ev.get("key") or "")

            if k in ("Space", "Tab", "Enter", "Return"):
                continue
            if k in ("Backspace", "Delete"):
                _backspace(ev, t)
                continue

            flush_before_non_typing()
            payload = {"key": ev.get("key"), "ctrl": ev.get("ctrl"), "alt": ev.get("alt"),
                       "shift": ev.get("shift"), "meta": ev.get("meta")}
            if steps and steps[-1].kind == "desktop_key" and _same_app(steps[-1], ev) and (t - steps[-1].t_end_ms) <= merge_within_ms:
                steps[-1].t_end_ms = max(steps[-1].t_end_ms, t)
                steps[-1].evidence_event_ids = list(dict.fromkeys(steps[-1].evidence_event_ids + evid))
                steps[-1].payload.setdefault("keys", [])
                steps[-1].payload["keys"].append(payload)
                steps[-1].summary = f"Press keys in {steps[-1].app_name}"
            else:
                _push_step(_mk_step(
                    t=t, kind="desktop_key", ev=ev, payload={"keys": [payload]}, evidence=evid,
                    summary=f"Press {str(ev.get('key') or '')} in {str(ev.get('app_name') or '')}"
                ))
            continue

        if et == "hotkey":
            flush_before_non_typing()
            payload = {"combo": ev.get("combo"), "key": ev.get("key"), "ctrl": ev.get("ctrl"),
                       "alt": ev.get("alt"), "shift": ev.get("shift"), "meta": ev.get("meta")}
            if steps and steps[-1].kind == "desktop_hotkey" and _same_app(steps[-1], ev) and (t - steps[-1].t_end_ms) <= merge_within_ms:
                steps[-1].t_end_ms = max(steps[-1].t_end_ms, t)
                steps[-1].evidence_event_ids = list(dict.fromkeys(steps[-1].evidence_event_ids + evid))
                steps[-1].payload.setdefault("hotkeys", [])
                steps[-1].payload["hotkeys"].append(payload)
                steps[-1].summary = f"Hotkeys in {steps[-1].app_name}"
                steps[-1].verify_hint = _default_desktop_verify_hint("desktop_hotkey", steps[-1].app_name, steps[-1].bundle_id, steps[-1].payload)
            else:
                _push_step(_mk_step(
                    t=t, kind="desktop_hotkey", ev=ev, payload={"hotkeys": [payload]}, evidence=evid,
                    summary=f'Hotkey {str(ev.get("combo") or "")} in {str(ev.get("app_name") or "")}'
                ))
            continue

        if et == "mouse_scroll":
            flush_before_non_typing()
            payload = {"x": ev.get("x"), "y": ev.get("y"), "dx": ev.get("dx"), "dy": ev.get("dy")}
            if steps and steps[-1].kind == "desktop_scroll" and _same_app(steps[-1], ev) and (t - steps[-1].t_end_ms) <= merge_within_ms:
                steps[-1].t_end_ms = max(steps[-1].t_end_ms, t)
                steps[-1].evidence_event_ids = list(dict.fromkeys(steps[-1].evidence_event_ids + evid))
                steps[-1].payload.setdefault("scrolls", [])
                steps[-1].payload["scrolls"].append(payload)
                steps[-1].summary = f"Scroll in {steps[-1].app_name}"
            else:
                _push_step(_mk_step(
                    t=t, kind="desktop_scroll", ev=ev, payload={"scrolls": [payload]}, evidence=evid,
                    summary=f"Scroll in {str(ev.get('app_name') or '')}"
                ))
            continue

        if et == "typing_burst":
            continue

    _flush_typing("eof")

    for i, s in enumerate(steps, start=1):
        s.step_id = f"desktop_{i}"

    return steps

# -----------------------------
# OCR enrichment for desktop_click
# -----------------------------
def enrich_desktop_click_steps_with_ocr(
    desktop_steps: List[DesktopStep],
    *,
    session_dir: Path,
    event_by_id: Dict[str, Dict[str, Any]],
    ocr_base_url: Optional[str],
    ocr_timeout_s: int = 120,
    min_conf: float = 0.55,
    near_radius_px: int = 60,
    skip_chrome_margin_top_px: int = 70,
    skip_chrome_margin_left_px: int = 90,
) -> List[DesktopStep]:
    if not ocr_base_url:
        return desktop_steps

    observe_cache: Dict[str, Dict[str, Any]] = {}

    def _load_image_size(img_path: Path) -> Tuple[int, int]:
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                return int(im.size[0]), int(im.size[1])
        except Exception:
            return 0, 0

    def _get_observe(img_path: Path) -> Dict[str, Any]:
        key = str(img_path)
        if key in observe_cache:
            return observe_cache[key]
        obs = call_observe(ocr_base_url, img_path, timeout=ocr_timeout_s)
        observe_cache[key] = obs
        return obs

    def _coord_space_note(coord_space: Any) -> Optional[str]:
        # We expect screenshot-space px here. If recorder says otherwise, capture it for debugging.
        cs = str(coord_space or "").strip().lower()
        if not cs:
            return None
        if "screenshot" in cs or "pixel" in cs or "px" in cs:
            return None
        if "pyautogui" in cs or "points" in cs or "logical" in cs:
            return f"coord_space={coord_space} (expected screenshot_px)"
        return f"coord_space={coord_space}"

    def _attempt_grounding(
        *,
        label: str,
        rel_path: str,
        crop_bbox_px: List[int],
        click_xy_px: Dict[str, Any],
        coord_space: Any,
        image_size_meta: Any,
    ) -> Dict[str, Any]:
        """
        Convert a global click (screenshot px) into crop-local px:
          local = click_xy_px - crop_bbox_px[:2]
        Then OCR-hit-test inside the crop image.
        """
        if not rel_path or not crop_bbox_px or len(crop_bbox_px) != 4:
            return {"ok": False, "source": label, "error": "missing_crop_evidence"}

        img_path = (session_dir / rel_path).resolve()
        if not img_path.exists():
            return {"ok": False, "source": label, "error": "image_not_found", "screenshot_path": rel_path}

        x_px = safe_int(click_xy_px.get("x"), -1)
        y_px = safe_int(click_xy_px.get("y"), -1)
        if x_px < 0 or y_px < 0:
            return {"ok": False, "source": label, "error": "missing_click_xy_px", "screenshot_path": rel_path}

        left, top, right, bottom = [safe_int(v) for v in crop_bbox_px]
        local_x = x_px - left
        local_y = y_px - top

        w, h = _load_image_size(img_path)
        if w <= 0 or h <= 0:
            return {"ok": False, "source": label, "error": "missing_image_size", "screenshot_path": rel_path}

        if local_x < 0 or local_y < 0 or local_x >= w or local_y >= h:
            return {
                "ok": False,
                "source": label,
                "error": "click_outside_crop",
                "screenshot_path": rel_path,
                "click_local_px": [int(local_x), int(local_y)],
                "image_size": [int(w), int(h)],
                "crop_bbox_px": crop_bbox_px,
                "click_xy_px": click_xy_px,
                "coord_space_note": _coord_space_note(coord_space),
                "image_size_meta": image_size_meta,
            }

        # Skip typical window controls/titlebar corner.
        if int(local_y) <= skip_chrome_margin_top_px and int(local_x) <= skip_chrome_margin_left_px:
            return {
                "ok": False,
                "source": label,
                "screenshot_path": rel_path,
                "click_local_px": [int(local_x), int(local_y)],
                "ocr": {"error": "skipped_window_chrome"},
                "coord_space_note": _coord_space_note(coord_space),
            }

        obs = _get_observe(img_path)
        match = resolve_click_to_ocr_element(
            obs,
            click_x=int(local_x),
            click_y=int(local_y),
            img_w=int(w),
            img_h=int(h),
            min_conf=min_conf,
            near_radius_px=near_radius_px,
        )

        if not match.get("ok"):
            return {
                "ok": False,
                "source": label,
                "screenshot_path": rel_path,
                "click_local_px": [int(local_x), int(local_y)],
                "ocr": {"error": match.get("error")},
                "coord_space_note": _coord_space_note(coord_space),
            }

        return {
            "ok": True,
            "source": label,
            "screenshot_path": rel_path,
            "click_local_px": [int(local_x), int(local_y)],
            "coord_space_note": _coord_space_note(coord_space),
            "ocr": {
                "method": match.get("method"),
                "text": match.get("text"),
                "confidence": match.get("confidence"),
                "bbox_px": match.get("bbox_px"),
                "element_id": match.get("element_id"),
                "distance_px": match.get("distance_px", None),
            },
        }

    for s in desktop_steps:
        if s.kind != "desktop_click":
            continue

        clicks = (s.payload.get("clicks") or [])
        grounded_clicks = []

        for idx, c in enumerate(clicks):
            ev_id_guess = s.evidence_event_ids[idx] if idx < len(s.evidence_event_ids) else (s.evidence_event_ids[-1] if s.evidence_event_ids else None)

            click_xy_px = (c.get("click_xy_px") or {})
            evidence = (c.get("evidence") or {})
            pre = evidence.get("pre") or {}
            post = evidence.get("post") or {}
            wf = evidence.get("window_fallback") or {}

            attempts: List[Dict[str, Any]] = []

            # Prefer PRE first (often contains labels right before click causes UI change).
            if pre.get("screenshot_path") and pre.get("bbox_px"):
                attempts.append(_attempt_grounding(
                    label="pre",
                    rel_path=pre.get("screenshot_path"),
                    crop_bbox_px=pre.get("bbox_px"),
                    click_xy_px=click_xy_px,
                    coord_space=pre.get("coord_space"),
                    image_size_meta=pre.get("image_size"),
                ))
            if post.get("screenshot_path") and post.get("bbox_px"):
                attempts.append(_attempt_grounding(
                    label="post",
                    rel_path=post.get("screenshot_path"),
                    crop_bbox_px=post.get("bbox_px"),
                    click_xy_px=click_xy_px,
                    coord_space=post.get("coord_space"),
                    image_size_meta=post.get("image_size"),
                ))
            if wf.get("screenshot_path") and wf.get("bbox_px"):
                attempts.append(_attempt_grounding(
                    label="window_fallback",
                    rel_path=wf.get("screenshot_path"),
                    crop_bbox_px=wf.get("bbox_px"),
                    click_xy_px=click_xy_px,
                    coord_space=wf.get("coord_space"),
                    image_size_meta=wf.get("image_size"),
                ))

            best: Optional[Dict[str, Any]] = None
            for a in attempts:
                if a.get("ok"):
                    best = a
                    break
            if best is None:
                best = attempts[0] if attempts else {"ok": False, "error": "no_evidence_images"}

            grounded_clicks.append({
                "ok": bool(best.get("ok")),
                "source_event_id": ev_id_guess,
                "click": {
                    "click_id": c.get("click_id"),
                    "raw_xy": c.get("raw_xy"),
                    "click_xy_px": click_xy_px,
                    "button": c.get("button"),
                },
                "attempts": attempts,
                "best": best,
            })

        s.payload["grounding"] = {
            "type": "ocr_click_grounding_v3",
            "base_url": ocr_base_url,
            "min_conf": min_conf,
            "near_radius_px": near_radius_px,
            "prefer": "pre_then_post_then_window_fallback",
            "skip_window_chrome": {"top_px": skip_chrome_margin_top_px, "left_px": skip_chrome_margin_left_px},
            "clicks": grounded_clicks,
        }

        text_hint = ""
        element_id = None
        bbox_px = None
        source_image = None
        conf = None

        for g in grounded_clicks:
            b = (g.get("best") or {})
            o = (b.get("ocr") or {})
            if g.get("ok") and (o.get("text") or "").strip():
                text_hint = str(o.get("text") or "").strip()
                element_id = o.get("element_id")
                bbox_px = o.get("bbox_px")
                source_image = b.get("screenshot_path")
                conf = o.get("confidence")
                break

        s.payload["target"] = {
            "intent": "click_text_hint" if text_hint else "click_recorded_point",
            "text_hint": text_hint or None,
            "element_id": element_id,
            "bbox_px": bbox_px,
            "ocr_confidence": conf,
            "source_image": source_image,
            "fallback": {
                "use_recorded_click_xy_px": True,
                "recorded_click_xy_px": (clicks[0].get("click_xy_px") if clicks else None),
            },
        }

        if text_hint:
            s.summary = f'Click "{text_hint}" in {s.app_name}'
            s.verify_hint = {
                "type": "desktop_click_verified",
                "strategy": "observe_active_app_and_optional_text_change",
                "timeout_ms": 5000,
                "checks": [
                    {"op": "active_app_is", "value": s.bundle_id or s.app_name},
                    {"op": "optional_ocr_contains", "value": text_hint},
                ],
            }
        else:
            s.verify_hint = _default_desktop_verify_hint("desktop_click", s.app_name, s.bundle_id, s.payload)

    return desktop_steps

# -----------------------------
# Optional: prepend focus/fullscreen per app
# -----------------------------
def maybe_prepend_desktop_focus_fullscreen(desktop_steps: List[DesktopStep]) -> List[DesktopStep]:
    out: List[DesktopStep] = []
    seen: set[str] = set()

    def _mk_synth(step_id: str, t: int, kind: str, app_name: str, bundle_id: Optional[str], summary: str, params: Dict[str, Any], verify_hint: Dict[str, Any]) -> DesktopStep:
        return DesktopStep(
            step_id=step_id,
            t_ms=t,
            t_end_ms=t,
            kind=kind,
            app_name=app_name,
            window_title=app_name,
            pid=None,
            bundle_id=bundle_id,
            evidence_event_ids=[],
            payload={"params": params},
            summary=summary,
            verify_hint=verify_hint,
        )

    for s in desktop_steps:
        sig = (s.bundle_id or s.app_name or "").strip()
        if sig and sig not in seen:
            seen.add(sig)
            t = int(s.t_ms) - 50
            out.append(_mk_synth(
                step_id=f"desktop_focus_{len(out)+1}",
                t=t,
                kind="desktop_focus_app",
                app_name=s.app_name,
                bundle_id=s.bundle_id,
                summary=f"Focus app {s.app_name}",
                params={"bundle_id": s.bundle_id, "app_name": s.app_name},
                verify_hint={
                    "type": "desktop_focus_verified",
                    "strategy": "observe_active_app_window",
                    "timeout_ms": 8000,
                    "checks": [{"op": "active_app_is", "value": s.bundle_id or s.app_name}],
                },
            ))
            out.append(_mk_synth(
                step_id=f"desktop_fullscreen_{len(out)+1}",
                t=t + 1,
                kind="desktop_fullscreen_window",
                app_name=s.app_name,
                bundle_id=s.bundle_id,
                summary=f"Fullscreen {s.app_name}",
                params={"bundle_id": s.bundle_id, "app_name": s.app_name},
                verify_hint={
                    "type": "desktop_fullscreen_verified",
                    "strategy": "observe_window_state",
                    "timeout_ms": 8000,
                    "checks": [{"op": "window_state_is", "value": "fullscreen"}],
                },
            ))

        out.append(s)

    return out

# -----------------------------
# Interleave + browser launch
# -----------------------------
def interleave_steps(
    browser_ui_steps: List[BrowserUIStep],
    browser_nav_steps: List[BrowserNavStep],
    desktop_steps: List[DesktopStep],
) -> List[Dict[str, Any]]:
    items: List[Tuple[int, Dict[str, Any]]] = []

    for s in browser_ui_steps:
        items.append((int(s.t_ms), {"agent": "browser", "type": s.kind, **asdict(s)}))

    for s in browser_nav_steps:
        items.append((int(s.t_anchor_ms), {"agent": "browser", "type": "navigate", **asdict(s)}))

    for s in desktop_steps:
        items.append((int(s.t_ms), {"agent": "desktop", "type": s.kind, **asdict(s)}))

    items.sort(key=lambda x: x[0])
    return [x[1] for x in items]

def maybe_prepend_browser_launch(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    has_browser = any(s.get("agent") == "browser" for s in steps)
    if not has_browser:
        return steps

    if steps and steps[0].get("agent") == "browser" and steps[0].get("type") in ("browser_launch", "launch"):
        return steps

    launch_step = {
        "agent": "browser",
        "type": "browser_launch",
        "step_id": "browser_launch_1",
        "params": {
            "browser_type": "chromium",
            "headless": False,
            "channel": "chrome",
            "new_context": True,
        },
        "verify_hint": {
            "type": "browser_ready",
            "strategy": "playwright_context_ready",
            "timeout_ms": 15000,
            "checks": [{"op": "browser_connected"}, {"op": "context_exists"}, {"op": "page_exists"}],
        },
        "summary": "Launch Playwright Chromium (Chrome channel)",
        "evidence_event_ids": [],
    }
    return [launch_step] + steps

def write_outputs(session_dir: Path, steps: List[Dict[str, Any]]) -> None:
    out_path = session_dir / "workflow_steps.v1.json"
    browser_count = sum(1 for s in steps if s.get("agent") == "browser")
    desktop_count = sum(1 for s in steps if s.get("agent") == "desktop")

    out_path.write_text(
        json.dumps(
            {
                "schema": "workflow_steps.v1",
                "total_steps": len(steps),
                "browser_steps": browser_count,
                "desktop_steps": desktop_count,
                "steps": steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ Wrote {out_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", default="recordings/20260107_190820")
    ap.add_argument("--events", default="events.jsonl")

    ap.add_argument("--web-pre-ms", type=int, default=200)
    ap.add_argument("--web-post-ms", type=int, default=8000)
    ap.add_argument("--web-min-gap-ms", type=int, default=600)
    ap.add_argument("--web-merge-same-url-within-ms", type=int, default=1500)

    ap.add_argument("--web-ui-dedupe-within-ms", type=int, default=450)
    ap.add_argument("--web-ui-merge-type-within-ms", type=int, default=1200)
    ap.add_argument("--web-ui-emit-focus", action="store_true")

    ap.add_argument("--desktop-merge-within-ms", type=int, default=900)
    ap.add_argument("--desktop-text-merge-within-ms", type=int, default=1800)

    ap.add_argument("--desktop-ocr-url", default="https://vqlzy77m77329p-8000.proxy.runpod.net")
    ap.add_argument("--desktop-ocr-timeout", type=int, default=120)
    ap.add_argument("--desktop-ocr-min-conf", type=float, default=0.55)
    ap.add_argument("--desktop-ocr-near-radius", type=int, default=60)

    ap.add_argument("--desktop-emit-focus-fullscreen", action="store_true",
                    help="Insert desktop_focus_app + desktop_fullscreen_window before first action per app")

    ap.add_argument("--desktop-ocr-skip-top", type=int, default=70)
    ap.add_argument("--desktop-ocr-skip-left", type=int, default=90)

    args = ap.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    events_path = session_dir / args.events
    if not events_path.exists():
        raise SystemExit(f"Missing events file: {events_path}")

    events = load_events(events_path)
    events = [e for e in events if e.get("event_type") != "workflow_control"]
    event_by_id: Dict[str, Dict[str, Any]] = {ev_id(e): e for e in events if ev_id(e)}

    browser_ui_steps = build_browser_ui_steps(
        events,
        dedupe_within_ms=args.web_ui_dedupe_within_ms,
        merge_type_within_ms=args.web_ui_merge_type_within_ms,
        emit_focus_steps=bool(args.web_ui_emit_focus),
    )

    anchors = collect_web_anchors(events, browser_ui_steps=browser_ui_steps)
    browser_nav_steps = build_browser_nav_steps(
        events,
        anchors,
        ui_steps=browser_ui_steps,
        pre_ms=args.web_pre_ms,
        post_ms=args.web_post_ms,
        min_gap_ms=args.web_min_gap_ms,
        merge_same_url_within_ms=args.web_merge_same_url_within_ms,
    )

    desktop_steps = build_desktop_steps(
        events,
        merge_within_ms=args.desktop_merge_within_ms,
        text_merge_within_ms=args.desktop_text_merge_within_ms,
    )

    ocr_url = (args.desktop_ocr_url or "").strip()
    if ocr_url:
        desktop_steps = enrich_desktop_click_steps_with_ocr(
            desktop_steps,
            session_dir=session_dir,
            event_by_id=event_by_id,
            ocr_base_url=ocr_url,
            ocr_timeout_s=args.desktop_ocr_timeout,
            min_conf=args.desktop_ocr_min_conf,
            near_radius_px=args.desktop_ocr_near_radius,
            skip_chrome_margin_top_px=args.desktop_ocr_skip_top,
            skip_chrome_margin_left_px=args.desktop_ocr_skip_left,
        )

    if args.desktop_emit_focus_fullscreen:
        desktop_steps = maybe_prepend_desktop_focus_fullscreen(desktop_steps)

    combined = interleave_steps(browser_ui_steps, browser_nav_steps, desktop_steps)
    combined = maybe_prepend_browser_launch(combined)
    write_outputs(session_dir, combined)

if __name__ == "__main__":
    main()
