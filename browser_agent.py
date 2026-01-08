#!/usr/bin/env python3
"""
primitive_browser_agent.py

Generic Browser Executor (DOM-grounded) that runs *primitive* browser workflows.

What this agent does:
- Launches Playwright (optionally with channel="chrome").
- Resolves targets primarily via selector_candidates, with safe fallbacks.
- Executes primitives: browser_launch, navigate, browser_click/submit, browser_type, browser_select.
- Verifies post-conditions via a small set of generic verify ops.
- Self-heals on page/browser crashes by relaunching.

Reliability updates:
- Click steps are idempotent: if we're no longer on the page where the click was recorded,
  we treat the click as already applied (useful for derived click→navigate sequences).
- css_path candidates are tried in multiple selector syntaxes.
- If structural selectors fail, falls back to link/text matching.
- On google.com/search pages, waits for #rso before resolving click targets.
- Google Forms submit verification is OR-based (URL change OR confirmation text).

Logging policy:
- INFO: one line per primitive
- DEBUG: resolution/diagnostics
- ERROR: only on failure
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import platform
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeoutError,
    Error as PWError,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
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
    level = _parse_log_level(os.getenv("BROWSER_LOG_LEVEL", "INFO"), default=logging.INFO)
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


logger = setup_logger("PrimitiveBrowserAgent")

# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
ERROR_TIMEOUT = "timeout"
ERROR_NOT_FOUND = "not_found"
ERROR_NAVIGATION = "navigation_error"
ERROR_CLICK = "click_error"
ERROR_TYPE = "type_error"
ERROR_SELECT = "select_error"
ERROR_ASSERT = "assertion_failed"
ERROR_UNKNOWN = "unknown_error"


@dataclass
class Backoff:
    max_retries: int = 2
    base_delay_s: float = 0.25
    max_delay_s: float = 1.5

    def sleep(self, attempt: int) -> None:
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        jitter = (attempt % 3) * 0.03
        time.sleep(delay + jitter)


# -----------------------------------------------------------------------------
# Primitive Browser Agent
# -----------------------------------------------------------------------------
class PrimitiveBrowserAgent:
    """
    Generic DOM-grounded executor using Playwright.

    macOS note:
      - On macOS Catalina (10.15) and older, Playwright-bundled Chromium may crash/disconnect.
        Using system Chrome via channel="chrome" is the simplest workaround.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        viewport: Tuple[int, int] = (1280, 800),
        artifacts_dir: str = "artifacts_browser",
        default_timeout_ms: int = 15_000,
        implicit_wait_ms: int = 6_000,
        backoff: Backoff = Backoff(),
        stealth_mode: bool = True,
        dom_observe_max_elems: int = 200,
        networkidle_after_goto: bool = True,
        networkidle_timeout_ms: int = 10_000,
        default_channel: Optional[str] = None,
        verbose_console: bool = False,
        google_serp_wait_rso: bool = True,
        click_idempotent_skip: bool = True,
        submit_wait_ms: int = 8000,  # NEW: wait longer for submit confirmation
    ):
        self.headless = headless
        self.viewport = viewport
        self.artifacts_dir = artifacts_dir
        self.default_timeout_ms = default_timeout_ms
        self.implicit_wait_ms = implicit_wait_ms
        self.backoff = backoff
        self.stealth_mode = stealth_mode
        self.dom_observe_max_elems = int(dom_observe_max_elems)
        self.networkidle_after_goto = bool(networkidle_after_goto)
        self.networkidle_timeout_ms = int(networkidle_timeout_ms)
        self.default_channel = default_channel
        self.verbose_console = bool(verbose_console) or (
            os.getenv("BROWSER_VERBOSE_CONSOLE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        )
        self.google_serp_wait_rso = bool(google_serp_wait_rso)
        self.click_idempotent_skip = bool(click_idempotent_skip)
        self.submit_wait_ms = int(submit_wait_ms)

        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._closed = True

        self._id_to_selector: Dict[str, str] = {}
        self._ref_to_selector: Dict[str, str] = {}
        self._fp_to_selector: Dict[str, str] = {}

        os.makedirs(self.artifacts_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # URL helpers
    # -------------------------------------------------------------------------
    def _netloc(self, url: str) -> str:
        try:
            return (urlsplit(url).netloc or "").lower()
        except Exception:
            return ""

    def _is_blank_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        return u in ("", "about:blank")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def _is_alive(self) -> bool:
        try:
            if self._closed:
                return False
            if not self._pw or not self._browser or not self._context or not self._page:
                return False
            if self._page.is_closed():
                return False
            return True
        except Exception:
            return False

    def _hard_reset(self) -> None:
        try:
            try:
                if self._context:
                    self._context.close()
            except Exception:
                pass
            try:
                if self._browser:
                    self._browser.close()
            except Exception:
                pass
            try:
                if self._pw:
                    self._pw.stop()
            except Exception:
                pass
        finally:
            self._pw = None
            self._browser = None
            self._context = None
            self._page = None
            self._closed = True

    def _ensure_alive(self) -> None:
        if self._is_alive():
            return
        logger.info("[lifecycle] browser not alive → relaunch")
        self._hard_reset()
        self.launch()

    def launch(self) -> Dict[str, Any]:
        params = {
            "browser_type": "chromium",
            "headless": self.headless,
            "channel": self.default_channel,
            "new_context": True,
        }
        return self.browser_launch(params=params, verify_hint=None)

    def _is_catalina_or_older(self) -> bool:
        if platform.system().lower() != "darwin":
            return False
        mac_ver = platform.mac_ver()[0] or ""
        try:
            parts = [int(x) for x in mac_ver.split(".") if x.strip().isdigit()]
            if len(parts) >= 2:
                major, minor = parts[0], parts[1]
                return (major == 10 and minor <= 15)
        except Exception:
            pass
        return False

    def browser_launch(self, params: Dict[str, Any], verify_hint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            browser_type = (params.get("browser_type") or "chromium").lower()
            headless = bool(params.get("headless", True))
            new_context = bool(params.get("new_context", True))

            channel = params.get("channel") or self.default_channel
            if self._is_catalina_or_older() and not channel:
                channel = "chrome"
                logger.info("[browser_launch] macOS<=10.15 detected → channel=chrome")

            self._hard_reset()
            self.headless = headless

            logger.info(
                "[browser_launch] type=%s headless=%s channel=%s",
                browser_type,
                headless,
                channel or "playwright-managed",
            )

            self._pw = sync_playwright().start()

            is_macos = platform.system().lower() == "darwin"
            launch_args: List[str] = []
            if self.stealth_mode:
                launch_args += ["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"]
                if not is_macos:
                    launch_args += ["--no-sandbox", "--disable-setuid-sandbox"]

            browser_launcher = getattr(self._pw, browser_type, None) or self._pw.chromium

            executable_path = "(unknown)"
            try:
                ep = getattr(browser_launcher, "executable_path", None)
                executable_path = ep() if callable(ep) else (ep or "(unknown)")
            except Exception:
                pass

            launch_kwargs: Dict[str, Any] = {"headless": headless, "args": launch_args}
            if channel:
                launch_kwargs["channel"] = channel

            self._browser = browser_launcher.launch(**launch_kwargs)

            if new_context or not self._context:
                context_options: Dict[str, Any] = {
                    "viewport": {"width": self.viewport[0], "height": self.viewport[1]},
                }
                if self.stealth_mode:
                    context_options["user_agent"] = (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                    context_options["locale"] = "en-US"
                    context_options["timezone_id"] = "America/Los_Angeles"

                self._context = self._browser.new_context(**context_options)

                if self.stealth_mode:
                    self._context.add_init_script(
                        "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
                    )

            self._page = self._context.new_page()
            self._page.set_default_timeout(self.default_timeout_ms)
            self._page.set_default_navigation_timeout(self.default_timeout_ms)
            self._closed = False

            if self.verbose_console:
                try:
                    self._page.on("console", lambda msg: logger.debug("console[%s]: %s", msg.type, msg.text))
                except Exception:
                    pass

            if verify_hint:
                v = self._verify(verify_hint, step_target=None, before_url="")
                if not v.get("ok"):
                    return v

            return self._ok(
                "browser_launch ok",
                {
                    "executable_path": executable_path,
                    "args": launch_args,
                    "params": {**params, "channel_effective": channel},
                },
                t0,
            )
        except Exception as e:
            msg = str(e)
            if "Executable doesn't exist" in msg or "executable doesn't exist" in msg:
                return self._fail(
                    ERROR_UNKNOWN,
                    "browser_launch failed: Playwright browsers not installed. Run: python -m playwright install",
                    {"raw_error": msg},
                    t0,
                )
            logger.error("[browser_launch] failed: %s", msg)
            return self._fail(ERROR_UNKNOWN, f"browser_launch failed: {e}", {"raw_error": msg}, t0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("[lifecycle] close")
        self._hard_reset()

    @property
    def page(self):
        if not self._page:
            raise RuntimeError("Browser not launched. Call browser_launch() first.")
        return self._page

    # -------------------------------------------------------------------------
    # Primitive runner
    # -------------------------------------------------------------------------
    def run_primitive(self, step: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if ctx is None:
            ctx = {}
        t0 = time.time()

        step_id = step.get("step_id") or step.get("id") or "unknown"
        stype = (step.get("type") or step.get("kind") or "").strip()
        max_retries = int(step.get("retries", self.backoff.max_retries))

        logger.info("[execute] step=%s type=%s", step_id, stype)

        def _do() -> Dict[str, Any]:
            if stype == "browser_launch":
                return self.browser_launch(step.get("params") or {}, step.get("verify_hint"))

            self._ensure_alive()

            if stype == "navigate":
                url = step.get("final_url_canon") or step.get("final_url") or ""
                return self._do_navigate(url, step.get("verify_hint"))

            if stype in ("browser_click", "browser_submit"):
                expected_page_url = str(step.get("url") or "")
                target_origin = str((step.get("target") or {}).get("origin") or "")
                is_submit = (stype == "browser_submit")
                return self._do_click(
                    step.get("target") or {},
                    step.get("verify_hint"),
                    expected_page_url=expected_page_url,
                    target_origin=target_origin,
                    is_submit=is_submit,
                )

            if stype == "browser_type":
                v = step.get("value") or {}
                text = v.get("text", v.get("value", step.get("text", "")))
                return self._do_type(step.get("target") or {}, "" if text is None else str(text), step.get("verify_hint"))

            if stype == "browser_select":
                return self._do_select(step.get("target") or {}, step.get("value") or {}, step.get("verify_hint"))

            return self._fail(ERROR_UNKNOWN, f"unknown primitive type: {stype}", {"step": step}, t0)

        last = None
        for attempt in range(max_retries + 1):
            res = _do()
            if res.get("ok"):
                res["step_id"] = step_id
                res["executor"] = "browser"
                res["type"] = stype
                res["attempt"] = attempt + 1
                return res

            last = res
            retriable = res.get("error_type") in {
                ERROR_TIMEOUT,
                ERROR_NAVIGATION,
                ERROR_CLICK,
                ERROR_TYPE,
                ERROR_SELECT,
                ERROR_UNKNOWN,
                ERROR_ASSERT,
                ERROR_NOT_FOUND,
            }
            if retriable and attempt < max_retries:
                logger.info("[retry] step=%s attempt=%d/%d", step_id, attempt + 2, max_retries + 1)
                self.backoff.sleep(attempt)
                continue
            break

        last = last or self._fail(ERROR_UNKNOWN, "unknown failure", {}, t0)
        last["step_id"] = step_id
        last["executor"] = "browser"
        last["type"] = stype
        last["attempt"] = max_retries + 1
        return last

    # -------------------------------------------------------------------------
    # Primitive implementations
    # -------------------------------------------------------------------------
    def _do_navigate(self, url: str, verify_hint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        if not url:
            return self._fail(ERROR_NAVIGATION, "navigate missing url", {}, t0)

        # Google Forms /formResponse is a submit endpoint (POST), not a browsable page (GET).
        if "/forms/" in url and url.rstrip("/").endswith("/formResponse"):
            logger.info("[navigate] skip formResponse GET: %s", url)
            return self._ok(
                "navigate skipped: formResponse is submit endpoint (use browser_submit click instead)",
                {"skipped_url": url, "reason": "formResponse_submit_endpoint"},
                t0,
            )

        try:
            before = self.page.url
            logger.info("[execute] navigate url=%s", url)

            self.page.goto(url, wait_until="domcontentloaded")
            if self.networkidle_after_goto:
                try:
                    self.page.wait_for_load_state("networkidle", timeout=self.networkidle_timeout_ms)
                except Exception:
                    pass
            self.page.wait_for_timeout(250)

            if verify_hint:
                v = self._verify(verify_hint, step_target=None, before_url=before)
                if not v["ok"]:
                    return v

            return self._ok("navigate ok", {"before_url": before, "url": self.page.url}, t0)

        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"navigate timeout: {e}", {}, t0)
        except PWError as e:
            msg = str(e)
            if "has been closed" in msg:
                self._hard_reset()
                self.launch()
                return self._fail(ERROR_NAVIGATION, f"navigate error (browser closed): {msg}", {}, t0)
            return self._fail(ERROR_NAVIGATION, f"navigate error: {e}", {}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"navigate error: {e}", {}, t0)

    def _do_click(
        self,
        target: Dict[str, Any],
        verify_hint: Optional[Dict[str, Any]],
        *,
        expected_page_url: str = "",
        target_origin: str = "",
        is_submit: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()
        try:
            cur_url = self.page.url or ""

            # Idempotence for derived click→navigate sequences (domain changed already).
            if self.click_idempotent_skip and expected_page_url and not self._is_blank_url(cur_url) and not is_submit:
                exp_netloc = self._netloc(expected_page_url)
                cur_netloc = self._netloc(cur_url)
                if exp_netloc and cur_netloc and exp_netloc != cur_netloc:
                    return self._ok(
                        "click skipped: already navigated away from expected page",
                        {
                            "skipped": True,
                            "reason": "already_navigated",
                            "expected_page_url": expected_page_url,
                            "current_url": cur_url,
                            "expected_netloc": exp_netloc,
                            "current_netloc": cur_netloc,
                        },
                        t0,
                    )

            # Stabilize Google SERP before resolving fragile selectors.
            try:
                if self.google_serp_wait_rso and ("google.com/search" in (cur_url or "")):
                    self.page.wait_for_selector("#rso", timeout=5000)
            except Exception:
                pass

            loc, meta = self._resolve_target(target)
            if not loc:
                return self._fail(ERROR_NOT_FOUND, "click target not found", {"target": target, "resolve": meta}, t0)

            before = self.page.url
            logger.info("[execute] click strategy=%s matches=%s", meta.get("strategy"), meta.get("matches"))
            logger.debug("selector=%s", meta.get("selector"))

            loc.click(timeout=self.implicit_wait_ms)

            # Submits often need longer settle time (may not change URL).
            if is_submit:
                self._wait_for_submit_settle(before_url=before, timeout_ms=self.submit_wait_ms)
            else:
                self.page.wait_for_timeout(250)

            self._maybe_cache_identity(target, meta)

            if verify_hint:
                v = self._verify(verify_hint, step_target=target, before_url=before)
                if not v["ok"]:
                    v["extracted"] = {**(v.get("extracted") or {}), "resolve": meta}
                    return v

            return self._ok(
                "click ok" if not is_submit else "submit click ok",
                {"resolve": meta},
                t0,
                matched_locator=meta.get("selector", ""),
                matches=meta.get("matches", 0),
            )

        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"click timeout: {e}", {"target": target}, t0)
        except PWError as e:
            msg = str(e)
            if "has been closed" in msg:
                self._hard_reset()
                self.launch()
                return self._fail(ERROR_CLICK, f"click error (browser closed): {msg}", {"target": target}, t0)
            return self._fail(ERROR_CLICK, f"click error: {e}", {"target": target}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"click error: {e}", {"target": target}, t0)

    def _wait_for_submit_settle(self, *, before_url: str, timeout_ms: int) -> None:
        """Best-effort wait for Google Forms submit confirmation or a URL change."""
        deadline = time.time() + (timeout_ms / 1000.0)

        # First, try networkidle (often helps, but not always).
        try:
            self.page.wait_for_load_state("networkidle", timeout=min(5000, timeout_ms))
        except Exception:
            pass

        confirm_phrases = [
            "Your response has been recorded",
            "Submit another response",
            "Edit your response",
            "Thank you",
            "Thanks",
        ]

        while time.time() <= deadline:
            # URL change is enough.
            try:
                if self.page.url and self.page.url != before_url:
                    return
            except Exception:
                pass

            # Confirmation text on same URL is also enough.
            if self._page_contains_any(confirm_phrases):
                return

            time.sleep(0.25)

        # no-op if we time out; verify step will decide.

    def _page_contains_any(self, phrases: List[str]) -> bool:
        """Fast, best-effort text presence check."""
        for p in phrases:
            p = (p or "").strip()
            if not p:
                continue
            try:
                if self.page.get_by_text(p, exact=False).count() > 0:
                    return True
            except Exception:
                continue
        return False

    def _do_type(self, target: Dict[str, Any], text: str, verify_hint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            loc, meta = self._resolve_target(target, want_fillable=True)
            if not loc:
                return self._fail(ERROR_NOT_FOUND, "type target not found", {"target": target, "resolve": meta}, t0)

            before = self.page.url
            logger.info("[execute] type chars=%d strategy=%s", len(text), meta.get("strategy"))
            logger.debug("preview=%r selector=%s", text[:80], meta.get("selector"))

            loc.click(timeout=self.implicit_wait_ms)
            try:
                loc.fill(text, timeout=self.implicit_wait_ms)
            except Exception:
                loc.press("ControlOrMeta+A")
                loc.type(text, delay=5)

            self.page.wait_for_timeout(250)
            self._maybe_cache_identity(target, meta)

            if verify_hint:
                v = self._verify(verify_hint, step_target=target, before_url=before, typed_text=text)
                if not v["ok"]:
                    v["extracted"] = {**(v.get("extracted") or {}), "resolve": meta}
                    return v

            return self._ok(
                "type ok",
                {"resolve": meta, "typed": text[:200]},
                t0,
                matched_locator=meta.get("selector", ""),
                matches=meta.get("matches", 0),
            )

        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"type timeout: {e}", {"target": target}, t0)
        except PWError as e:
            msg = str(e)
            if "has been closed" in msg:
                self._hard_reset()
                self.launch()
                return self._fail(ERROR_TYPE, f"type error (browser closed): {msg}", {"target": target}, t0)
            return self._fail(ERROR_TYPE, f"type error: {e}", {"target": target}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"type error: {e}", {"target": target}, t0)

    def _do_select(self, target: Dict[str, Any], value: Dict[str, Any], verify_hint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            loc, meta = self._resolve_target(target)
            if not loc:
                return self._fail(ERROR_NOT_FOUND, "select target not found", {"target": target, "resolve": meta}, t0)

            before = self.page.url
            field_kind = (value.get("field_kind") or "").lower()
            desired_value = value.get("value", None)
            selected_text = value.get("selected_text", None)

            logger.info("[execute] select strategy=%s", meta.get("strategy"))
            logger.debug("selector=%s value=%s selected_text=%s", meta.get("selector"), desired_value, selected_text)

            tag = (target.get("tag") or "").lower()
            if tag == "select":
                try:
                    loc.select_option(value=str(desired_value) if desired_value is not None else None)
                except Exception:
                    loc.click()
                    self.page.wait_for_timeout(150)
                    choice = str(selected_text or desired_value or "")
                    if choice:
                        opt = self.page.get_by_role("option", name=choice).first
                        if opt.count() > 0:
                            opt.click()
                        else:
                            self.page.locator(f'text="{choice}"').first.click()
                self.page.wait_for_timeout(250)
            else:
                loc.click(timeout=self.implicit_wait_ms)
                self.page.wait_for_timeout(250)
                if field_kind in ("select", "dropdown", "listbox") and (selected_text or desired_value):
                    choice = str(selected_text or desired_value)
                    opt = self.page.get_by_role("option", name=choice).first
                    if opt.count() > 0:
                        opt.click()
                    else:
                        self.page.locator(f'text="{choice}"').first.click()
                    self.page.wait_for_timeout(200)

            self._maybe_cache_identity(target, meta)

            if verify_hint:
                v = self._verify(verify_hint, step_target=target, before_url=before, select_value=value)
                if not v["ok"]:
                    v["extracted"] = {**(v.get("extracted") or {}), "resolve": meta}
                    return v

            return self._ok(
                "select ok",
                {"resolve": meta, "value": value},
                t0,
                matched_locator=meta.get("selector", ""),
                matches=meta.get("matches", 0),
            )

        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"select timeout: {e}", {"target": target, "value": value}, t0)
        except PWError as e:
            msg = str(e)
            if "has been closed" in msg:
                self._hard_reset()
                self.launch()
                return self._fail(ERROR_SELECT, f"select error (browser closed): {msg}", {"target": target, "value": value}, t0)
            return self._fail(ERROR_SELECT, f"select error: {e}", {"target": target, "value": value}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"select error: {e}", {"target": target, "value": value}, t0)

    # -------------------------------------------------------------------------
    # Target identity cache helpers
    # -------------------------------------------------------------------------
    def _maybe_cache_identity(self, target: Dict[str, Any], meta: Dict[str, Any]) -> None:
        sel = (meta.get("selector") or "").strip()
        if not sel:
            return
        eid = (target.get("element_id") or "").strip()
        tref = (target.get("target_ref") or "").strip()
        fp = (target.get("element_fingerprint") or "").strip()
        if eid:
            self._id_to_selector[eid] = sel
        if tref:
            self._ref_to_selector[tref] = sel
        if fp:
            self._fp_to_selector[fp] = sel

    # -------------------------------------------------------------------------
    # Selector helper + resolution
    # -------------------------------------------------------------------------
    def _try_locator(self, sel: str):
        try:
            loc = self.page.locator(sel)
            return loc, loc.count()
        except Exception:
            return None, 0

    def _extract_title_from_snippet(self, text_snippet: str) -> str:
        s = (text_snippet or "").strip()
        if not s:
            return ""
        s = s.replace("\n", " ").strip()
        for token in (" http://", " https://", "http://", "https://"):
            idx = s.find(token)
            if idx > 0:
                s = s[:idx].strip()
                break
        return s[:140]

    def _resolve_target(self, target: Dict[str, Any], *, want_fillable: bool = False):
        meta: Dict[str, Any] = {"strategy": "", "selector": "", "matches": 0}

        if not target:
            return None, {"strategy": "none", "selector": "", "matches": 0}

        # 0) cached selector by stable ids
        eid = (target.get("element_id") or "").strip()
        tref = (target.get("target_ref") or "").strip()
        fp = (target.get("element_fingerprint") or "").strip()

        for key, cache, strategy in [
            (eid, self._id_to_selector, "cache_element_id"),
            (tref, self._ref_to_selector, "cache_target_ref"),
            (fp, self._fp_to_selector, "cache_fingerprint"),
        ]:
            if key and key in cache:
                sel = cache[key]
                loc, c = self._try_locator(sel)
                if loc and c > 0:
                    meta.update({"strategy": strategy, "selector": sel, "matches": c})
                    return loc.first, meta

        def _label_to_fillable(label_text: str):
            if not label_text:
                return None, {"strategy": "label_to_fillable_empty", "selector": "", "matches": 0}
            try:
                label_loc = self.page.get_by_text(label_text, exact=False).first
                if label_loc.count() <= 0:
                    return None, {"strategy": "label_to_fillable_not_found", "selector": f"text~={label_text}", "matches": 0}

                ancestors = [
                    label_loc.locator("xpath=ancestor-or-self::*[@role='listitem'][1]").first,
                    label_loc.locator("xpath=ancestor-or-self::*[self::div][1]").first,
                    label_loc.locator("xpath=ancestor-or-self::*[self::form][1]").first,
                ]

                for anc in ancestors:
                    try:
                        if anc.count() <= 0:
                            continue
                        inp = anc.locator("input,textarea,[contenteditable='true'],div[role='textbox']").first
                        if inp.count() > 0:
                            return inp, {"strategy": "label_container_fillable", "selector": f"label~={label_text} -> input", "matches": inp.count()}
                    except Exception:
                        continue
            except Exception:
                pass
            return None, {"strategy": "label_to_fillable_failed", "selector": f"text~={label_text}", "matches": 0}

        # 1) selector_candidates
        cands = target.get("selector_candidates") or []
        if isinstance(cands, list) and cands:
            structural: List[Dict[str, Any]] = []
            textual: List[Dict[str, Any]] = []
            for cand in cands:
                kind = (cand.get("kind") or "").strip().lower()
                (textual if kind == "text" else structural).append(cand)

            for cand in structural:
                kind = (cand.get("kind") or "").strip().lower()
                val = (cand.get("value") or "").strip()

                if kind in ("css", "css_path") and val:
                    for sel in (f"css={val}", val):
                        loc, c = self._try_locator(sel)
                        if loc and c > 0:
                            meta.update({"strategy": f"candidate_{kind}", "selector": sel, "matches": c})
                            return loc.first, meta

                if kind == "xpath" and val:
                    sel = f"xpath={val}"
                    loc, c = self._try_locator(sel)
                    if loc and c > 0:
                        meta.update({"strategy": "candidate_xpath", "selector": sel, "matches": c})
                        return loc.first, meta

                if kind in ("role", "role_name"):
                    role = (cand.get("role") or "").strip()
                    name = (cand.get("name") or "").strip()
                    if role:
                        loc = self.page.get_by_role(role, name=name) if name else self.page.get_by_role(role)
                        if loc.count() > 0:
                            meta.update({"strategy": f"candidate_{kind}", "selector": f"role={role} name={name}", "matches": loc.count()})
                            return loc.first, meta

            for cand in textual:
                val = (cand.get("value") or "").strip()
                if not val:
                    continue
                if want_fillable:
                    loc, m = _label_to_fillable(val)
                    if loc:
                        return loc, m
                loc = self.page.get_by_text(val, exact=False)
                if loc.count() > 0:
                    meta.update({"strategy": "candidate_text", "selector": f"text~={val}", "matches": loc.count()})
                    return loc.first, meta

        # 2) fallbacks
        css_path = (target.get("css_path") or "").strip()
        if css_path:
            for sel in (f"css={css_path}", css_path):
                loc, c = self._try_locator(sel)
                if loc and c > 0:
                    meta.update({"strategy": "css_path", "selector": sel, "matches": c})
                    return loc.first, meta

        xpath = (target.get("xpath") or "").strip()
        if xpath:
            sel = f"xpath={xpath}"
            loc, c = self._try_locator(sel)
            if loc and c > 0:
                meta.update({"strategy": "xpath", "selector": sel, "matches": c})
                return loc.first, meta

        role = (target.get("role") or "").strip()
        a11y = (target.get("a11y_name") or "").strip()
        if role:
            loc = self.page.get_by_role(role, name=a11y) if a11y else self.page.get_by_role(role)
            if loc.count() > 0:
                meta.update({"strategy": "role_name", "selector": f"role={role} name={a11y}", "matches": loc.count()})
                return loc.first, meta

        el_id = (target.get("id") or "").strip()
        if el_id:
            for sel in (f"css=#{el_id}", f"#{el_id}"):
                loc, c = self._try_locator(sel)
                if loc and c > 0:
                    meta.update({"strategy": "id", "selector": sel, "matches": c})
                    return loc.first, meta

        # SERP-friendly text fallback
        tag = (target.get("tag") or "").strip().lower()
        txt = (target.get("a11y_name") or "").strip()
        snip = (target.get("text_snippet") or "").strip()
        title = self._extract_title_from_snippet(snip)
        fallback_text = (txt or title or snip).split("\n")[0].strip()[:140]

        if fallback_text:
            if tag == "a":
                try:
                    lnk = self.page.get_by_role("link", name=fallback_text).first
                    if lnk.count() > 0:
                        meta.update({"strategy": "fallback_role_link", "selector": f"role=link name~={fallback_text}", "matches": lnk.count()})
                        return lnk, meta
                except Exception:
                    pass

            loc = self.page.get_by_text(fallback_text, exact=False)
            if loc.count() > 0:
                meta.update({"strategy": "fallback_text", "selector": f"text~={fallback_text}", "matches": loc.count()})
                return loc.first, meta

        return None, {"strategy": "unresolved", "selector": "", "matches": 0}

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------
    def _verify(
        self,
        verify_hint: Dict[str, Any],
        *,
        step_target: Optional[Dict[str, Any]],
        before_url: str,
        typed_text: Optional[str] = None,
        select_value: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        try:
            strategy = str(verify_hint.get("strategy") or "")
            checks = verify_hint.get("checks") or []

            # Special-case: submit semantics are OR (url changed OR url contains tokens OR confirmation text).
            if strategy == "url_changed_or_confirmation":
                url_changed = False
                try:
                    url_changed = (self.page.url != before_url)
                except Exception:
                    url_changed = False

                url_tokens = []
                for c in checks:
                    if (c.get("op") or "").strip() == "url_contains_any":
                        url_tokens = [str(x) for x in (c.get("values") or []) if str(x)]
                        break

                url_contains_any = False
                if url_tokens:
                    try:
                        url_contains_any = any(tok in (self.page.url or "") for tok in url_tokens)
                    except Exception:
                        url_contains_any = False

                confirm_phrases = [
                    "Your response has been recorded",
                    "Submit another response",
                    "Edit your response",
                    "Thank you",
                    "Thanks",
                ]
                confirm_present = self._page_contains_any(confirm_phrases)

                if url_changed or url_contains_any or confirm_present:
                    return self._ok(
                        "verify ok (submit)",
                        {
                            "verify_hint": verify_hint,
                            "url_changed": url_changed,
                            "url_contains_any": url_contains_any,
                            "confirmation_text": confirm_present,
                        },
                        t0,
                    )

                return self._fail(
                    ERROR_ASSERT,
                    f"verify failed: submit confirmation not detected (url still {self.page.url})",
                    {
                        "verify_hint": verify_hint,
                        "url_changed": url_changed,
                        "url_contains_any": url_contains_any,
                        "confirmation_text": confirm_present,
                        "before_url": before_url,
                        "after_url": self.page.url,
                    },
                    t0,
                )

            # Default: checks are AND (with optional_* being soft).
            for c in checks:
                op = (c.get("op") or "").strip()

                if op == "optional_url_change":
                    continue

                if op == "browser_connected":
                    if not self._browser:
                        return self._fail(ERROR_ASSERT, "verify failed: browser not connected", {"verify_hint": verify_hint}, t0)

                elif op == "context_exists":
                    if not self._context:
                        return self._fail(ERROR_ASSERT, "verify failed: context missing", {"verify_hint": verify_hint}, t0)

                elif op == "page_exists":
                    if not self._page or self._page.is_closed():
                        return self._fail(ERROR_ASSERT, "verify failed: page missing/closed", {"verify_hint": verify_hint}, t0)

                elif op == "url_changed":
                    if self.page.url == before_url:
                        return self._fail(ERROR_ASSERT, f"verify failed: url did not change (still {self.page.url})", {"verify_hint": verify_hint}, t0)

                elif op == "url_contains":
                    val = str(c.get("value") or "")
                    if val and val not in self.page.url:
                        return self._fail(ERROR_ASSERT, f"verify failed: url does not contain {val}", {"url": self.page.url, "verify_hint": verify_hint}, t0)

                elif op == "url_contains_any":
                    vals = [str(x) for x in (c.get("values") or []) if str(x)]
                    if vals and not any(v in self.page.url for v in vals):
                        return self._fail(ERROR_ASSERT, f"verify failed: url does not contain any of {vals}", {"url": self.page.url, "verify_hint": verify_hint}, t0)

                elif op in ("element_exists", "optional_element_exists", "element_value_equals", "element_state_matches", "element_is_focused"):
                    if not step_target:
                        if op == "optional_element_exists":
                            continue
                        return self._fail(ERROR_ASSERT, f"verify failed: no step_target for {op}", {"verify_hint": verify_hint}, t0)

                    loc, meta = self._resolve_target(step_target, want_fillable=(op in ("element_value_equals", "element_is_focused")))
                    if not loc:
                        if op == "optional_element_exists":
                            continue
                        return self._fail(ERROR_ASSERT, f"verify failed: cannot resolve target for {op}", {"resolve": meta, "verify_hint": verify_hint}, t0)

                    if op == "element_exists":
                        if loc.count() <= 0:
                            return self._fail(ERROR_ASSERT, "verify failed: element not found", {"resolve": meta}, t0)

                    elif op == "element_is_focused":
                        try:
                            if not loc.evaluate("el => el === document.activeElement"):
                                return self._fail(ERROR_ASSERT, "verify failed: element not focused", {"resolve": meta}, t0)
                        except Exception:
                            return self._fail(ERROR_ASSERT, "verify failed: could not verify focus", {"resolve": meta}, t0)

                    elif op == "element_value_equals":
                        expected = typed_text if typed_text is not None else c.get("value")
                        try:
                            try:
                                observed = (loc.input_value() or "").strip()
                            except Exception:
                                observed = (loc.inner_text() or "").strip()
                            if expected is not None and str(expected) not in str(observed):
                                return self._fail(ERROR_ASSERT, f"verify failed: value mismatch (observed='{observed[:80]}')", {"resolve": meta, "observed": observed}, t0)
                        except Exception:
                            pass

                    elif op == "element_state_matches":
                        desired = (select_value or {})
                        want_checked = desired.get("checked", None)
                        if want_checked is None:
                            continue
                        try:
                            aria = loc.get_attribute("aria-checked")
                            if aria is not None:
                                obs = (aria == "true")
                                if bool(want_checked) != bool(obs):
                                    return self._fail(ERROR_ASSERT, f"verify failed: aria-checked mismatch (aria={aria})", {"resolve": meta}, t0)
                        except Exception:
                            pass

                else:
                    logger.debug("[verify] unknown op ignored: %s", op)

            return self._ok("verify ok", {"verify_hint": verify_hint}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"verify error: {e}", {"verify_hint": verify_hint}, t0)

    # -------------------------------------------------------------------------
    # Evidence helpers
    # -------------------------------------------------------------------------
    def _screenshot(self) -> str:
        fname = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(self.artifacts_dir, fname)
        try:
            if self._page and not self._page.is_closed():
                self.page.screenshot(path=path, full_page=True)
                return path
        except Exception:
            pass
        return ""

    def screenshot_base64(self) -> str:
        try:
            shot = self.page.screenshot(full_page=True)
            return base64.b64encode(shot).decode("utf-8")
        except Exception:
            return ""

    def _ok(self, message: str, extracted: Dict[str, Any], t0: float, matched_locator: str = "", matches: int = 0) -> Dict[str, Any]:
        return {
            "ok": True,
            "error_type": None,
            "message": message,
            "evidence": {
                "url": self.page.url if self._page and not self._page.is_closed() else "",
                "screenshot_path": self._screenshot(),
                "matched_locator": matched_locator,
                "matches": matches,
            },
            "extracted": extracted or {},
            "timing_ms": int((time.time() - t0) * 1000),
        }

    def _fail(self, error_type: str, message: str, extracted: Dict[str, Any], t0: float, matched_locator: str = "", matches: int = 0) -> Dict[str, Any]:
        url = ""
        shot = ""
        try:
            if self._page and not self._page.is_closed():
                url = self.page.url
                shot = self._screenshot()
        except Exception:
            pass
        return {
            "ok": False,
            "error_type": error_type,
            "message": message,
            "evidence": {
                "url": url,
                "screenshot_path": shot,
                "matched_locator": matched_locator,
                "matches": matches,
            },
            "extracted": extracted or {},
            "timing_ms": int((time.time() - t0) * 1000),
        }


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def load_workflow_steps(path: str) -> List[Dict[str, Any]]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    steps = data.get("steps") or []
    return [s for s in steps if (s.get("agent") == "browser")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="recordings/20260107_104916/workflow_steps.v1.json")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--artifacts", default=os.getenv("BROWSER_ARTIFACTS_DIR", "artifacts_browser"))
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("BROWSER_MAX_RETRIES", "2")))
    ap.add_argument("--observe-between", action="store_true")
    ap.add_argument("--channel", default=os.getenv("BROWSER_CHANNEL", ""), help="e.g. chrome")
    ap.add_argument("--verbose-console", action="store_true")
    ap.add_argument("--no-google-serp-wait", action="store_true")
    ap.add_argument("--no-click-idempotent-skip", action="store_true")
    ap.add_argument("--submit-wait-ms", type=int, default=int(os.getenv("BROWSER_SUBMIT_WAIT_MS", "8000")))
    args = ap.parse_args()

    agent = PrimitiveBrowserAgent(
        headless=not args.headful,
        artifacts_dir=args.artifacts,
        backoff=Backoff(max_retries=args.max_retries),
        stealth_mode=True,
        default_channel=(args.channel.strip() or None),
        verbose_console=bool(args.verbose_console),
        google_serp_wait_rso=(not bool(args.no_google_serp_wait)),
        click_idempotent_skip=(not bool(args.no_click_idempotent_skip)),
        submit_wait_ms=int(args.submit_wait_ms),
    )

    try:
        steps = load_workflow_steps(args.workflow)
        if not steps or steps[0].get("type") != "browser_launch":
            agent.launch()

        ctx: Dict[str, Any] = {"vars": {}}

        for s in steps:
            print("\n" + "=" * 90)
            print(f"RUN: {s.get('step_id')}  type={s.get('type')}")
            res = agent.run_primitive(s, ctx)
            print(json.dumps(res, indent=2))
            if not res.get("ok"):
                break

            if args.observe_between:
                obs = agent.observe()
                o = ((obs.get("extracted") or {}).get("observation") or {})
                print(json.dumps({"url": o.get("url"), "title": o.get("title"), "interactive": len(o.get("interactive") or [])}, indent=2))

        input("\nPress Enter to close...")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
