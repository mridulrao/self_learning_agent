from __future__ import annotations

"""
Google Forms Browser Agent
"""

import os
import time
import uuid
import base64
import logging
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError, Error as PWError
import anthropic
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Env helpers
# =============================================================================
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


# =============================================================================
# Logging
# =============================================================================
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


logger = setup_logger("GoogleFormsBrowserAgent", level=logging.INFO)


# =============================================================================
# Error types
# =============================================================================
ERROR_TIMEOUT = "timeout"
ERROR_NOT_FOUND = "not_found"
ERROR_NAVIGATION = "navigation_error"
ERROR_CLICK = "click_error"
ERROR_TYPE = "type_error"
ERROR_EXTRACT = "extract_error"
ERROR_ASSERT = "assertion_failed"
ERROR_UNKNOWN = "unknown_error"
ERROR_VISION = "vision_error"
ERROR_VALIDATION = "validation_failed"


@dataclass
class Backoff:
    max_retries: int = 2
    base_delay_s: float = 0.25
    max_delay_s: float = 1.5

    def sleep(self, attempt: int) -> None:
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        jitter = (attempt % 3) * 0.03
        time.sleep(delay + jitter)


# =============================================================================
# JSON extraction helpers (robust to fences / chatter)
# =============================================================================
def _extract_first_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_positions = []
    for ch in ["{", "["]:
        i = cleaned.find(ch)
        if i != -1:
            start_positions.append(i)
    if not start_positions:
        return None

    start = min(start_positions)
    s = cleaned[start:]

    stack = []
    for idx, ch in enumerate(s):
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack[-1]
            if (open_ch == "{" and ch == "}") or (open_ch == "[" and ch == "]"):
                stack.pop()
                if not stack:
                    candidate = s[: idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
    return None


def _deep_get(d: Dict[str, Any], path: str) -> Any:
    if not path:
        return None
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

# Add this near your other regexes
FORM_ID_NUMERIC_RE = re.compile(r"\bForm\s*ID\s*:\s*([0-9]+)\b", re.IGNORECASE)


# =============================================================================
# Google Forms Browser Agent (Google-form-focused)
# =============================================================================
class GoogleFormsBrowserAgent:
    """
    Refactored Browser Agent:
      - Focused ONLY on Google Forms flows
      - Includes validation loops to fill required missing fields
      - Supports:
          - text inputs + textarea
          - Google Forms Material textbox: div[role="textbox"] (contenteditable)
          - checkbox / radio / scale / dropdown
      - Submission validation:
          - DOM-based required checks
          - DOM success text checks
          - Optional Claude vision confirmation

    Added:
      - Extract + return FormID (from under title) and filled details on successful submit
      - Close browser automatically after successful submit (idempotent)
    """

    REQUIRED_ERR_TEXT = "This is a required question"
    SUCCESS_TEXT_CANDIDATES = [
        "Your response has been recorded",
        "Thanks for submitting",
        "Thank you",
        "Response recorded",
        "Submit another response",
    ]

    # Google Form IDs often look like "1FAIpQLS..." (long token)
    FORM_ID_REGEXES = [
        re.compile(r"\b(1FAIpQL[0-9A-Za-z_-]{10,})\b"),
        re.compile(r"\b(Form ID|FormID)\s*[:\-]?\s*([0-9A-Za-z_-]{10,})\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        headless: bool = True,
        viewport: Tuple[int, int] = (1280, 800),
        artifacts_dir: str = "artifacts",
        default_timeout_ms: int = 15_000,
        implicit_wait_ms: int = 6_000,
        backoff: Backoff = Backoff(),
        close_popups: bool = True,
        anthropic_api_key: Optional[str] = None,
        use_vision: bool = True,
        stealth_mode: bool = True,
        use_chrome_channel: bool = True,
        fill_passes: int = 2,
        missing_retry_delay_s: float = 0.25,
        scroll_between_questions_px: int = 150,
        lazy_mount_scroll: bool = True,
    ):
        self.headless = headless
        self.viewport = viewport
        self.artifacts_dir = artifacts_dir
        self.default_timeout_ms = default_timeout_ms
        self.implicit_wait_ms = implicit_wait_ms
        self.backoff = backoff
        self.close_popups = close_popups
        self.use_vision = use_vision
        self.stealth_mode = stealth_mode
        self.use_chrome_channel = use_chrome_channel

        self.fill_passes = max(1, int(fill_passes))
        self.missing_retry_delay_s = float(missing_retry_delay_s)
        self.scroll_between_questions_px = int(scroll_between_questions_px)
        self.lazy_mount_scroll = bool(lazy_mount_scroll)

        if anthropic_api_key and use_vision:
            logger.info("Initializing Claude Vision for GoogleFormsBrowserAgent")
            self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            logger.info("Claude Vision disabled for GoogleFormsBrowserAgent")
            self.claude = None

        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._closed = False

        # ---- NEW: state to return on success ----
        self._last_form_data: Dict[str, Any] = {}
        self._last_fill_results: Dict[str, Any] = {}
        self._last_form_id: str = ""

        os.makedirs(self.artifacts_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def launch(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info("Launching browser...")
            self._pw = sync_playwright().start()

            launch_args = []
            if self.stealth_mode:
                launch_args = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                ]

            launch_options = {
                "headless": self.headless,
                "args": launch_args,
            }

            if self.use_chrome_channel:
                try:
                    launch_options["channel"] = "chrome"
                    self._browser = self._pw.chromium.launch(**launch_options)
                    logger.info("✓ Launched using installed Chrome (channel=chrome)")
                except Exception as e:
                    logger.warning(f"Chrome channel failed ({e}), falling back to chromium")
                    launch_options.pop("channel", None)
                    self._browser = self._pw.chromium.launch(**launch_options)
            else:
                self._browser = self._pw.chromium.launch(**launch_options)

            context_options = {"viewport": {"width": self.viewport[0], "height": self.viewport[1]}}

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
            logger.info("✓ Browser launched successfully")
            return self._ok("launched", {}, t0)

        except Exception as e:
            logger.exception(f"Failed to launch browser: {e}")
            return self._fail(ERROR_UNKNOWN, f"launch failed: {e}", {}, t0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        logger.info("Closing browser...")
        try:
            try:
                if self._context:
                    self._context.close()
            except Exception:
                pass
        finally:
            try:
                if self._browser:
                    self._browser.close()
            except Exception:
                pass
            finally:
                try:
                    if self._pw:
                        self._pw.stop()
                except Exception:
                    pass
        logger.info("✓ Browser closed")

    @property
    def page(self):
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page

    # -------------------------------------------------------------------------
    # Orchestrator-facing entrypoint
    # -------------------------------------------------------------------------
    def run_step(self, step: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not self._page:
            self.launch()

        t0 = time.time()
        step_id = step.get("id", "unknown")
        action = (step.get("action") or "").strip()
        args = step.get("args") or {}
        assertions = step.get("assertions") or []
        max_retries = int(step.get("retries", self.backoff.max_retries))

        logger.info(f"{'='*80}")
        logger.info(f"EXECUTING STEP: {step_id}")
        logger.info(f"Action: {action}")
        logger.info(f"Args: {args}")
        logger.info(f"Max retries: {max_retries}")
        logger.info(f"{'='*80}")

        prev_url = self.page.url
        ctx.setdefault("_evidence", {})
        ctx["_evidence"]["prev_url"] = prev_url

        def _do_action() -> Dict[str, Any]:
            if action == "goto":
                return self.goto(args["url"])
            if action == "wait_for":
                return self.wait_for(args["locator"])
            if action == "wait_for_url_change":
                before = args.get("before_url") or ctx.get("_evidence", {}).get("prev_url", "")
                return self.wait_for_url_change(before)
            if action == "scroll":
                return self.scroll(direction=args.get("direction", "down"), amount=args.get("amount", 300))

            if action == "fill_google_form":
                return self.fill_google_form(
                    form_data=args["form_data"],
                    passes=int(args.get("passes", self.fill_passes)),
                )
            if action == "submit_form":
                return self.submit_form(
                    submit_button_locator=args.get("submit_button_locator"),
                    require_no_missing_required=args.get("require_no_missing_required", True),
                    return_filled_details=args.get("return_filled_details", True),
                    return_form_id=args.get("return_form_id", True),
                )
            if action == "get_missing_required":
                missing = self._get_required_error_questions()
                return self._ok("missing_required", {"missing_required_questions": missing}, t0)

            # Optional: extract text
            if action == "extract_text":
                return self.extract_text_from_page(
                    locator=args.get("locator"),
                    description=args.get("description"),
                )

            # Optional: form id extraction as a standalone step
            if action == "extract_form_id":
                form_id = self._extract_form_id_best_effort()
                if form_id:
                    self._last_form_id = form_id
                    return self._ok("extract_form_id ok", {"form_id": form_id}, t0)
                return self._fail(ERROR_EXTRACT, "Could not extract form_id", {}, t0)

            if action == "verify_by_vision":
                return self.verify_by_vision(args["expected_state"])

            return self._fail(ERROR_UNKNOWN, f"unknown action: {action}", {}, t0)

        last = None
        for attempt in range(max_retries + 1):
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            res = _do_action()
            logger.info(f"Result: ok={res.get('ok')}, message={res.get('message')}")

            if res.get("ok"):
                ares = self.assert_all(assertions, ctx)
                if not ares["ok"]:
                    res = ares
                else:
                    res["extracted"] = {**(res.get("extracted") or {}), **(ares.get("extracted") or {})}

            if res.get("ok"):
                logger.info(f"✓ Step {step_id} succeeded")
                res["step_id"] = step_id
                res["executor"] = "browser"
                res["action"] = action
                res["attempt"] = attempt + 1
                return res

            last = res
            logger.warning(f"✗ Attempt {attempt + 1} failed: {res.get('message')}")

            if res.get("error_type") in {
                ERROR_TIMEOUT,
                ERROR_CLICK,
                ERROR_NAVIGATION,
                ERROR_UNKNOWN,
                ERROR_VISION,
                ERROR_EXTRACT,
                ERROR_VALIDATION,
            } and attempt < max_retries:
                self.backoff.sleep(attempt)
                continue
            break

        logger.error(f"Step {step_id} failed after all retries")
        last = last or self._fail(ERROR_UNKNOWN, "unknown failure", {}, t0)
        last["step_id"] = step_id
        last["executor"] = "browser"
        last["action"] = action
        last["attempt"] = max_retries + 1
        return last

    # -------------------------------------------------------------------------
    # Core actions
    # -------------------------------------------------------------------------
    def goto(self, url: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Navigating to: {url}")
            before = self.page.url
            self.page.goto(url, wait_until="domcontentloaded")
            self._maybe_close_popups()
            self._maybe_mount_lazy_sections()
            logger.info(f"✓ Navigation successful: {self.page.url}")
            return self._ok("goto ok", {"previous_url": before, "url": self.page.url}, t0)
        except PWTimeoutError as e:
            logger.error(f"Navigation timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"goto timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Navigation error: {e}")
            return self._fail(ERROR_NAVIGATION, f"goto navigation error: {e}", {}, t0)

    def wait_for(self, locator: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            sel = self._to_selector(locator)
            logger.info(f"Waiting for element: {sel}")
            self.page.wait_for_selector(sel, state="attached", timeout=self.implicit_wait_ms)
            return self._ok("wait_for ok", {"waited_for": sel}, t0, matched_locator=sel, matches=1)
        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"wait_for timeout: {e}", {}, t0)
        except PWError as e:
            return self._fail(ERROR_UNKNOWN, f"wait_for error: {e}", {}, t0)

    def wait_for_url_change(self, before_url: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Waiting for URL to change from: {before_url}")
            self._wait_for_url_change_or_timeout(before_url, self.implicit_wait_ms)
            return self._ok("url_changed ok", {"previous_url": before_url, "url": self.page.url}, t0)
        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"url_change timeout: {e}", {}, t0)

    def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        t0 = time.time()
        try:
            direction_lower = direction.lower()
            if direction_lower == "down":
                self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction_lower == "up":
                self.page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction_lower == "top":
                self.page.evaluate("window.scrollTo(0, 0)")
            elif direction_lower == "bottom":
                self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                return self._fail(ERROR_UNKNOWN, f"Unknown scroll direction: {direction}", {}, t0)

            time.sleep(0.25)
            return self._ok("scroll ok", {"direction": direction, "amount": amount}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"scroll error: {e}", {}, t0)

    # -------------------------------------------------------------------------
    # Google Forms: fill + validation loops
    # -------------------------------------------------------------------------
    def fill_google_form(self, form_data: Dict[str, Any], passes: int = 2) -> Dict[str, Any]:
        t0 = time.time()
        try:
            if not isinstance(form_data, dict) or not form_data:
                return self._fail(ERROR_UNKNOWN, "form_data must be a non-empty dict", {}, t0)

            # ---- NEW: remember what was intended to be filled ----
            self._last_form_data = dict(form_data)

            passes = max(1, int(passes))

            all_questions = self._get_all_form_questions()
            logger.info(f"Found {len(all_questions)} question blocks on the form")

            results_by_pass: List[Dict[str, Any]] = []
            overall_results: Dict[str, Any] = {}

            keys_to_attempt = list(form_data.keys())

            for p in range(passes):
                logger.info("=" * 80)
                logger.info(f"FILL PASS {p+1}/{passes} - attempting {len(keys_to_attempt)} keys")
                logger.info("=" * 80)

                if self.lazy_mount_scroll:
                    self._maybe_mount_lazy_sections()

                pass_results: Dict[str, Any] = {}

                for question_key in keys_to_attempt:
                    answer = form_data[question_key]
                    logger.info(f"-> Filling: '{question_key}' = '{str(answer)[:80]}'")

                    container = self._find_question_container(question_key, all_questions)
                    if not container:
                        pass_results[question_key] = {"status": "not_found", "answer": answer}
                        continue

                    try:
                        container.scroll_into_view_if_needed()
                        time.sleep(0.2)
                    except Exception:
                        pass

                    fill_result = self._fill_form_field(container, question_key, str(answer))
                    pass_results[question_key] = fill_result

                    try:
                        self.page.evaluate(f"window.scrollBy(0, {self.scroll_between_questions_px})")
                        time.sleep(0.15)
                    except Exception:
                        pass

                results_by_pass.append(pass_results)

                for k, v in pass_results.items():
                    overall_results[k] = v

                missing_required_titles = self._get_required_error_questions()
                missing_keys = self._map_missing_titles_to_form_keys(missing_required_titles, form_data)

                logger.info(f"Missing required question blocks detected: {missing_required_titles}")
                logger.info(f"Mapped missing required to keys: {missing_keys}")

                if not missing_required_titles:
                    break

                if p < passes - 1 and missing_keys:
                    keys_to_attempt = missing_keys
                    time.sleep(self.missing_retry_delay_s)
                    continue

                if missing_required_titles and not missing_keys:
                    logger.warning("Missing required fields exist but could not map them to provided form_data keys.")
                    break

            # ---- NEW: remember fill results for return on submit ----
            self._last_fill_results = dict(overall_results)

            filled_count = sum(1 for r in overall_results.values() if r.get("status") == "filled")
            total = len(form_data)
            success_rate = (filled_count / total) if total else 0.0

            missing_required_titles_final = self._get_required_error_questions()
            ok = (len(missing_required_titles_final) == 0)

            msg = f"Filled {filled_count}/{total} fields; missing_required={len(missing_required_titles_final)}"
            if ok:
                return self._ok(
                    msg,
                    {
                        "results": overall_results,
                        "passes": passes,
                        "results_by_pass": results_by_pass,
                        "filled_count": filled_count,
                        "total_fields": total,
                        "success_rate": success_rate,
                        "missing_required_questions": missing_required_titles_final,
                    },
                    t0,
                )
            else:
                return self._fail(
                    ERROR_VALIDATION,
                    msg,
                    {
                        "results": overall_results,
                        "passes": passes,
                        "results_by_pass": results_by_pass,
                        "filled_count": filled_count,
                        "total_fields": total,
                        "success_rate": success_rate,
                        "missing_required_questions": missing_required_titles_final,
                    },
                    t0,
                )

        except Exception as e:
            logger.exception(f"fill_google_form error: {e}")
            return self._fail(ERROR_UNKNOWN, f"fill_google_form error: {e}", {}, t0)

    def submit_form(
        self,
        submit_button_locator: Union[str, Dict[str, Any], None] = None,
        require_no_missing_required: bool = True,
        return_filled_details: bool = True,
        return_form_id: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit the form and validate submission.

        NEW on success:
          - returns:
              - form_id (best-effort)
              - filled_details (what we attempted to fill)
              - fill_results (per-field fill status)
        """
        t0 = time.time()
        try:
            logger.info("Submitting form...")

            self._maybe_close_popups()

            # ---- NEW: extract form_id BEFORE submit (form page has it; confirmation may not) ----
            form_id = ""
            if return_form_id:
                form_id = self._extract_form_id_best_effort()
                if form_id:
                    self._last_form_id = form_id

            missing_before = self._get_required_error_questions()
            if require_no_missing_required and missing_before:
                return self._fail(
                    ERROR_VALIDATION,
                    "Cannot submit: required questions missing",
                    {"missing_required_questions": missing_before, "submitted": False},
                    t0,
                )

            before_url = self.page.url

            clicked = self._click_submit(submit_button_locator)
            if not clicked:
                return self._fail(ERROR_NOT_FOUND, "Could not find submit button", {}, t0)

            time.sleep(1.2)
            self._maybe_close_popups()

            after_url = self.page.url
            url_changed = (after_url != before_url)

            dom_success = self._detect_dom_submission_success()
            missing_after = self._get_required_error_questions()

            vision = None
            if self.claude:
                vision = self._validate_form_submission_by_vision()

            submitted = bool(
                url_changed
                or dom_success
                or (vision and vision.get("submitted") and vision.get("confidence", 0) >= 0.7)
            )

            extracted_payload: Dict[str, Any] = {
                "submitted": submitted,
                "before_url": before_url,
                "after_url": after_url,
                "url_changed": url_changed,
                "dom_success": dom_success,
                "missing_required_questions": missing_after,
                "vision": vision,
            }

            # ---- NEW: attach form_id + details on success ----
            if submitted:
                if return_form_id:
                    extracted_payload["form_id"] = form_id or self._last_form_id
                if return_filled_details:
                    extracted_payload["filled_details"] = dict(self._last_form_data or {})
                    extracted_payload["fill_results"] = dict(self._last_fill_results or {})

                # close browser after success
                self.close()

                return self._ok(
                    "Form submitted successfully",
                    extracted_payload,
                    t0,
                )

            # Not submitted
            return self._fail(
                ERROR_VALIDATION,
                "Could not confirm submission",
                extracted_payload,
                t0,
            )

        except PWTimeoutError as e:
            return self._fail(ERROR_TIMEOUT, f"submit_form timeout: {e}", {}, t0)
        except PWError as e:
            return self._fail(ERROR_CLICK, f"submit_form error: {e}", {}, t0)
        except Exception as e:
            logger.exception(f"submit_form error: {e}")
            return self._fail(ERROR_UNKNOWN, f"submit_form error: {e}", {}, t0)

    # -------------------------------------------------------------------------
    # Form ID extraction (best-effort)
    # -------------------------------------------------------------------------
    def _extract_form_id_best_effort(self) -> str:
        """
        DOM-first extraction of "Form ID: <digits>" shown under the form title.
        """
        try:
            self._maybe_close_popups()

            # Ensure top of form is visible (Form ID is near title)
            try:
                self.page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.15)
            except Exception:
                pass

            # 1) Best: direct regex text locator (fast + robust)
            # Playwright supports text=/regex/
            try:
                loc = self.page.locator(r"text=/Form\s*ID\s*:\s*[0-9]+/i").first
                if loc.count() > 0:
                    txt = (loc.inner_text() or "").strip()
                    m = FORM_ID_NUMERIC_RE.search(txt)
                    if m:
                        return m.group(1).strip()
            except Exception:
                pass

            # 2) Search within header/title region (Google Forms viewer classes)
            header_selectors = [
                ".freebirdFormviewerViewHeaderHeader",
                ".freebirdFormviewerViewHeaderTitleRow",
                ".freebirdFormviewerViewHeaderDescription",
                "header",
            ]
            for sel in header_selectors:
                try:
                    header = self.page.locator(sel).first
                    if header.count() <= 0:
                        continue
                    txt = (header.inner_text() or "").strip()
                    m = FORM_ID_NUMERIC_RE.search(txt)
                    if m:
                        return m.group(1).strip()
                except Exception:
                    continue

            # 3) Fallback: scan a small part of body text (bounded)
            try:
                full = self.page.evaluate(
                    "() => (document.body && document.body.innerText) ? document.body.innerText : ''"
                )
                if isinstance(full, str) and full:
                    m = FORM_ID_NUMERIC_RE.search(full[:15000])
                    if m:
                        return m.group(1).strip()
            except Exception:
                pass

            return ""
        except Exception:
            return ""


    def _extract_form_id_from_text(self, txt: str) -> str:
        if not txt:
            return ""
        s = " ".join(txt.split())
        for rx in self.FORM_ID_REGEXES:
            m = rx.search(s)
            if not m:
                continue
            if m.lastindex and m.lastindex >= 2:
                # "Form ID: <id>" form
                return (m.group(2) or "").strip()
            return (m.group(1) or "").strip()
        return ""

    # -------------------------------------------------------------------------
    # Field filling primitives (DOM-first)
    # -------------------------------------------------------------------------
    def _fill_form_field(self, container: Any, question_key: str, answer: str) -> Dict[str, Any]:
        try:
            ans = (answer or "").strip()

            text_res = self._try_fill_text(container, ans)
            if text_res:
                self._post_fill_scroll()
                return text_res

            cb_res = self._try_fill_checkbox(container, ans)
            if cb_res:
                self._post_fill_scroll()
                return cb_res

            radio_res = self._try_fill_radio(container, ans)
            if radio_res:
                self._post_fill_scroll()
                return radio_res

            scale_res = self._try_fill_scale(container, ans)
            if scale_res:
                self._post_fill_scroll()
                return scale_res

            dd_res = self._try_fill_dropdown(container, ans)
            if dd_res:
                self._post_fill_scroll()
                return dd_res

            if self.claude:
                vision_res = self._fill_field_by_vision(container, question_key, ans)
                if vision_res and vision_res.get("status") == "filled":
                    self._post_fill_scroll()
                    return vision_res

            return {"status": "failed", "answer": ans, "reason": "unknown_field_type"}

        except Exception as e:
            return {"status": "error", "answer": answer, "error": str(e)}

    def _try_fill_text(self, container: Any, ans: str) -> Optional[Dict[str, Any]]:
        try:
            inputs = container.locator('input[type="text"], input:not([type]), textarea').all()
            for inp in inputs:
                try:
                    if not inp.is_visible():
                        continue
                    inp.click()
                    time.sleep(0.08)
                    inp.fill("")
                    inp.fill(ans)
                    time.sleep(0.08)

                    val = (inp.input_value() or "").strip()
                    if val == ans:
                        return {"status": "filled", "type": "text", "answer": ans, "verified": True}
                    return {"status": "filled", "type": "text", "answer": ans, "verified": False, "observed": val}
                except Exception:
                    continue

            tb = container.locator('div[role="textbox"]').first
            if tb.count() > 0 and tb.is_visible():
                try:
                    tb.click()
                    time.sleep(0.08)

                    try:
                        tb.fill(ans)
                    except Exception:
                        tb.press("ControlOrMeta+A")
                        tb.type(ans, delay=5)

                    time.sleep(0.08)

                    observed = (tb.inner_text() or "").strip()
                    if observed == ans or ans in observed:
                        return {
                            "status": "filled",
                            "type": "textbox",
                            "answer": ans,
                            "verified": True,
                            "observed": observed,
                        }
                    return {
                        "status": "filled",
                        "type": "textbox",
                        "answer": ans,
                        "verified": False,
                        "observed": observed,
                    }
                except Exception:
                    pass

            return None
        except Exception:
            return None

    def _try_fill_checkbox(self, container: Any, ans: str) -> Optional[Dict[str, Any]]:
        try:
            checkboxes = container.locator('div[role="checkbox"]').all()
            if not checkboxes:
                return None

            target = ans.lower()
            best = None

            for cb in checkboxes:
                try:
                    if not cb.is_visible():
                        continue
                    aria_label = (cb.get_attribute("aria-label") or "").strip()
                    try:
                        parent_text = (cb.locator("xpath=..").first.inner_text() or "").strip()
                    except Exception:
                        parent_text = ""

                    hay = (aria_label + "\n" + parent_text).lower()
                    if target and target in hay:
                        best = cb
                        break
                except Exception:
                    continue

            if not best:
                return {"status": "failed", "type": "checkbox", "answer": ans, "reason": "no_option_match"}

            best.click()
            time.sleep(0.2)
            checked = best.get_attribute("aria-checked")
            verified = (checked == "true")
            return {"status": "filled", "type": "checkbox", "answer": ans, "verified": verified}
        except Exception:
            return None

    def _try_fill_radio(self, container: Any, ans: str) -> Optional[Dict[str, Any]]:
        try:
            radios = container.locator('div[role="radio"]').all()
            if not radios:
                return None

            target = ans.lower()
            best = None

            for r in radios:
                try:
                    if not r.is_visible():
                        continue
                    aria_label = (r.get_attribute("aria-label") or "").strip()
                    text = (r.inner_text() or "").strip()
                    hay = (aria_label + "\n" + text).lower()
                    if target and target in hay:
                        best = r
                        break
                except Exception:
                    continue

            if not best:
                return {"status": "failed", "type": "radio", "answer": ans, "reason": "no_option_match"}

            best.click()
            time.sleep(0.2)
            checked = best.get_attribute("aria-checked")
            verified = (checked == "true")
            return {"status": "filled", "type": "radio", "answer": ans, "verified": verified}
        except Exception:
            return None

    def _try_fill_scale(self, container: Any, ans: str) -> Optional[Dict[str, Any]]:
        try:
            if not ans:
                return None
            loc = container.locator(f'div[role="radio"][aria-label="{ans}"]').first
            if loc.count() <= 0 or not loc.is_visible():
                return None
            loc.click()
            time.sleep(0.2)
            checked = loc.get_attribute("aria-checked")
            verified = (checked == "true")
            return {"status": "filled", "type": "scale", "answer": ans, "verified": verified}
        except Exception:
            return None

    def _try_fill_dropdown(self, container: Any, ans: str) -> Optional[Dict[str, Any]]:
        try:
            dropdowns = container.locator('select, div[role="listbox"]').all()
            if not dropdowns:
                return None

            dd = None
            for d in dropdowns:
                try:
                    if d.is_visible():
                        dd = d
                        break
                except Exception:
                    continue

            if not dd:
                return None

            dd.click()
            time.sleep(0.25)

            opt = self.page.locator(f'div[role="option"]:has-text("{ans}")').first
            if opt.count() <= 0:
                return {"status": "failed", "type": "dropdown", "answer": ans, "reason": "no_option_match"}

            opt.click()
            time.sleep(0.2)
            return {"status": "filled", "type": "dropdown", "answer": ans, "verified": False}
        except Exception:
            return None

    def _post_fill_scroll(self) -> None:
        try:
            self.page.evaluate(f"window.scrollBy(0, {self.scroll_between_questions_px})")
            time.sleep(0.15)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Required validation detection (DOM)
    # -------------------------------------------------------------------------
    def _get_required_error_questions(self) -> List[str]:
        out: List[str] = []
        try:
            err = self.page.locator(f'text="{self.REQUIRED_ERR_TEXT}"')
            n = err.count()
            if n <= 0:
                return out

            for i in range(min(n, 25)):
                e = err.nth(i)
                block = e.locator("xpath=ancestor::div[@role='listitem']").first
                if block.count() <= 0:
                    continue

                title = ""
                try:
                    title_loc = block.locator("div[role='heading'], span[role='heading']").first
                    if title_loc.count() > 0:
                        title = (title_loc.inner_text() or "").strip()
                except Exception:
                    title = ""

                if not title:
                    try:
                        txt = (block.inner_text() or "").strip()
                        title = (txt.split("\n")[0] or "").strip()[:160]
                    except Exception:
                        title = ""

                if title:
                    out.append(title)

            seen = set()
            uniq = []
            for t in out:
                k = t.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(t)
            return uniq
        except Exception:
            return out

    def _map_missing_titles_to_form_keys(self, missing_titles: List[str], form_data: Dict[str, Any]) -> List[str]:
        keys = list(form_data.keys())
        out: List[str] = []
        for title in missing_titles:
            t = (title or "").strip().lower()
            if not t:
                continue
            for k in keys:
                kk = (k or "").strip().lower()
                if not kk:
                    continue
                if kk in t or t in kk:
                    out.append(k)
                    break

        seen = set()
        uniq = []
        for k in out:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        return uniq

    # -------------------------------------------------------------------------
    # Submission success detection
    # -------------------------------------------------------------------------
    def _detect_dom_submission_success(self) -> bool:
        try:
            for txt in self.SUCCESS_TEXT_CANDIDATES:
                loc = self.page.locator(f'text="{txt}"')
                if loc.count() > 0:
                    return True
            return False
        except Exception:
            return False

    def _click_submit(self, submit_button_locator: Union[str, Dict[str, Any], None]) -> bool:
        try:
            if submit_button_locator:
                loc, matched, count = self._resolve_locator(submit_button_locator, strategy="first_match")
                if count <= 0:
                    return False
                loc.click()
                return True

            submit_selectors = [
                "div[role='button']:has-text('Submit')",
                "span:has-text('Submit')",
                "button:has-text('Submit')",
                "input[type='submit']",
            ]

            for sel in submit_selectors:
                try:
                    loc = self.page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible():
                        loc.first.click()
                        return True
                except Exception:
                    continue

            return False
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Vision helpers (optional)
    # -------------------------------------------------------------------------
    def verify_by_vision(self, expected_state: str) -> Dict[str, Any]:
        t0 = time.time()
        if not self.claude:
            return self._fail(ERROR_VISION, "Vision not available", {}, t0)

        try:
            screenshot_b64 = self._take_screenshot_base64()
            prompt = f"""Verify if the screenshot matches:

Expected: {expected_state}

Return JSON only:
{{
  "matches": true/false,
  "confidence": 0.0-1.0,
  "observed_state": "string"
}}"""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=350,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            text = "".join([c.text for c in response.content if hasattr(c, "text")])
            result = _extract_first_json(text) or {}
            if result.get("matches") and result.get("confidence", 0) > 0.7:
                return self._ok("Vision verification successful", {"verification": result}, t0)
            return self._fail(ERROR_ASSERT, "Vision verification failed", {"verification": result}, t0)

        except Exception as e:
            return self._fail(ERROR_VISION, f"Vision verification error: {e}", {}, t0)

    def _validate_form_submission_by_vision(self) -> Dict[str, Any]:
        if not self.claude:
            return {"validated": False, "reason": "vision_not_available"}

        try:
            screenshot_b64 = self._take_screenshot_base64()
            prompt = """Look at this screenshot and determine if a Google Form was successfully submitted.

Common success indicators:
- "Your response has been recorded"
- "Thank you" message
- Confirmation page
- "Submit another response"

Also detect if there is a required-field validation error.

Return JSON only:
{
  "submitted": true/false,
  "has_required_error": true/false,
  "confidence": 0.0-1.0,
  "observations": "what you see"
}"""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=350,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            text = "".join([c.text for c in response.content if hasattr(c, "text")])
            result = _extract_first_json(text) or {}
            return {
                "validated": True,
                "submitted": bool(result.get("submitted", False)),
                "has_required_error": bool(result.get("has_required_error", False)),
                "confidence": float(result.get("confidence", 0) or 0),
                "observations": result.get("observations", ""),
            }
        except Exception as e:
            return {"validated": False, "error": str(e)}

    def _fill_field_by_vision(self, container: Any, question_key: str, answer: str) -> Dict[str, Any]:
        if not self.claude:
            return {"status": "failed", "answer": answer, "reason": "vision_not_available"}
        try:
            container.scroll_into_view_if_needed()
            time.sleep(0.25)

            screenshot_b64 = self._take_screenshot_base64()
            prompt = f"""Look at this Google Form question and help identify how to select the answer.

Question: {question_key}
Answer to select: {answer}

Return JSON only:
{{
  "field_type": "checkbox/radio/dropdown/text",
  "selector": "CSS selector for the element to click or type into",
  "action": "click/type",
  "confidence": 0.0-1.0
}}"""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = "".join([c.text for c in response.content if hasattr(c, "text")])
            result = _extract_first_json(text) or {}
            if float(result.get("confidence", 0) or 0) < 0.6:
                return {"status": "failed", "answer": answer, "reason": "low_vision_confidence", "vision": result}

            selector = (result.get("selector") or "").strip()
            action = (result.get("action") or "click").strip().lower()

            if not selector:
                return {"status": "failed", "answer": answer, "reason": "vision_no_selector", "vision": result}

            elem = self.page.locator(selector).first
            if elem.count() <= 0:
                return {"status": "failed", "answer": answer, "reason": "vision_selector_not_found", "vision": result}

            if action == "type":
                try:
                    elem.click()
                    time.sleep(0.08)
                    try:
                        elem.fill(answer)
                    except Exception:
                        elem.press("ControlOrMeta+A")
                        elem.type(answer, delay=5)
                except Exception as e:
                    return {"status": "failed", "answer": answer, "reason": f"vision_type_failed: {e}", "vision": result}
            else:
                try:
                    elem.click()
                except Exception as e:
                    return {"status": "failed", "answer": answer, "reason": f"vision_click_failed: {e}", "vision": result}

            time.sleep(0.15)
            return {"status": "filled", "type": f"vision_{result.get('field_type','unknown')}", "answer": answer, "verified": False, "vision": result}
        except Exception as e:
            return {"status": "failed", "answer": answer, "error": str(e)}

    # -------------------------------------------------------------------------
    # Question discovery + matching
    # -------------------------------------------------------------------------
    def _get_all_form_questions(self) -> List[Any]:
        try:
            selectors_to_try = [
                '[role="listitem"]',
                ".freebirdFormviewerComponentsQuestionBaseRoot",
                '[data-params*="question"]',
            ]
            for selector in selectors_to_try:
                try:
                    locs = self.page.locator(selector).all()
                    if locs:
                        return locs
                except Exception:
                    continue
            return []
        except Exception:
            return []

    def _find_question_container(self, question_text: str, all_questions: List[Any]) -> Optional[Any]:
        try:
            q = (question_text or "").strip().lower()
            if not q:
                return None

            for container in all_questions:
                try:
                    txt = (container.inner_text() or "").lower()
                    if q in txt:
                        return container
                except Exception:
                    continue

            selectors = [
                f"//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{q}')]",
                f"//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{q}')]",
                f"//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{q}')]",
            ]
            for sel in selectors:
                try:
                    loc = self.page.locator(sel).first
                    if loc.count() > 0:
                        parent = loc.locator("xpath=ancestor::div[@role='listitem']").first
                        if parent.count() > 0:
                            return parent
                except Exception:
                    continue

            return None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Optional: extract text from page
    # -------------------------------------------------------------------------
    def extract_text_from_page(self, locator: Union[str, Dict[str, Any]] = None, description: str = None) -> Dict[str, Any]:
        t0 = time.time()
        try:
            if locator:
                loc, matched, count = self._resolve_locator(locator, strategy="first_match")
                if count > 0:
                    try:
                        txt = (loc.inner_text() or "").strip()
                    except Exception:
                        txt = (loc.text_content() or "").strip()
                    return self._ok("extract_text ok", {"text": txt}, t0, matched_locator=matched, matches=count)

            if description and self.claude:
                screenshot_b64 = self._take_screenshot_base64()
                prompt = f"""Find and extract the following text from this screenshot:

{description}

Return JSON only:
{{
  "text": "the extracted text",
  "found": true/false,
  "confidence": 0.0-1.0
}}"""
                response = self.claude.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                text = "".join([c.text for c in response.content if hasattr(c, "text")])
                result = _extract_first_json(text) or {}
                if result.get("found") and float(result.get("confidence", 0) or 0) >= 0.6:
                    return self._ok("extract_text vision ok", {"text": result.get("text", "")}, t0)
                return self._fail(ERROR_EXTRACT, "vision extract_text failed", {"raw": text[:500]}, t0)

            return self._fail(ERROR_EXTRACT, "Could not extract text", {"locator": locator, "description": description}, t0)

        except Exception as e:
            return self._fail(ERROR_EXTRACT, f"extract_text error: {e}", {}, t0)

    # -------------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------------
    def assert_all(self, assertions: list, ctx: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            for a in assertions or []:
                atype = a.get("type")

                if atype == "dom.exists":
                    sel = self._to_selector(a["locator"])
                    if self.page.locator(sel).count() <= 0:
                        return self._fail(ERROR_ASSERT, f"dom.exists failed: {sel}", {}, t0, matched_locator=sel, matches=0)

                elif atype == "browser.url_changed":
                    before = ctx.get("_evidence", {}).get("prev_url", "")
                    if before and self.page.url == before:
                        return self._fail(ERROR_ASSERT, f"url did not change (still {self.page.url})", {"url": self.page.url}, t0)

                elif atype == "field.nonempty":
                    path = a.get("field") or ""
                    val = _deep_get(ctx.get("vars", {}), path)
                    if not (val and str(val).strip()):
                        return self._fail(ERROR_ASSERT, f"field.nonempty failed: {path}", {"field": path, "value": val}, t0)

                else:
                    return self._fail(ERROR_UNKNOWN, f"unknown assertion type: {atype}", {}, t0)

            return self._ok("assertions ok", {}, t0)
        except Exception as e:
            return self._fail(ERROR_UNKNOWN, f"assertions error: {e}", {}, t0)

    # -------------------------------------------------------------------------
    # Locator handling
    # -------------------------------------------------------------------------
    def _to_selector(self, locator: Union[str, Dict[str, Any]]) -> str:
        if isinstance(locator, str):
            s = locator.strip()
            if s.startswith(("css=", "xpath=", "text=", "role=", "id=", "label=")):
                return s
            if ">>" in s:
                return s
            return f"css={s}"

        if isinstance(locator, dict):
            strategy = (locator.get("strategy") or "").strip().lower()
            value = (locator.get("value") or "").strip()

            if strategy in {"", "css"}:
                return f"css={value}"
            if strategy == "xpath":
                return f"xpath={value}"
            if strategy == "text":
                return f"text={value}"
            if strategy == "id":
                return f"css=#{value}"
            if strategy == "label":
                return f"label={value}"
            if strategy == "role":
                role = (locator.get("role") or value).strip()
                name = (locator.get("name") or "").strip()
                return f"role={role}[name='{name}']" if name else f"role={role}"

            return f"css={value}"

        raise ValueError(f"Invalid locator: {locator}")

    def _resolve_locator(self, locator: Union[str, Dict[str, Any]], strategy: str) -> Tuple[Any, str, int]:
        sel = self._to_selector(locator)
        loc = self.page.locator(sel)
        count = loc.count()
        if count <= 0:
            return loc, sel, 0

        if strategy == "first_match":
            return loc.first, sel, count
        if strategy == "strict":
            if count != 1:
                raise PWError(f"strict expected 1 match, got {count}: {sel}")
            return loc, sel, count
        return loc.first, sel, count

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _wait_for_url_change_or_timeout(self, before_url: str, timeout_ms: int) -> None:
        deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < deadline:
            if self.page.url != before_url:
                return
            time.sleep(0.05)
        raise PWTimeoutError(f"URL did not change within {timeout_ms}ms (still {self.page.url})")

    def _maybe_mount_lazy_sections(self) -> None:
        if not self.lazy_mount_scroll:
            return
        try:
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.35)
            self.page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.25)
        except Exception:
            pass

    def _maybe_close_popups(self) -> None:
        if not self.close_popups:
            return

        candidates = [
            "button:has-text('Accept')",
            "button:has-text('Accept All')",
            "button:has-text('Accept Cookies')",
            "button:has-text('Close')",
            "button:has-text('Dismiss')",
            "button:has-text('No Thanks')",
            "button:has-text('Not Now')",
            "[aria-label='close']",
            "[aria-label='Close']",
            "[aria-label='Dismiss']",
            ".modal-close",
            ".popup-close",
            ".close-button",
            "button.close",
            "#onetrust-accept-btn-handler",
            ".cookie-banner button:has-text('Accept')",
        ]

        for sel in candidates:
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.click(timeout=1000)
                    time.sleep(0.25)
                    return
            except Exception:
                continue

        if self.claude:
            self._close_popup_by_vision()

    def _close_popup_by_vision(self) -> bool:
        try:
            screenshot_b64 = self._take_screenshot_base64()
            prompt = """If there is a blocking popup/cookie banner, return JSON only:
{
  "has_popup": true/false,
  "close_selector": "css selector or null",
  "confidence": 0.0-1.0
}"""
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
            text = "".join([c.text for c in response.content if hasattr(c, "text")])
            result = _extract_first_json(text) or {}
            if result.get("has_popup") and float(result.get("confidence", 0) or 0) > 0.6:
                close_selector = result.get("close_selector")
                if close_selector:
                    self.page.locator(close_selector).first.click(timeout=2000)
                    time.sleep(0.25)
                    return True
            return False
        except Exception:
            return False

    def _screenshot(self) -> str:
        fname = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(self.artifacts_dir, fname)
        try:
            if self._page:
                self.page.screenshot(path=path, full_page=True)
                return path
        except Exception:
            pass
        return ""

    def _take_screenshot_base64(self) -> str:
        try:
            from PIL import Image
            import io

            screenshot_bytes = self.page.screenshot(full_page=True)
            img = Image.open(io.BytesIO(screenshot_bytes))
            width, height = img.size

            max_dimension = 7500
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                screenshot_bytes = buffer.getvalue()

            return base64.b64encode(screenshot_bytes).decode()
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""

    def _ok(self, message: str, extracted: Dict[str, Any], t0: float, matched_locator: str = "", matches: int = 0) -> Dict[str, Any]:
        return {
            "ok": True,
            "error_type": None,
            "message": message,
            "evidence": {
                "url": self.page.url if self._page else "",
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
            if self._page:
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


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set. Vision will be disabled.")
        api_key = None

    agent = GoogleFormsBrowserAgent(
        headless=_env_bool("BROWSER_HEADLESS", False),
        artifacts_dir=os.getenv("BROWSER_ARTIFACTS_DIR", "artifacts"),
        anthropic_api_key=api_key,
        use_vision=_env_bool("BROWSER_USE_VISION", True),
        stealth_mode=_env_bool("BROWSER_STEALTH_MODE", True),
        use_chrome_channel=_env_bool("BROWSER_USE_CHROME_CHANNEL", True),
        fill_passes=_env_int("FORM_FILL_PASSES", 2),
        lazy_mount_scroll=_env_bool("FORM_LAZY_MOUNT_SCROLL", True),
    )

    try:
        steps = [
            {
                "id": "navigate",
                "action": "goto",
                "args": {"url": "https://forms.gle/j9FWqyBRidYCCoCRA"},
                "retries": 1,
            },
            {
                "id": "fill",
                "action": "fill_google_form",
                "args": {
                    "passes": 2,
                    "form_data": {
                        "Full Name": "John Doe",
                        "Already filled": "No",
                        "Location": "LA",
                        "preferred method of communication": "Email",
                        "enjoy Mondays": "7",
                        "most productive": "Morning",
                        "I certify all the details filled are somewhat correct. Please enter your full name": "John Doe",
                    },
                },
                "retries": 0,
            },
            {
                "id": "submit",
                "action": "submit_form",
                "args": {
                    "require_no_missing_required": True,
                    "return_filled_details": True,
                    "return_form_id": True,
                },
                "retries": 2,
            },
        ]

        ctx: Dict[str, Any] = {"vars": {}}

        for step in steps:
            res = agent.run_step(step, ctx)
            print(json.dumps(res, indent=2))
            if not res.get("ok"):
                break

        input("\nPress Enter to close browser...")

    finally:
        agent.close()
