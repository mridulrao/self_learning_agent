from __future__ import annotations

import os
import time
import uuid
import base64
import logging
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError, Error as PWError
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


logger = setup_logger("PlaywrightBrowserAgent", level=logging.INFO)

# -----------------------------
# Error types
# -----------------------------
ERROR_TIMEOUT = "timeout"
ERROR_NOT_FOUND = "not_found"
ERROR_NAVIGATION = "navigation_error"
ERROR_CLICK = "click_error"
ERROR_TYPE = "type_error"
ERROR_EXTRACT = "extract_error"
ERROR_ASSERT = "assertion_failed"
ERROR_UNKNOWN = "unknown_error"
ERROR_VISION = "vision_error"


@dataclass
class Backoff:
    max_retries: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 2.0

    def sleep(self, attempt: int) -> None:
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        jitter = (attempt % 3) * 0.03
        time.sleep(delay + jitter)


# -----------------------------
# JSON extraction helpers (robust against Claude chatter/fences)
# -----------------------------
def _extract_first_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extract and parse the first JSON object/array found in arbitrary text.

    Handles:
    - leading explanations
    - fenced blocks
    - partially fenced blocks (best-effort)
    """
    if not text:
        return None

    # Strip code fences markers but keep content
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Find first '{' or '[' and attempt balanced parsing
    start_positions = []
    for ch in ["{", "["]:
        i = cleaned.find(ch)
        if i != -1:
            start_positions.append(i)
    if not start_positions:
        return None

    start = min(start_positions)
    s = cleaned[start:]

    # Balanced scan
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
                        # continue scanning; maybe earlier close created invalid json
                        pass
    return None


class PlaywrightBrowserAgent:
    """
    Browser agent with:
    - bot-friendlier search
    - popup closing
    - HTML/DOM extraction (preferred)
    - optional Claude vision fallback
    """

    def __init__(
        self,
        headless: bool = True,
        viewport: Tuple[int, int] = (1280, 800),
        artifacts_dir: str = "artifacts",
        default_timeout_ms: int = 15_000,
        implicit_wait_ms: int = 5_000,
        backoff: Backoff = Backoff(),
        close_popups: bool = True,
        anthropic_api_key: Optional[str] = None,
        use_vision: bool = True,
        search_engine: str = "duckduckgo",
        stealth_mode: bool = True,
        use_chrome_channel: bool = True,
    ):
        self.headless = headless
        self.viewport = viewport
        self.artifacts_dir = artifacts_dir
        self.default_timeout_ms = default_timeout_ms
        self.implicit_wait_ms = implicit_wait_ms
        self.backoff = backoff
        self.close_popups = close_popups
        self.use_vision = use_vision
        self.search_engine = search_engine
        self.stealth_mode = stealth_mode
        self.use_chrome_channel = use_chrome_channel

        if anthropic_api_key and use_vision:
            logger.info("Initializing Claude Vision for browser agent")
            self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            logger.info("Claude Vision disabled for browser agent")
            self.claude = None

        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

        os.makedirs(self.artifacts_dir, exist_ok=True)

    # -----------------------------
    # Lifecycle
    # -----------------------------
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
                    logger.info("✓ Launched using installed Chrome browser (channel=chrome)")
                except Exception as e:
                    logger.warning(f"Chrome channel failed ({e}), falling back to chromium")
                    launch_options.pop("channel", None)
                    self._browser = self._pw.chromium.launch(**launch_options)
            else:
                self._browser = self._pw.chromium.launch(**launch_options)

            context_options = {
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
                    """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    """
                )

            self._page = self._context.new_page()
            self._page.set_default_timeout(self.default_timeout_ms)
            self._page.set_default_navigation_timeout(self.default_timeout_ms)

            logger.info("✓ Browser launched successfully")
            return self._ok("launched", {}, t0)
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            return self._fail(ERROR_UNKNOWN, f"launch failed: {e}", {}, t0)

    def close(self) -> None:
        logger.info("Closing browser...")
        try:
            if self._context:
                self._context.close()
        finally:
            try:
                if self._browser:
                    self._browser.close()
            finally:
                if self._pw:
                    self._pw.stop()
        logger.info("✓ Browser closed")

    @property
    def page(self):
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page

    # -----------------------------
    # Orchestrator-facing entrypoint
    # -----------------------------
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
            if action == "search":
                return self.search(args["query"], engine=args.get("engine", self.search_engine))
            if action == "click_first_result":
                return self.click_first_quality_result(
                    result_type=args.get("result_type", "any"),
                    max_attempts=args.get("max_attempts", 5),
                    skip_domains=args.get("skip_domains"),
                )
            if action == "click":
                locator = args["locator"]
                strategy = args.get("strategy", "first_match")
                return self.click(locator, strategy=strategy)
            if action == "type":
                return self.type(args["locator"], args.get("text", ""))
            if action == "type_and_submit":
                return self.type_and_submit(args["locator"], args.get("text", ""))
            if action == "wait_for":
                if "locator" in args:
                    return self.wait_for(args["locator"])
                if args.get("url_change"):
                    return self.wait_for_url_change(ctx["_evidence"]["prev_url"])
                return self._fail(ERROR_UNKNOWN, "wait_for requires args.locator or args.url_change=true", {}, t0)
            if action == "scroll":
                return self.scroll(
                    direction=args.get("direction", "down"),
                    amount=args.get("amount", 300)
                )
            if action == "extract":
                return self.extract(args.get("fields", {}))

            # Vision
            if action == "click_by_vision":
                return self.click_by_vision(args["element_description"], strategy=args.get("strategy", "single"))
            if action == "extract_by_vision":
                return self.extract_by_vision(args.get("fields", {}), args.get("prompt"))
            if action == "verify_by_vision":
                return self.verify_by_vision(args["expected_state"])

            # Generic content extraction (replaces restaurant-specific functions)
            if action == "extract_page_content_from_results":
                return self.extract_page_content_from_results(
                    max_items=args.get("max_items", 5),
                    wait_between=args.get("wait_between", 1.0),
                )
            if action == "extract_page_content":
                return self.extract_page_content(
                    max_items=args.get("max_items", 5),
                )

            return self._fail(ERROR_UNKNOWN, f"unknown browser action: {action}", {}, t0)

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

    # -----------------------------
    # Search Actions
    # -----------------------------
    def search(self, query: str, engine: str = "duckduckgo") -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Searching {engine} for: '{query}'")

            search_configs = {
                "duckduckgo": {"url": "https://duckduckgo.com", "input_selector": "input[name='q']"},
                "bing": {"url": "https://www.bing.com", "input_selector": "input[name='q']"},
                "brave": {"url": "https://search.brave.com", "input_selector": "input[name='q']"},
            }

            config = search_configs.get(engine.lower())
            if not config:
                return self._fail(ERROR_UNKNOWN, f"Unknown search engine: {engine}", {}, t0)

            before_url = self.page.url
            self.page.goto(config["url"], wait_until="domcontentloaded")
            logger.info(f"Loaded {engine}: {self.page.url}")

            self.page.wait_for_selector(config["input_selector"], state="attached", timeout=self.implicit_wait_ms)

            search_box = self.page.locator(config["input_selector"]).first
            search_box.click()
            search_box.fill(query)
            logger.info(f"Typed query: {query}")
            search_box.press("Enter")

            self._wait_for_url_change_or_timeout(config["url"], self.default_timeout_ms)
            time.sleep(1.0)

            self._maybe_close_popups()

            logger.info(f"✓ Search completed: {self.page.url}")
            return self._ok(
                f"Search completed on {engine}",
                {"query": query, "engine": engine, "previous_url": before_url, "results_url": self.page.url},
                t0,
            )

        except PWTimeoutError as e:
            logger.error(f"Search timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"search timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Search error: {e}")
            return self._fail(ERROR_NAVIGATION, f"search error: {e}", {}, t0)
        except Exception as e:
            logger.exception(f"Search exception: {e}")
            return self._fail(ERROR_UNKNOWN, f"search exception: {e}", {}, t0)

    def _detect_bot_check(self) -> bool:
        """
        Detect if we've hit a bot check page (Cloudflare, reCAPTCHA, etc.)
        Returns True if bot check detected
        """
        bot_indicators = [
            # Cloudflare
            "checking your browser",
            "cloudflare",
            "cf-browser-verification",
            "Just a moment",
            "Enable JavaScript and cookies",
            # reCAPTCHA
            "recaptcha",
            "g-recaptcha",
            "I'm not a robot",
            # Generic bot checks
            "verify you are human",
            "prove you're not a robot",
            "security check",
            "automated access",
            "suspicious activity",
            # Specific sites
            "access denied",
            "blocked",
            "too many requests",
        ]
        
        try:
            # Check page title
            title = self.page.title().lower()
            for indicator in bot_indicators:
                if indicator.lower() in title:
                    logger.warning(f"Bot check detected in title: '{title}'")
                    return True
            
            # Check page content
            body_text = self.page.locator("body").inner_text().lower()
            for indicator in bot_indicators:
                if indicator.lower() in body_text:
                    logger.warning(f"Bot check detected in body: indicator='{indicator}'")
                    return True
            
            # Check for common bot check elements
            bot_selectors = [
                "#challenge-form",  # Cloudflare
                ".g-recaptcha",  # reCAPTCHA
                "[data-callback='onRecaptchaSuccess']",
                "iframe[src*='recaptcha']",
                "#cf-wrapper",
                ".cf-browser-verification",
            ]
            
            for selector in bot_selectors:
                if self.page.locator(selector).count() > 0:
                    logger.warning(f"Bot check element found: {selector}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error detecting bot check: {e}")
            return False

    def _is_sponsored_or_low_quality_result(self, url: str, title: str = "") -> bool:
        """
        Detect if a search result is sponsored or from a low-quality aggregator site
        """
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Sponsored indicators in URL
        sponsored_params = [
            "ad=",
            "sponsored",
            "gclid=",
            "msclkid=",
            "fbclid=",
            "utm_source=",
        ]
        
        for param in sponsored_params:
            if param in url_lower:
                logger.info(f"Skipping sponsored result (param: {param}): {url[:80]}")
                return True
        
        # Low-quality aggregator sites
        low_quality_domains = [
            "yelp.com",
            "groupon.com",
            "tripadvisor.com",
            "yellowpages.com",
            "manta.com",
            "superpages.com",
            "mapquest.com",
            "foursquare.com",
            # Add more as needed
        ]
        
        for domain in low_quality_domains:
            if domain in url_lower:
                logger.info(f"Skipping aggregator site: {domain} ({url[:80]})")
                return True
        
        # Sponsored indicators in title
        sponsored_keywords = ["ad", "sponsored", "promoted"]
        for keyword in sponsored_keywords:
            if keyword in title_lower and len(title_lower.split()) < 10:
                logger.info(f"Skipping result with sponsored keyword in title: {title}")
                return True
        
        return False

    def click_first_quality_result(
        self, 
        result_type: str = "any", 
        max_attempts: int = 5,
        skip_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Click the first quality (non-sponsored, non-bot-checked) search result.
        Will try multiple results if encountering bot checks.
        
        Args:
            result_type: Type of result to look for
            max_attempts: Maximum number of results to try before giving up
            skip_domains: Additional domains to skip
        """
        t0 = time.time()
        skip_domains = skip_domains or ["yelp.com", "groupon.com"]
        
        try:
            logger.info(f"Looking for first quality search result (max_attempts={max_attempts})")
            self._maybe_close_popups()
            time.sleep(0.5)

            selectors = [
                'article[data-testid="result"] h2 a',
                'div[data-testid="result"] h2 a',
                'article h2 a',
                'a[data-testid="result-title-a"]',
                ".result__a",
                "h2 > a",
                "article a[href]",
            ]

            # Collect all potential results
            all_results = []
            for selector in selectors:
                try:
                    loc = self.page.locator(selector)
                    count = loc.count()
                    for i in range(min(count, max_attempts * 2)):  # Get more than we need
                        try:
                            elem = loc.nth(i)
                            href = elem.get_attribute("href") or ""
                            title = (elem.text_content() or "").strip()
                            
                            if not href or not title:
                                continue
                                
                            all_results.append({
                                "element": elem,
                                "href": href,
                                "title": title,
                                "index": i,
                                "selector": selector
                            })
                        except Exception:
                            continue
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not all_results:
                logger.warning("No results found with selectors, trying vision...")
                return self.click_by_vision(
                    "the first organic (non-sponsored) search result link"
                )
            
            logger.info(f"Found {len(all_results)} total results, filtering...")
            
            # Try results one by one
            attempts = 0
            for result in all_results:
                if attempts >= max_attempts:
                    break
                
                href = result["href"]
                title = result["title"]
                
                # Skip sponsored/low-quality results
                if self._is_sponsored_or_low_quality_result(href, title):
                    continue
                
                # Skip explicitly blocked domains
                if any(domain in href.lower() for domain in skip_domains):
                    logger.info(f"Skipping blocked domain: {href[:80]}")
                    continue
                
                attempts += 1
                logger.info(f"Attempt {attempts}/{max_attempts}: Trying result '{title[:60]}'")
                logger.info(f"URL: {href[:120]}")
                
                try:
                    before_url = self.page.url
                    result["element"].click(timeout=5000)
                    
                    # Wait for navigation
                    self._wait_for_url_change_or_timeout(before_url, self.default_timeout_ms)
                    time.sleep(1.5)
                    
                    # Check if we hit a bot check
                    if self._detect_bot_check():
                        logger.warning(f"Bot check detected on {self.page.url}")
                        logger.info("Going back to search results...")
                        self.page.go_back(wait_until="domcontentloaded")
                        time.sleep(1.0)
                        continue
                    
                    # Success!
                    self._maybe_close_popups()
                    logger.info(f"✓ Successfully navigated to quality result: {self.page.url}")
                    
                    return self._ok(
                        "Clicked first quality search result",
                        {
                            "previous_url": before_url,
                            "result_url": self.page.url,
                            "result_link": href,
                            "result_title": title,
                            "attempts": attempts,
                        },
                        t0,
                    )
                    
                except PWTimeoutError:
                    logger.warning(f"Timeout clicking result {attempts}, trying next...")
                    try:
                        self.page.go_back(wait_until="domcontentloaded")
                        time.sleep(0.5)
                    except Exception:
                        pass
                    continue
                except Exception as e:
                    logger.warning(f"Error clicking result {attempts}: {e}")
                    try:
                        self.page.go_back(wait_until="domcontentloaded")
                        time.sleep(0.5)
                    except Exception:
                        pass
                    continue
            
            # If we got here, all attempts failed
            return self._fail(
                ERROR_CLICK,
                f"Failed to find quality result after {attempts} attempts (all results were sponsored/bot-checked)",
                {"attempts": attempts, "total_results": len(all_results)},
                t0,
            )

        except Exception as e:
            logger.exception(f"Click quality result error: {e}")
            return self._fail(ERROR_CLICK, f"click quality result error: {e}", {}, t0)

    def click_first_result(self, result_type: str = "any") -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Clicking first search result (type: {result_type})")
            self._maybe_close_popups()
            time.sleep(0.5)

            selectors = [
                'article[data-testid="result"] h2 a',
                'div[data-testid="result"] h2 a',
                'article h2 a',
                'a[data-testid="result-title-a"]',
                ".result__a",
                "h2 > a",
                "article a[href]",
            ]

            for selector in selectors:
                try:
                    loc = self.page.locator(selector)
                    if loc.count() > 0:
                        first_link = loc.first
                        href = first_link.get_attribute("href") or "unknown"
                        title = (first_link.text_content() or "Unknown title").strip()[:120]

                        logger.info(f"Found first result: {title}")
                        logger.info(f"URL: {href[:120]}...")

                        before_url = self.page.url
                        first_link.click()
                        self._wait_for_url_change_or_timeout(before_url, self.default_timeout_ms)

                        time.sleep(1.5)
                        self._maybe_close_popups()

                        logger.info(f"✓ Navigated to: {self.page.url}")
                        return self._ok(
                            "Clicked first search result",
                            {"previous_url": before_url, "result_url": self.page.url, "result_link": href, "result_title": title},
                            t0,
                        )
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue

            logger.warning("Could not find first result with selectors, trying vision...")
            return self.click_by_vision(
                "the first search result link"
            )

        except PWTimeoutError as e:
            logger.error(f"Click first result timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"click first result timeout: {e}", {}, t0)
        except Exception as e:
            logger.exception(f"Click first result error: {e}")
            return self._fail(ERROR_CLICK, f"click first result error: {e}", {}, t0)

    # -----------------------------
    # Standard Browser Actions
    # -----------------------------
    def goto(self, url: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Navigating to: {url}")
            before = self.page.url
            self.page.goto(url, wait_until="domcontentloaded")
            self._maybe_close_popups()
            logger.info(f"✓ Navigation successful: {self.page.url}")
            return self._ok("goto ok", {"previous_url": before, "url": self.page.url}, t0)
        except PWTimeoutError as e:
            logger.error(f"Navigation timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"goto timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Navigation error: {e}")
            return self._fail(ERROR_NAVIGATION, f"goto navigation error: {e}", {}, t0)

    def click(self, locator: Union[str, Dict[str, Any]], strategy: str = "first_match") -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Clicking element: {locator}")
            loc, matched, count = self._resolve_locator(locator, strategy=strategy)
            if count == 0:
                return self._fail(ERROR_NOT_FOUND, "click: locator not found", {}, t0, matched_locator=matched, matches=0)
            loc.click()
            self._maybe_close_popups()
            logger.info(f"✓ Click successful (matched {count} elements)")
            return self._ok("click ok", {}, t0, matched_locator=matched, matches=count)
        except PWTimeoutError as e:
            logger.error(f"Click timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"click timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Click error: {e}")
            return self._fail(ERROR_CLICK, f"click error: {e}", {}, t0)

    def type(self, locator: Union[str, Dict[str, Any]], text: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Typing into element: {locator}")
            loc, matched, count = self._resolve_locator(locator, strategy="first_match")
            if count == 0:
                return self._fail(ERROR_NOT_FOUND, "type: locator not found", {}, t0, matched_locator=matched, matches=0)
            loc.click()
            loc.fill(text)
            logger.info(f"✓ Typed {len(text)} characters")
            return self._ok("type ok", {"typed": True}, t0, matched_locator=matched, matches=count)
        except PWTimeoutError as e:
            logger.error(f"Type timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"type timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Type error: {e}")
            return self._fail(ERROR_TYPE, f"type error: {e}", {}, t0)

    def type_and_submit(self, locator: Union[str, Dict[str, Any]], text: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Typing and submitting: {locator}")
            loc, matched, count = self._resolve_locator(locator, strategy="first_match")
            if count == 0:
                return self._fail(ERROR_NOT_FOUND, "type_and_submit: locator not found", {}, t0, matched_locator=matched, matches=0)
            before = self.page.url
            loc.click()
            loc.fill(text)
            loc.press("Enter")
            self._wait_for_url_change_or_timeout(before, self.default_timeout_ms)
            self._maybe_close_popups()
            logger.info(f"✓ Submitted form, navigated to: {self.page.url}")
            return self._ok("type_and_submit ok", {"previous_url": before, "url": self.page.url}, t0, matched_locator=matched, matches=count)
        except PWTimeoutError as e:
            logger.error(f"Type and submit timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"type_and_submit timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Type and submit error: {e}")
            return self._fail(ERROR_TYPE, f"type_and_submit error: {e}", {}, t0)

    def wait_for(self, locator: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        try:
            sel = self._to_selector(locator)
            logger.info(f"Waiting for element: {sel}")
            self.page.wait_for_selector(sel, state="attached", timeout=self.implicit_wait_ms)
            logger.info(f"✓ Element appeared: {sel}")
            return self._ok("wait_for ok", {"waited_for": sel}, t0, matched_locator=sel, matches=1)
        except PWTimeoutError as e:
            logger.error(f"Wait timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"wait_for timeout: {e}", {}, t0)
        except PWError as e:
            logger.error(f"Wait error: {e}")
            return self._fail(ERROR_UNKNOWN, f"wait_for error: {e}", {}, t0)

    def wait_for_url_change(self, before_url: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            logger.info(f"Waiting for URL to change from: {before_url}")
            self._wait_for_url_change_or_timeout(before_url, self.implicit_wait_ms)
            logger.info(f"✓ URL changed to: {self.page.url}")
            return self._ok("url_changed ok", {"previous_url": before_url, "url": self.page.url}, t0)
        except PWTimeoutError as e:
            logger.error(f"URL change timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"url_change timeout: {e}", {}, t0)

    def extract(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        out: Dict[str, Any] = {}
        try:
            logger.info(f"Extracting {len(fields)} fields")
            for key, spec in (fields or {}).items():
                if isinstance(spec, dict):
                    sel = self._to_selector(spec.get("locator") or spec.get("selector") or "")
                else:
                    sel = self._to_selector(spec)

                out[key] = self._extract_text(sel)
                logger.info(f"Extracted {key}: {str(out[key])[:50]}...")

            logger.info(f"✓ Extracted {len(out)} fields")
            return self._ok("extract ok", out, t0)
        except PWTimeoutError as e:
            logger.error(f"Extract timeout: {e}")
            return self._fail(ERROR_TIMEOUT, f"extract timeout: {e}", out, t0)
        except PWError as e:
            logger.error(f"Extract error: {e}")
            return self._fail(ERROR_EXTRACT, f"extract error: {e}", out, t0)
        except Exception as e:
            logger.error(f"Extract exception: {e}")
            return self._fail(ERROR_EXTRACT, f"extract exception: {e}", out, t0)

    def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        """Scroll the page in the specified direction"""
        t0 = time.time()
        try:
            logger.info(f"Scrolling {direction} by {amount}px")
            
            direction_lower = direction.lower()
            
            if direction_lower == "down":
                self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction_lower == "up":
                self.page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction_lower == "left":
                self.page.evaluate(f"window.scrollBy(-{amount}, 0)")
            elif direction_lower == "right":
                self.page.evaluate(f"window.scrollBy({amount}, 0)")
            elif direction_lower == "top":
                self.page.evaluate("window.scrollTo(0, 0)")
            elif direction_lower == "bottom":
                self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                return self._fail(
                    ERROR_UNKNOWN, 
                    f"Unknown scroll direction: {direction}", 
                    {}, 
                    t0
                )
            
            time.sleep(0.3)  # Brief pause after scroll
            
            logger.info(f"✓ Scrolled {direction} successfully")
            return self._ok(
                f"Scrolled {direction} by {amount}px",
                {"direction": direction, "amount": amount},
                t0
            )
            
        except Exception as e:
            logger.exception(f"Scroll error: {e}")
            return self._fail(ERROR_UNKNOWN, f"Scroll error: {e}", {}, t0)

    # -----------------------------
    # Vision-Based Actions
    # -----------------------------
    def click_by_vision(self, element_description: str, strategy: str = "single") -> Dict[str, Any]:
        t0 = time.time()
        if not self.claude:
            return self._fail(ERROR_VISION, "Vision not available", {}, t0)

        try:
            logger.info(f"Looking for element using vision: '{element_description}'")
            screenshot_b64 = self._take_screenshot_base64()

            prompt = f"""Find the element in the screenshot.

Element: {element_description}

Return JSON only:
{{
  "found": true/false,
  "selector": "css selector",
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
            logger.info(f"Vision result: {result}")

            if not result.get("found") or result.get("confidence", 0) < 0.7:
                return self._fail(ERROR_NOT_FOUND, f"Vision could not find: {element_description}", {"vision_result": result}, t0)

            selector = result.get("selector", "")
            return self.click(selector, strategy="first_match")

        except Exception as e:
            logger.exception(f"Vision click error: {e}")
            return self._fail(ERROR_VISION, f"Vision click error: {e}", {}, t0)

    def extract_by_vision(self, fields: Dict[str, str], prompt: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()
        if not self.claude:
            return self._fail(ERROR_VISION, "Vision not available", {}, t0)

        try:
            screenshot_b64 = self._take_screenshot_base64()
            if not prompt:
                field_list = "\n".join([f"- {k}: {v}" for k, v in fields.items()])
                prompt = f"""Extract the following from the screenshot:

{field_list}

Return JSON only with keys: {list(fields.keys())}.
Use null if missing."""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=900,
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
            if not isinstance(result, dict) or not result:
                return self._fail(ERROR_VISION, "Vision extraction returned no JSON", {"raw": text[:500]}, t0)

            return self._ok("Vision extraction successful", result, t0)

        except Exception as e:
            logger.exception(f"Vision extraction error: {e}")
            return self._fail(ERROR_VISION, f"Vision extraction error: {e}", {}, t0)

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
            return self._fail(ERROR_ASSERT, f"Vision verification failed", {"verification": result}, t0)

        except Exception as e:
            logger.exception(f"Vision verification error: {e}")
            return self._fail(ERROR_VISION, f"Vision verification error: {e}", {}, t0)

    # -----------------------------
    # Assertions
    # -----------------------------
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

    # -----------------------------
    # Locator handling
    # -----------------------------
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

    # -----------------------------
    # Generic Content Extraction (replaces restaurant-specific functions)
    # -----------------------------
    def extract_page_content(self, max_items: int = 5) -> Dict[str, Any]:
        """
        Extract structured content from the current page using DOM parsing.
        Tries to intelligently extract list items, articles, or main content.
        Falls back to Claude vision if DOM extraction yields no results.
        
        Args:
            max_items: Maximum number of items to extract
            
        Returns:
            Dict with extraction results including 'extracted_content' and 'items'
        """
        t0 = time.time()
        try:
            logger.info(f"Extracting up to {max_items} content items from page")
            self._maybe_close_popups()
            time.sleep(0.4)

            # 1) Try DOM-first extraction
            items = self._extract_content_items_dom(max_items=max_items)

            if items:
                logger.info(f"✓ DOM extracted {len(items)} content items")
                
                # Format items as text for downstream use
                text_content = self._format_items_as_text(items)
                
                return self._ok(
                    f"Extracted {len(items)} content items from DOM",
                    {
                        "extracted_content": text_content,
                        "items": items,
                        "total_found": len(items),
                        "source": "dom",
                        "page_url": self.page.url
                    },
                    t0,
                )

            # 2) Try extracting main article/content text
            main_text = self._extract_main_content_text()
            if main_text and len(main_text) > 100:
                logger.info(f"✓ Extracted main content text ({len(main_text)} chars)")
                return self._ok(
                    "Extracted main content text",
                    {
                        "extracted_content": main_text,
                        "items": [],
                        "total_found": 0,
                        "source": "main_content",
                        "page_url": self.page.url
                    },
                    t0,
                )

            # 3) Claude fallback (only if configured)
            if not self.claude:
                return self._fail(
                    ERROR_EXTRACT,
                    "Claude not configured and DOM extraction found no content",
                    {},
                    t0
                )

            logger.info("DOM extraction yielded no results, trying Claude vision...")
            return self._extract_content_with_vision(max_items=max_items, t0=t0)

        except Exception as e:
            logger.exception(f"Content extraction error: {e}")
            return self._fail(ERROR_EXTRACT, f"Content extraction error: {e}", {}, t0)

    def extract_page_content_from_results(
        self, 
        max_items: int = 5, 
        wait_between: float = 1.0
    ) -> Dict[str, Any]:
        """
        Extract structured content from search results using Claude vision.
        Uses vision to identify and extract visible content items.
        
        Args:
            max_items: Maximum number of items to extract
            wait_between: Delay before taking screenshot
            
        Returns:
            Dict with extraction results
        """
        t0 = time.time()
        if not self.claude:
            return self._fail(ERROR_VISION, "Vision not available", {}, t0)

        try:
            logger.info(f"Extracting up to {max_items} content items from results (vision)")
            self._maybe_close_popups()
            time.sleep(wait_between)

            return self._extract_content_with_vision(max_items=max_items, t0=t0)

        except Exception as e:
            logger.exception(f"Vision content extraction error: {e}")
            return self._fail(ERROR_VISION, f"Vision content extraction error: {e}", {}, t0)

    def _extract_content_items_dom(self, max_items: int = 5) -> List[Dict[str, Any]]:
        """
        DOM-based content extraction. Targets common patterns:
        - List items with headings (articles, guides, listicles)
        - Structured content blocks
        - Recipe items, product listings, etc.
        """
        candidates: List[str] = []

        # Selectors for common content patterns
        selectors = [
            # Articles and listicles
            "article h2",
            "article h3",
            "main h2",
            "main h3",
            "[data-testid] h2",
            "[data-testid] h3",
            ".content h2",
            ".content h3",
            # Specific patterns (Eater, food blogs, etc.)
            ".c-mapstack__card h2",
            ".c-mapstack__heading",
            ".c-entry-content h2",
            ".c-entry-content h3",
            # Generic list items
            "li h2",
            "li h3",
            "li strong",
            # Recipe titles
            ".recipe-title",
            ".recipe-name",
            "[itemprop='name']",
        ]

        for sel in selectors:
            try:
                loc = self.page.locator(sel)
                n = min(loc.count(), 60)
                for i in range(n):
                    txt = (loc.nth(i).inner_text() or "").strip()
                    if not txt:
                        continue
                    
                    # Heuristics to skip generic headings
                    bad_keywords = {
                        "map", "related", "more", "where to", "updates", 
                        "sign up", "newsletter", "subscribe", "follow",
                        "advertisement", "sponsored"
                    }
                    low = txt.lower()
                    if any(b in low for b in bad_keywords):
                        continue
                    
                    # Skip very long headings (likely not item titles)
                    if len(txt) > 100:
                        continue
                    
                    # Skip very short headings (likely not meaningful)
                    if len(txt) < 3:
                        continue
                    
                    candidates.append(txt)
            except Exception:
                continue

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c.lower() in seen:
                continue
            seen.add(c.lower())
            unique.append(c)

        # Clean up formatting (remove numbering, trailing punctuation)
        cleaned = []
        for name in unique:
            # Remove leading numbers (e.g., "1. Restaurant Name")
            name2 = re.sub(r"^\d+\.\s*", "", name).strip()
            # Remove trailing pipes and content after them
            name2 = re.sub(r"\s+\|.*$", "", name2).strip()
            # Remove extra whitespace
            name2 = re.sub(r"\s+", " ", name2).strip()
            
            if len(name2) >= 3:
                cleaned.append(name2)

        cleaned = cleaned[:max_items]
        return [{"name": n} for n in cleaned]

    def _extract_main_content_text(self) -> str:
        """
        Extract main text content from the page.
        Tries common content containers in order of preference.
        """
        content_selectors = [
            "article",
            "main",
            "[role='main']",
            ".main-content",
            ".content",
            ".post-content",
            ".entry-content",
            "#content",
            "body"
        ]
        
        for selector in content_selectors:
            try:
                loc = self.page.locator(selector)
                if loc.count() > 0:
                    text = loc.first.inner_text()
                    if text and len(text.strip()) > 100:  # Meaningful content threshold
                        return text.strip()
            except Exception:
                continue
        
        return ""

    def _format_items_as_text(self, items: List[Dict[str, Any]]) -> str:
        """
        Format extracted items as readable text for downstream use.
        """
        if not items:
            return ""
        
        lines = []
        for i, item in enumerate(items, 1):
            name = item.get("name", "")
            if name:
                lines.append(f"{i}. {name}")
        
        return "\n".join(lines)

    def _extract_content_with_vision(
        self, 
        max_items: int = 5, 
        t0: float = None
    ) -> Dict[str, Any]:
        """
        Use Claude vision to extract content from the page.
        This is a fallback when DOM extraction doesn't work.
        """
        if t0 is None:
            t0 = time.time()
        
        try:
            screenshot_b64 = self._take_screenshot_base64()
            
            prompt = f"""Extract up to {max_items} content items visible in this screenshot.
Look for structured content like:
- List items (restaurants, products, articles, etc.)
- Article titles or headings
- Key information blocks

Return JSON only:
{{
  "items": [{{"name": "...", "description": null}}],
  "page_type": "unknown"
}}

If you cannot identify structured items, extract the main text content instead.
"""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1400,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64
                            }
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            text = "".join([c.text for c in response.content if hasattr(c, "text")])
            parsed = _extract_first_json(text) or {}
            
            items = parsed.get("items", []) if isinstance(parsed, dict) else []
            items = items[:max_items]

            if not items:
                return self._fail(
                    ERROR_EXTRACT,
                    "Vision extraction returned no content items",
                    {"raw": text[:800]},
                    t0
                )

            # Format as text
            text_content = self._format_items_as_text(items)

            return self._ok(
                f"Extracted {len(items)} content items (vision)",
                {
                    "extracted_content": text_content,
                    "items": items,
                    "total_found": len(items),
                    "page_type": parsed.get("page_type", "unknown"),
                    "source": "vision",
                    "page_url": self.page.url
                },
                t0,
            )

        except Exception as e:
            logger.exception(f"Vision content extraction error: {e}")
            return self._fail(
                ERROR_VISION,
                f"Vision content extraction error: {e}",
                {},
                t0
            )

    # -----------------------------
    # Helpers
    # -----------------------------
    def _extract_text(self, selector: str) -> str:
        self.page.wait_for_selector(selector, state="attached", timeout=self.implicit_wait_ms)
        loc = self.page.locator(selector).first
        try:
            tag = loc.evaluate("el => el.tagName.toLowerCase()")
        except Exception:
            tag = ""
        if tag in {"input", "textarea"}:
            v = loc.input_value(timeout=self.implicit_wait_ms)
            return (v or "").strip()
        txt = loc.text_content(timeout=self.implicit_wait_ms)
        return (txt or "").strip()

    def _wait_for_url_change_or_timeout(self, before_url: str, timeout_ms: int) -> None:
        deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < deadline:
            if self.page.url != before_url:
                return
            time.sleep(0.05)
        raise PWTimeoutError(f"URL did not change within {timeout_ms}ms (still {self.page.url})")

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
            ".cookie-banner button",
            "[id*='cookie'] button:has-text('Accept')",
            ".newsletter-popup button:has-text('Close')",
            ".newsletter-popup [aria-label='Close']",
        ]

        for sel in candidates:
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    logger.info(f"Closing popup using selector: {sel}")
                    loc.first.click(timeout=1000)
                    time.sleep(0.35)
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
            if result.get("has_popup") and result.get("confidence", 0) > 0.6:
                close_selector = result.get("close_selector")
                if close_selector:
                    try:
                        self.page.locator(close_selector).first.click(timeout=2000)
                        time.sleep(0.35)
                        logger.info("✓ Popup closed by vision")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to click vision selector: {e}")
            return False
        except Exception as e:
            logger.debug(f"Vision popup detection error: {e}")
            return False

    def _screenshot(self) -> str:
        fname = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(self.artifacts_dir, fname)
        try:
            self.page.screenshot(path=path, full_page=True)
            return path
        except Exception:
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
                logger.info(f"Resizing screenshot from {width}x{height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                screenshot_bytes = buffer.getvalue()
            else:
                logger.info(f"Screenshot size OK: {width}x{height}")

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


# -----------------------------
# Example Usage (debug individually)
# -----------------------------
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment to run browser_agent.py directly.")

    agent = PlaywrightBrowserAgent(
        headless=_env_bool("BROWSER_HEADLESS", False),
        artifacts_dir=os.getenv("BROWSER_ARTIFACTS_DIR", "artifacts"),
        anthropic_api_key=api_key,
        use_vision=_env_bool("BROWSER_USE_VISION", True),
        stealth_mode=_env_bool("BROWSER_STEALTH_MODE", True),
        search_engine=os.getenv("BROWSER_SEARCH_ENGINE", "duckduckgo"),
        use_chrome_channel=_env_bool("BROWSER_USE_CHROME_CHANNEL", True),
    )

    try:
        steps = [
            {"id": "step_1", "action": "search", "args": {"query": "best sushi restaurants in San Francisco"}, "retries": 2},
            {"id": "step_2", "action": "click_first_result", "args": {"result_type": "any"}, "retries": 3},
            {"id": "step_3", "action": "extract_page_content", "args": {"max_items": 5}, "retries": 2},
        ]

        ctx: Dict[str, Any] = {"vars": {}}

        for step in steps:
            result = agent.run_step(step, ctx)
            print("\nRESULT:", json.dumps(result, indent=2))
            if not result.get("ok"):
                break

    finally:
        agent.close()