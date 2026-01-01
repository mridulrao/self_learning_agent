from __future__ import annotations

import json
import time
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path
from datetime import datetime, timezone

from desktop_agent import MacDesktopAgent
from browser_agent import PlaywrightBrowserAgent
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
# Logging Setup
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


logger = setup_logger("WorkflowOrchestrator", level=logging.INFO)


# -----------------------------
# Data Models
# -----------------------------
@dataclass
class StepResult:
    step_id: str
    agent: Literal["browser", "desktop"]
    action: str
    ok: bool
    error_type: Optional[str]
    message: str
    evidence: Dict[str, Any]
    extracted: Dict[str, Any]
    timing_ms: int
    attempt: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class WorkflowResult:
    workflow_id: str
    description: str
    status: Literal["success", "failed", "partial"]
    total_steps: int
    completed_steps: int
    failed_step: Optional[str]
    total_time_ms: int
    step_results: List[StepResult]
    context: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "description": self.description,
            "status": self.status,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "total_time_ms": self.total_time_ms,
            "timestamp": self.timestamp,
            "step_results": [
                {
                    "step_id": r.step_id,
                    "agent": r.agent,
                    "action": r.action,
                    "ok": r.ok,
                    "error_type": r.error_type,
                    "message": r.message,
                    "evidence": r.evidence,
                    "extracted": r.extracted,
                    "timing_ms": r.timing_ms,
                    "attempt": r.attempt,
                    "timestamp": r.timestamp,
                }
                for r in self.step_results
            ],
            "context": self.context,
        }


# -----------------------------
# Helper Function to Format Restaurant Text
# -----------------------------
def format_restaurants_for_textedit(restaurants: List[Dict[str, Any]], title: str = "🍽️ Restaurants") -> str:
    if not restaurants:
        return "No restaurants found.\n"

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("")

    for i, restaurant in enumerate(restaurants, 1):
        name = restaurant.get("name") or "Unknown Restaurant"
        lines.append(f"{i}. {name}")
        lines.append("-" * 40)

        for k, label in [
            ("rating", "⭐ Rating"),
            ("cuisine", "🍜 Cuisine"),
            ("price_range", "💰 Price"),
            ("address", "📍 Address"),
            ("phone", "📞 Phone"),
            ("website", "🌐 Website"),
            ("description", "📝"),
        ]:
            v = restaurant.get(k)
            if v:
                lines.append(f"   {label}: {v}" if label != "📝" else f"   📝 {v}")

        lines.append("")

    return "\n".join(lines)


# -----------------------------
# Workflow Orchestrator
# -----------------------------
class WorkflowOrchestrator:
    """
    Orchestrates multi-agent workflows combining browser and desktop automation.

    Adds:
    - Dynamic arg resolution from ctx (e.g. text_from_ctx)
    - Auto-derivation of restaurants_text after extraction
    """

    def __init__(
        self,
        anthropic_api_key: str,
        browser_headless: bool = False,
        browser_use_chrome_channel: bool = True,
        browser_stealth_mode: bool = True,
        browser_search_engine: str = "duckduckgo",
        browser_artifacts_dir: str = "artifacts",
        desktop_save_screenshots: bool = True,
        desktop_debug_clicks: bool = True,
        desktop_apply_menubar_offset: bool = True,
        desktop_debug_dir: str = "/tmp/desktop_agent_debug",
        results_dir: str = "workflow_results",
        auto_cleanup: bool = True,
        default_step_delay: float = 0.5,
    ):
        self.anthropic_api_key = anthropic_api_key
        self.browser_headless = browser_headless
        self.browser_use_chrome_channel = browser_use_chrome_channel
        self.browser_stealth_mode = browser_stealth_mode
        self.browser_search_engine = browser_search_engine
        self.browser_artifacts_dir = browser_artifacts_dir

        self.desktop_save_screenshots = desktop_save_screenshots
        self.desktop_debug_clicks = desktop_debug_clicks
        self.desktop_apply_menubar_offset = desktop_apply_menubar_offset
        self.desktop_debug_dir = desktop_debug_dir

        self.results_dir = Path(results_dir)
        self.auto_cleanup = auto_cleanup
        self.default_step_delay = default_step_delay

        self.results_dir.mkdir(exist_ok=True)

        self._browser_agent: Optional[PlaywrightBrowserAgent] = None
        self._desktop_agent: Optional[MacDesktopAgent] = None

        logger.info("WorkflowOrchestrator initialized")
        logger.info(f"Results directory: {self.results_dir}")

    # -----------------------------
    # Agent Management
    # -----------------------------
    def _get_browser_agent(self) -> PlaywrightBrowserAgent:
        if self._browser_agent is None:
            logger.info("Initializing browser agent...")
            self._browser_agent = PlaywrightBrowserAgent(
                headless=self.browser_headless,
                artifacts_dir=self.browser_artifacts_dir,
                anthropic_api_key=self.anthropic_api_key,
                use_vision=True,
                search_engine=self.browser_search_engine,
                stealth_mode=self.browser_stealth_mode,
                use_chrome_channel=self.browser_use_chrome_channel,
            )
            self._browser_agent.launch()
            logger.info("✓ Browser agent initialized")
        return self._browser_agent

    def _get_desktop_agent(self) -> MacDesktopAgent:
        if self._desktop_agent is None:
            logger.info("Initializing desktop agent...")
            self._desktop_agent = MacDesktopAgent(
                anthropic_api_key=self.anthropic_api_key,
                save_screenshots=self.desktop_save_screenshots,
                debug_show_clicks=self.desktop_debug_clicks,
                apply_menubar_offset=self.desktop_apply_menubar_offset,
                debug_dir=self.desktop_debug_dir,
            )
            logger.info("✓ Desktop agent initialized")
        return self._desktop_agent

    def cleanup(self):
        logger.info("Cleaning up agents...")

        if self._browser_agent:
            try:
                self._browser_agent.close()
                logger.info("✓ Browser agent closed")
            except Exception as e:
                logger.warning(f"Error closing browser agent: {e}")
            finally:
                self._browser_agent = None

        if self._desktop_agent:
            self._desktop_agent = None
            logger.info("✓ Desktop agent cleaned up")

    # -----------------------------
    # Dynamic arg resolution
    # -----------------------------
    def _resolve_step_args(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supports:
        - args.text_from_ctx: name of ctx["vars"] key to use as desktop type_text input
        - args.fallback_text: used if ctx var missing/empty
        """
        args = dict(step.get("args") or {})
        vars_ = context.get("vars", {})

        # Resolve text_from_ctx → text
        if "text_from_ctx" in args and "text" not in args:
            key = args.get("text_from_ctx")
            fallback = args.get("fallback_text", "")
            val = vars_.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                args["text"] = fallback
            else:
                args["text"] = val

        return args

    # -----------------------------
    # Workflow Execution
    # -----------------------------
    def execute_workflow(
        self,
        workflow: Dict[str, Any],
        workflow_id: Optional[str] = None,
        stop_on_error: bool = True,
        save_result: bool = True,
    ) -> WorkflowResult:
        t0 = time.time()

        if not workflow_id:
            workflow_id = f"workflow_{int(time.time() * 1000)}"

        description = workflow.get("description", "No description")
        steps = workflow.get("steps", [])

        logger.info("=" * 80)
        logger.info(f"STARTING WORKFLOW: {workflow_id}")
        logger.info(f"Description: {description}")
        logger.info(f"Total steps: {len(steps)}")
        logger.info(f"Stop on error: {stop_on_error}")
        logger.info("=" * 80)

        context: Dict[str, Any] = {
            "vars": {},
            "_evidence": {},
            "_workflow_id": workflow_id,
            "_start_time": t0,
        }

        step_results: List[StepResult] = []
        completed_steps = 0
        failed_step = None
        status: Literal["success", "failed", "partial"] = "success"

        try:
            for idx, step in enumerate(steps):
                step_num = idx + 1
                step_id = step.get("id", f"step_{step_num}")

                logger.info("")
                logger.info("─" * 80)
                logger.info(f"STEP {step_num}/{len(steps)}: {step_id}")
                logger.info(f"Description: {step.get('description', 'N/A')}")
                logger.info("─" * 80)

                # Resolve dynamic args before execution
                step = dict(step)
                step["args"] = self._resolve_step_args(step, context)

                result = self._execute_step(step, context)

                step_result = StepResult(
                    step_id=step_id,
                    agent=step.get("agent", "unknown"),
                    action=step.get("action", "unknown"),
                    ok=result.get("ok", False),
                    error_type=result.get("error_type"),
                    message=result.get("message", ""),
                    evidence=result.get("evidence", {}),
                    extracted=result.get("extracted", {}),
                    timing_ms=result.get("timing_ms", 0),
                    attempt=result.get("attempt", 1),
                )
                step_results.append(step_result)

                # Update context
                if step_result.extracted:
                    context["vars"].update(step_result.extracted)

                    # If restaurants present, derive restaurants_text automatically
                    if "restaurants" in step_result.extracted:
                        restaurants = step_result.extracted["restaurants"] or []
                        context["vars"]["restaurants"] = restaurants
                        context["vars"]["total_restaurants"] = len(restaurants)

                        # Derive formatted text used by desktop steps
                        context["vars"]["restaurants_text"] = format_restaurants_for_textedit(
                            restaurants,
                            title="🍣 Best Sushi Restaurants in San Francisco",
                        )

                    logger.info(f"Updated context with {len(step_result.extracted)} variables")

                if step_result.ok:
                    completed_steps += 1
                    logger.info(f"✓ Step {step_id} completed successfully ({step_result.timing_ms}ms)")

                    if idx < len(steps) - 1:
                        delay = float(step.get("delay_after", self.default_step_delay))
                        if delay > 0:
                            time.sleep(delay)
                else:
                    logger.error(f"✗ Step {step_id} failed: {step_result.message}")
                    failed_step = step_id
                    status = "failed" if stop_on_error else "partial"
                    if stop_on_error:
                        logger.error("Stopping workflow due to error")
                        break

        except KeyboardInterrupt:
            logger.warning("Workflow interrupted by user")
            status = "partial"
            failed_step = "interrupted"

        except Exception as e:
            logger.exception(f"Unexpected error during workflow execution: {e}")
            status = "failed"
            failed_step = "orchestrator_error"

        finally:
            if self.auto_cleanup:
                self.cleanup()

        total_time_ms = int((time.time() - t0) * 1000)

        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            description=description,
            status=status,
            total_steps=len(steps),
            completed_steps=completed_steps,
            failed_step=failed_step,
            total_time_ms=total_time_ms,
            step_results=step_results,
            context=context,
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"WORKFLOW COMPLETE: {workflow_id}")
        logger.info(f"Status: {status.upper()}")
        logger.info(f"Completed: {completed_steps}/{len(steps)} steps")
        logger.info(f"Total time: {total_time_ms}ms ({total_time_ms/1000:.2f}s)")
        if failed_step:
            logger.info(f"Failed at: {failed_step}")
        logger.info("=" * 80)

        if save_result:
            self._save_result(workflow_result)

        return workflow_result

    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        agent_type = (step.get("agent") or "").lower()

        if agent_type == "browser":
            agent = self._get_browser_agent()
            return agent.run_step(step, context)

        if agent_type == "desktop":
            agent = self._get_desktop_agent()
            return agent.run_step(step, context)

        return {"ok": False, "error_type": "unknown_agent", "message": f"Unknown agent type: {agent_type}", "evidence": {}, "extracted": {}, "timing_ms": 0}

    # -----------------------------
    # Result Persistence
    # -----------------------------
    def _save_result(self, result: WorkflowResult):
        filename = f"{result.workflow_id}_{result.timestamp.replace(':', '-').replace('.', '-')}.json"
        filepath = self.results_dir / filename
        try:
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"✓ Result saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def load_result(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        pattern = f"{workflow_id}_*.json"
        matches = list(self.results_dir.glob(pattern))
        if not matches:
            logger.warning(f"No result found for workflow: {workflow_id}")
            return None
        filepath = sorted(matches)[-1]
        try:
            with open(filepath, "r") as f:
                result = json.load(f)
            logger.info(f"✓ Loaded result from: {filepath}")
            return result
        except Exception as e:
            logger.error(f"Failed to load result: {e}")
            return None


# -----------------------------
# Workflow Builder Helper
# -----------------------------
class WorkflowBuilder:
    def __init__(self, description: str = ""):
        self.description = description
        self.steps: List[Dict[str, Any]] = []
        self._step_counter = 0

    def add_browser_step(self, action: str, args: Dict[str, Any], description: str = "", retries: int = 2, delay_after: float = 0.5) -> "WorkflowBuilder":
        self._step_counter += 1
        self.steps.append(
            {
                "id": f"step_{self._step_counter}",
                "action": action,
                "args": args,
                "retries": retries,
                "description": description or f"{action} with {args}",
                "agent": "browser",
                "delay_after": delay_after,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return self

    def add_desktop_step(self, action: str, args: Dict[str, Any], description: str = "", retries: int = 2, delay_after: float = 0.5) -> "WorkflowBuilder":
        self._step_counter += 1
        self.steps.append(
            {
                "id": f"step_{self._step_counter}",
                "action": action,
                "args": args,
                "retries": retries,
                "description": description or f"{action} with {args}",
                "agent": "desktop",
                "delay_after": delay_after,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return self

    def build(self) -> Dict[str, Any]:
        return {"description": self.description, "steps": self.steps}


# -----------------------------
# Example Usage (end-to-end demo)
# -----------------------------
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment to run orchestrator.py.")

    orchestrator = WorkflowOrchestrator(
        anthropic_api_key=api_key,
        browser_headless=_env_bool("BROWSER_HEADLESS", False),
        browser_use_chrome_channel=_env_bool("BROWSER_USE_CHROME_CHANNEL", True),
        browser_stealth_mode=_env_bool("BROWSER_STEALTH_MODE", True),
        browser_search_engine=os.getenv("BROWSER_SEARCH_ENGINE", "duckduckgo"),
        browser_artifacts_dir=os.getenv("BROWSER_ARTIFACTS_DIR", "artifacts"),
        desktop_save_screenshots=_env_bool("DESKTOP_SAVE_SCREENSHOTS", True),
        desktop_debug_clicks=_env_bool("DESKTOP_DEBUG_CLICKS", True),
        desktop_apply_menubar_offset=_env_bool("DESKTOP_APPLY_MENUBAR_OFFSET", True),
        desktop_debug_dir=os.getenv("DESKTOP_DEBUG_DIR", "/tmp/desktop_agent_debug"),
        results_dir=os.getenv("ORCH_RESULTS_DIR", "workflow_results"),
        auto_cleanup=_env_bool("ORCH_AUTO_CLEANUP", True),
        default_step_delay=_env_float("ORCH_DEFAULT_STEP_DELAY", 0.5),
    )

    workflow = (
        WorkflowBuilder("Search for best pizza restaurants, extract info, and save to TextEdit")
        .add_browser_step(
            action="search",
            args={"query": "best pizza restaurants in San Francisco"},
            description="Search on DuckDuckGo",
            retries=2,
            delay_after=2.0,
        )
        .add_browser_step(
            action="click_first_result",
            args={"result_type": "any"},
            description="Click first search result",
            retries=3,
            delay_after=3.0,
        )
        .add_browser_step(
            action="extract_restaurants_from_html",
            args={"max_restaurants": 5},
            description="Extract restaurant information (DOM-first, Claude fallback)",
            retries=2,
            delay_after=1.0,
        )
        # Desktop phase
        .add_desktop_step(
            action="launch_fullscreen",
            args={"app_name": "TextEdit"},
            description="Launch TextEdit",
            retries=2,
            delay_after=1.0,
        )
        .add_desktop_step(
            action="click_element",
            args={"element_description": "New Document button"},
            description="Create a new document",
            retries=3,
            delay_after=0.5,
        )
        .add_desktop_step(
            action="type_text",
            args={"text_from_ctx": "restaurants_text", "fallback_text": "No restaurants found.\n"},
            description="Type extracted restaurants into document",
            retries=1,
            delay_after=0.5,
        )
        .add_desktop_step(
            action="hotkey",
            args={"keys": ["CMD", "S"]},
            description="Save (CMD+S)",
            retries=1,
            delay_after=0.5,
        )
        .add_desktop_step(
            action="type_text",
            args={"text": "sf_pizza_restaurants.txt"},
            description="Enter filename",
            retries=1,
            delay_after=0.3,
        )
        .add_desktop_step(
            action="hotkey",
            args={"keys": ["ENTER"]},
            description="Confirm save",
            retries=1,
            delay_after=0.5,
        )
        .build()
    )

    result = orchestrator.execute_workflow(
        workflow,
        workflow_id="sushi_search_save_demo",
        stop_on_error=True,
        save_result=True,
    )

    print("\nStatus:", result.status)
    restaurants = result.context.get("vars", {}).get("restaurants", [])
    print("Restaurants extracted:", len(restaurants))
