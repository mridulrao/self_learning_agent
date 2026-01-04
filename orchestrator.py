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
from browser_agent import GoogleFormsBrowserAgent  # ✅ updated: your Google Forms agent
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
# Helper: format form submission details for TextEdit
# -----------------------------
def format_form_submission_for_textedit(
    *,
    form_id: str = "",
    submitted: Optional[bool] = None,
    filled_details: Optional[Dict[str, Any]] = None,
    fill_results: Optional[Dict[str, Any]] = None,
    before_url: str = "",
    after_url: str = "",
    missing_required_questions: Optional[List[str]] = None,
    title: str = "✅ Google Form Submission",
) -> str:
    filled_details = filled_details or {}
    fill_results = fill_results or {}
    missing_required_questions = missing_required_questions or []

    lines: List[str] = []
    lines.append(title)
    lines.append("=" * 72)
    lines.append("")
    if form_id:
        lines.append(f"Form ID: {form_id}")
    if submitted is not None:
        lines.append(f"Submitted: {submitted}")
    if before_url:
        lines.append(f"Before URL: {before_url}")
    if after_url:
        lines.append(f"After URL:  {after_url}")
    lines.append("")

    if missing_required_questions:
        lines.append("⚠️ Missing required questions detected:")
        for q in missing_required_questions:
            lines.append(f"  - {q}")
        lines.append("")

    lines.append("🧾 Filled Details")
    lines.append("-" * 72)
    if not filled_details:
        lines.append("(No filled_details returned)")
    else:
        for k, v in filled_details.items():
            vv = "" if v is None else str(v)
            lines.append(f"- {k}: {vv}")
    lines.append("")

    lines.append("🔎 Fill Results (status per key)")
    lines.append("-" * 72)
    if not fill_results:
        lines.append("(No fill_results returned)")
    else:
        for k, r in fill_results.items():
            if not isinstance(r, dict):
                lines.append(f"- {k}: {r}")
                continue
            status = r.get("status", "")
            ftype = r.get("type", "")
            verified = r.get("verified", None)
            reason = r.get("reason", "")
            observed = r.get("observed", "")
            bits = [b for b in [status, ftype] if b]
            suffix = f" ({', '.join(bits)})" if bits else ""
            lines.append(f"- {k}{suffix}")
            if verified is not None:
                lines.append(f"    verified: {verified}")
            if reason:
                lines.append(f"    reason: {reason}")
            if observed:
                lines.append(f"    observed: {observed}")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Workflow Orchestrator
# -----------------------------
class WorkflowOrchestrator:
    """
    Orchestrates multi-agent workflows combining:
      - browser: GoogleFormsBrowserAgent (your actions: goto, fill_google_form, submit_form, etc.)
      - desktop: MacDesktopAgent

    Key update:
      - When submit_form returns {form_id, filled_details, fill_results}, we derive a formatted
        ctx var `form_submission_text` and pass it to desktop agent via text_from_ctx.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        # Browser (GoogleFormsBrowserAgent) settings
        browser_headless: bool = False,
        browser_use_chrome_channel: bool = True,
        browser_stealth_mode: bool = True,
        browser_artifacts_dir: str = "artifacts",
        browser_default_timeout_ms: int = 15_000,
        browser_implicit_wait_ms: int = 6_000,
        browser_fill_passes: int = 2,
        browser_lazy_mount_scroll: bool = True,
        browser_scroll_between_questions_px: int = 150,
        browser_use_vision: bool = True,
        # Desktop
        desktop_save_screenshots: bool = True,
        desktop_debug_clicks: bool = True,
        desktop_apply_menubar_offset: bool = True,
        desktop_debug_dir: str = "/tmp/desktop_agent_debug",
        # Orchestrator
        results_dir: str = "workflow_results",
        auto_cleanup: bool = True,
        default_step_delay: float = 0.5,
    ):
        self.anthropic_api_key = anthropic_api_key

        # Browser config
        self.browser_headless = browser_headless
        self.browser_use_chrome_channel = browser_use_chrome_channel
        self.browser_stealth_mode = browser_stealth_mode
        self.browser_artifacts_dir = browser_artifacts_dir
        self.browser_default_timeout_ms = browser_default_timeout_ms
        self.browser_implicit_wait_ms = browser_implicit_wait_ms
        self.browser_fill_passes = browser_fill_passes
        self.browser_lazy_mount_scroll = browser_lazy_mount_scroll
        self.browser_scroll_between_questions_px = browser_scroll_between_questions_px
        self.browser_use_vision = browser_use_vision

        # Desktop config
        self.desktop_save_screenshots = desktop_save_screenshots
        self.desktop_debug_clicks = desktop_debug_clicks
        self.desktop_apply_menubar_offset = desktop_apply_menubar_offset
        self.desktop_debug_dir = desktop_debug_dir

        # Orchestrator config
        self.results_dir = Path(results_dir)
        self.auto_cleanup = auto_cleanup
        self.default_step_delay = default_step_delay

        self.results_dir.mkdir(exist_ok=True)

        self._browser_agent: Optional[GoogleFormsBrowserAgent] = None
        self._desktop_agent: Optional[MacDesktopAgent] = None

        logger.info("WorkflowOrchestrator initialized")
        logger.info(f"Results directory: {self.results_dir}")

    # -----------------------------
    # Agent Management
    # -----------------------------
    def _get_browser_agent(self) -> GoogleFormsBrowserAgent:
        if self._browser_agent is None:
            logger.info("Initializing GoogleFormsBrowserAgent...")
            self._browser_agent = GoogleFormsBrowserAgent(
                headless=self.browser_headless,
                viewport=(1280, 800),
                artifacts_dir=self.browser_artifacts_dir,
                default_timeout_ms=self.browser_default_timeout_ms,
                implicit_wait_ms=self.browser_implicit_wait_ms,
                stealth_mode=self.browser_stealth_mode,
                use_chrome_channel=self.browser_use_chrome_channel,
                fill_passes=self.browser_fill_passes,
                scroll_between_questions_px=self.browser_scroll_between_questions_px,
                lazy_mount_scroll=self.browser_lazy_mount_scroll,
                anthropic_api_key=self.anthropic_api_key,
                use_vision=self.browser_use_vision,
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
                # Note: submit_form() may already close the browser; this is safe.
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
          - args.json_from_ctx: similar, but serializes dict/list to pretty JSON string
        """
        args = dict(step.get("args") or {})
        vars_ = context.get("vars", {})

        # Resolve text_from_ctx -> text
        if "text_from_ctx" in args and "text" not in args:
            key = args.get("text_from_ctx")
            fallback = args.get("fallback_text", "")
            val = vars_.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                args["text"] = fallback
            else:
                args["text"] = val

        # Resolve json_from_ctx -> text (pretty JSON)
        if "json_from_ctx" in args and "text" not in args:
            key = args.get("json_from_ctx")
            fallback = args.get("fallback_text", "")
            val = vars_.get(key)
            if val is None:
                args["text"] = fallback
            else:
                try:
                    args["text"] = json.dumps(val, indent=2, ensure_ascii=False)
                except Exception:
                    args["text"] = str(val)

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

                # Update context vars with extracted payload
                if step_result.extracted:
                    context["vars"].update(step_result.extracted)
                    logger.info(f"Updated context with {len(step_result.extracted)} variables")

                    # ✅ NEW: If submit_form returned form details, derive text for desktop agent
                    # Your submit_form extracted_payload includes:
                    #   submitted, form_id, filled_details, fill_results, before_url, after_url, missing_required_questions, ...
                    if ("filled_details" in step_result.extracted) or ("form_id" in step_result.extracted):
                        try:
                            form_text = format_form_submission_for_textedit(
                                form_id=str(step_result.extracted.get("form_id") or ""),
                                submitted=step_result.extracted.get("submitted"),
                                filled_details=step_result.extracted.get("filled_details") or {},
                                fill_results=step_result.extracted.get("fill_results") or {},
                                before_url=str(step_result.extracted.get("before_url") or ""),
                                after_url=str(step_result.extracted.get("after_url") or ""),
                                missing_required_questions=step_result.extracted.get("missing_required_questions") or [],
                                title="✅ Google Form Submission (Captured by Orchestrator)",
                            )
                            context["vars"]["form_submission_text"] = form_text
                            context["vars"]["form_id"] = str(step_result.extracted.get("form_id") or "")
                            logger.info("Derived ctx.vars['form_submission_text'] for desktop agent")
                        except Exception as e:
                            logger.warning(f"Failed to derive form_submission_text: {e}")

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

        return {
            "ok": False,
            "error_type": "unknown_agent",
            "message": f"Unknown agent type: {agent_type}",
            "evidence": {},
            "extracted": {},
            "timing_ms": 0,
        }

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

    def add_browser_step(
        self,
        action: str,
        args: Dict[str, Any],
        description: str = "",
        retries: int = 2,
        delay_after: float = 0.5,
        assertions: Optional[List[Dict[str, Any]]] = None,
    ) -> "WorkflowBuilder":
        self._step_counter += 1
        self.steps.append(
            {
                "id": f"step_{self._step_counter}",
                "action": action,
                "args": args,
                "retries": retries,
                "assertions": assertions or [],
                "description": description or f"{action} with {args}",
                "agent": "browser",
                "delay_after": delay_after,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return self

    def add_desktop_step(
        self,
        action: str,
        args: Dict[str, Any],
        description: str = "",
        retries: int = 2,
        delay_after: float = 0.5,
    ) -> "WorkflowBuilder":
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
# Example Usage (Google Form -> TextEdit)
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
        browser_artifacts_dir=os.getenv("BROWSER_ARTIFACTS_DIR", "artifacts"),
        browser_default_timeout_ms=_env_int("BROWSER_DEFAULT_TIMEOUT_MS", 15_000),
        browser_implicit_wait_ms=_env_int("BROWSER_IMPLICIT_WAIT_MS", 6_000),
        browser_fill_passes=_env_int("FORM_FILL_PASSES", 2),
        browser_lazy_mount_scroll=_env_bool("FORM_LAZY_MOUNT_SCROLL", True),
        browser_scroll_between_questions_px=_env_int("FORM_SCROLL_BETWEEN_QUESTIONS_PX", 150),
        browser_use_vision=_env_bool("BROWSER_USE_VISION", True),
        desktop_save_screenshots=_env_bool("DESKTOP_SAVE_SCREENSHOTS", True),
        desktop_debug_clicks=_env_bool("DESKTOP_DEBUG_CLICKS", True),
        desktop_apply_menubar_offset=_env_bool("DESKTOP_APPLY_MENUBAR_OFFSET", True),
        desktop_debug_dir=os.getenv("DESKTOP_DEBUG_DIR", "/tmp/desktop_agent_debug"),
        results_dir=os.getenv("ORCH_RESULTS_DIR", "workflow_results"),
        auto_cleanup=_env_bool("ORCH_AUTO_CLEANUP", True),
        default_step_delay=_env_float("ORCH_DEFAULT_STEP_DELAY", 0.5),
    )

    # ✅ This workflow uses ONLY your GoogleFormsBrowserAgent actions.
    workflow = (
        WorkflowBuilder("Fill Google Form, submit, then save returned details to TextEdit")
        .add_browser_step(
            action="goto",
            args={"url": "https://forms.gle/j9FWqyBRidYCCoCRA"},
            description="Open the Google Form",
            retries=2,
            delay_after=1.5,
        )
        .add_browser_step(
            action="fill_google_form",
            args={
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
            description="Fill the form fields (multi-pass with required validation)",
            retries=0,
            delay_after=0.8,
        )
        .add_browser_step(
            action="submit_form",
            args={
                "require_no_missing_required": True,
                "return_filled_details": True,
                "return_form_id": True,
            },
            description="Submit the form and capture {form_id, filled_details, fill_results}",
            retries=2,
            delay_after=0.8,
        )
        # Desktop phase: save the returned details
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
            args={
                # ✅ the orchestrator derives this after submit_form
                "text_from_ctx": "form_submission_text",
                "fallback_text": "No form submission text found.\n",
            },
            description="Type the captured form submission details into TextEdit",
            retries=1,
            delay_after=0.4,
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
            args={"text": "google_form_submission.txt"},
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
        workflow_id="google_form_submit_save_demo",
        stop_on_error=True,
        save_result=True,
    )

    print("\nStatus:", result.status)
    print("Form ID:", result.context.get("vars", {}).get("form_id", ""))
    print("Submitted:", result.context.get("vars", {}).get("submitted", None))
