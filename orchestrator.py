#!/usr/bin/env python3
from __future__ import annotations

"""
orchestrator_v1.py

Workflow Orchestrator for schema "workflow_steps.v1".

This orchestrator:
- Reads a workflow_steps.v1 JSON (your derived primitive workflow)
- Dispatches each step to:
  - browser: PrimitiveBrowserAgent (Playwright)
  - desktop: MacDesktopAgent (OCR-grounded)

Design:
- The orchestrator does *not* reinterpret primitives.
- It forwards the step dict as-is to the agent’s runner:
    browser_agent.run_primitive(step, ctx)
    desktop_agent.run_primitive_step(step, ctx)
- Keeps a minimal ctx for cross-step variables (optional, mostly for future).
- Writes a result JSON to results_dir.

Logging policy:
- INFO: step-level progress
- DEBUG: optional details (enable via ORCH_LOG_LEVEL=DEBUG)
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv

# IMPORTANT: adjust import names to your actual filenames
from desktop_agent import MacDesktopAgent, Backoff as DesktopBackoff
from browser_agent import PrimitiveBrowserAgent, Backoff as BrowserBackoff

load_dotenv()


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------
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


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip()


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
    level = _parse_log_level(_env_str("ORCH_LOG_LEVEL", "INFO"), default=logging.INFO)
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


logger = setup_logger("WorkflowOrchestrator")


# -----------------------------------------------------------------------------
# Results models
# -----------------------------------------------------------------------------
@dataclass
class StepResult:
    step_id: str
    agent: Literal["browser", "desktop"]
    step_type: str
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
                    "step_type": r.step_type,
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


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
class WorkflowOrchestrator:
    """Executes workflow_steps.v1 by dispatching to browser/desktop agents."""

    def __init__(
        self,
        *,
        anthropic_api_key: Optional[str],
        workflow_artifacts_dir: Optional[str] = None,
        results_dir: str = "workflow_results",
        auto_cleanup: bool = True,
        default_step_delay_s: float = 0.2,
        stop_on_error: bool = True,
    ):
        self.anthropic_api_key = anthropic_api_key

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.workflow_artifacts_dir = Path(workflow_artifacts_dir) if workflow_artifacts_dir else None

        self.auto_cleanup = bool(auto_cleanup)
        self.default_step_delay_s = float(default_step_delay_s)
        self.stop_on_error = bool(stop_on_error)

        self._browser: Optional[PrimitiveBrowserAgent] = None
        self._desktop: Optional[MacDesktopAgent] = None

    # -------------------------
    # Agent management
    # -------------------------
    def _get_browser(self) -> PrimitiveBrowserAgent:
        if self._browser is None:
            self._browser = PrimitiveBrowserAgent(
                headless=_env_bool("BROWSER_HEADLESS", False) is True,
                artifacts_dir=_env_str("BROWSER_ARTIFACTS_DIR", "artifacts_browser"),
                backoff=BrowserBackoff(max_retries=_env_int("BROWSER_MAX_RETRIES", 2)),
                stealth_mode=_env_bool("BROWSER_STEALTH_MODE", True),
                default_channel=(_env_str("BROWSER_CHANNEL", "").strip() or None),
                verbose_console=_env_bool("BROWSER_VERBOSE_CONSOLE", False),
            )
        return self._browser

    def _get_desktop(self) -> MacDesktopAgent:
        if self._desktop is None:
            self._desktop = MacDesktopAgent(
                anthropic_api_key=self.anthropic_api_key,
                save_screenshots=_env_bool("DESKTOP_SAVE_SCREENSHOTS", True),
                debug_show_clicks=_env_bool("DESKTOP_DEBUG_CLICKS", False),
                apply_menubar_offset=_env_bool("DESKTOP_APPLY_MENUBAR_OFFSET", False),
                debug_dir=_env_str("DESKTOP_DEBUG_DIR", "/tmp/desktop_agent_debug"),
                backoff=DesktopBackoff(max_retries=_env_int("DESKTOP_MAX_RETRIES", 2)),
                ocr_base_url=_env_str("DESKTOP_OCR_URL", "").strip() or None,
                ocr_timeout_s=_env_int("DESKTOP_OCR_TIMEOUT_S", 120),
                ocr_min_conf=float(_env_float("DESKTOP_OCR_MIN_CONF", 0.55)),
                # Save “sent-to-ocr” screenshots alongside workflow file if provided
                workflow_artifacts_dir=str(self.workflow_artifacts_dir) if self.workflow_artifacts_dir else None,
            )
        return self._desktop

    def cleanup(self) -> None:
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None

        # Desktop agent holds no external session; just drop reference.
        self._desktop = None

    # -------------------------
    # Execution
    # -------------------------
    def execute(
        self,
        workflow: Dict[str, Any],
        *,
        workflow_id: Optional[str] = None,
    ) -> WorkflowResult:
        t0 = time.time()
        if not workflow_id:
            workflow_id = f"workflow_{int(time.time() * 1000)}"

        schema = str(workflow.get("schema") or "")
        steps = workflow.get("steps") or []

        if schema and schema != "workflow_steps.v1":
            logger.info("[orch] schema=%s (expected workflow_steps.v1)", schema)

        logger.info("[orch] start workflow_id=%s steps=%d", workflow_id, len(steps))

        ctx: Dict[str, Any] = {
            "vars": {},
            "_workflow_id": workflow_id,
            "_start_time": t0,
        }

        step_results: List[StepResult] = []
        completed = 0
        failed_step: Optional[str] = None
        status: Literal["success", "failed", "partial"] = "success"

        try:
            for i, step in enumerate(steps, start=1):
                agent = (step.get("agent") or "").strip().lower()
                step_type = (step.get("type") or step.get("kind") or "").strip()
                step_id = step.get("step_id") or step.get("id") or f"step_{i}"

                logger.info("[orch] step %d/%d %s agent=%s type=%s", i, len(steps), step_id, agent, step_type)

                res = self._dispatch_step(step, ctx)

                step_results.append(
                    StepResult(
                        step_id=step_id,
                        agent=("browser" if agent == "browser" else "desktop"),
                        step_type=step_type,
                        ok=bool(res.get("ok")),
                        error_type=res.get("error_type"),
                        message=str(res.get("message") or ""),
                        evidence=dict(res.get("evidence") or {}),
                        extracted=dict(res.get("extracted") or {}),
                        timing_ms=int(res.get("timing_ms") or 0),
                        attempt=int(res.get("attempt") or 1),
                    )
                )

                # Optional cross-step vars (not heavily used yet)
                if res.get("extracted"):
                    ctx["vars"].update(res["extracted"])

                if res.get("ok"):
                    completed += 1
                    delay = float(step.get("delay_after") or self.default_step_delay_s)
                    if delay > 0 and i < len(steps):
                        time.sleep(delay)
                    continue

                failed_step = step_id
                status = "failed" if self.stop_on_error else "partial"
                logger.error("[orch] failed step=%s error_type=%s", step_id, res.get("error_type"))

                if self.stop_on_error:
                    break

        except KeyboardInterrupt:
            status = "partial"
            failed_step = "interrupted"
            logger.info("[orch] interrupted")

        except Exception as e:
            status = "failed"
            failed_step = "orchestrator_error"
            logger.error("[orch] exception: %s", e)

        finally:
            if self.auto_cleanup:
                self.cleanup()

        total_ms = int((time.time() - t0) * 1000)
        out = WorkflowResult(
            workflow_id=workflow_id,
            status=status,
            total_steps=len(steps),
            completed_steps=completed,
            failed_step=failed_step,
            total_time_ms=total_ms,
            step_results=step_results,
            context=ctx,
        )

        self._save_result(out)
        logger.info("[orch] done status=%s completed=%d/%d time=%.2fs", status, completed, len(steps), total_ms / 1000.0)
        return out

    def _dispatch_step(self, step: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        agent = (step.get("agent") or "").strip().lower()
        if agent == "browser":
            b = self._get_browser()
            return b.run_primitive(step, ctx)

        if agent == "desktop":
            d = self._get_desktop()
            return d.run_primitive_step(step, ctx)

        return {
            "ok": False,
            "error_type": "unknown_agent",
            "message": f"Unknown agent: {agent}",
            "evidence": {},
            "extracted": {},
            "timing_ms": 0,
            "attempt": 1,
        }

    # -------------------------
    # Persistence
    # -------------------------
    def _save_result(self, result: WorkflowResult) -> None:
        fname = f"{result.workflow_id}_{result.timestamp.replace(':','-').replace('.','-')}.json"
        path = self.results_dir / fname
        try:
            path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.info("[orch] saved result %s", path)
        except Exception as e:
            logger.error("[orch] failed to save result: %s", e)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default="recordings/20260107_190820/workflow_steps.v1.json")
    ap.add_argument("--results-dir", default=_env_str("ORCH_RESULTS_DIR", "workflow_results"))
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--no-stop-on-error", action="store_true")
    ap.add_argument("--delay", type=float, default=_env_float("ORCH_DEFAULT_STEP_DELAY", 0.2))
    args = ap.parse_args()

    wf_path = Path(args.workflow).expanduser().resolve()
    if not wf_path.exists():
        raise SystemExit(f"Missing workflow file: {wf_path}")

    workflow = json.loads(wf_path.read_text(encoding="utf-8"))

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None

    orch = WorkflowOrchestrator(
        anthropic_api_key=api_key,
        workflow_artifacts_dir=str(wf_path.parent),
        results_dir=args.results_dir,
        auto_cleanup=(not args.no_cleanup),
        default_step_delay_s=float(args.delay),
        stop_on_error=(not args.no_stop_on_error),
    )

    orch.execute(workflow, workflow_id=wf_path.parent.name)


if __name__ == "__main__":
    main()
