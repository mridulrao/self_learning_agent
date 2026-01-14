"""
Execute Workflow with Automatic Validation
Validates and normalizes workflow before execution
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv
from anthropic import Anthropic
from orchestrator import WorkflowOrchestrator

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("workflow_execution.log"),
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()


# =============================================================================
# Execution-time normalization helpers
# =============================================================================
_REQUIRED_STAR_RE = re.compile(r"\s*\*\s*$")
_WS_RE = re.compile(r"\s+")


def _normalize_label(label: str) -> str:
    """
    Normalize Google Form question labels to be more robust.
    - Strip trailing required marker star "*"
    - Collapse whitespace
    - Preserve casing and punctuation otherwise (label-sensitive forms)
    """
    s = (label or "").strip()
    s = _REQUIRED_STAR_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _coerce_singleton_lists(value: Any) -> Any:
    """
    If the validator outputs ["Yes"] for a single-choice field, coerce to "Yes".
    Keep real multi-select lists (len>1) as-is.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def normalize_workflow_for_execution(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize workflow right before execution so the orchestrator + browser agent
    receive consistent, label-matchable values.

    Handles:
    - wrapper format: {"metadata": ..., "workflow": {...}}
    - fill_google_form.args.form_data key normalization (strip trailing '*')
    - singleton list coercion in form_data values
    """
    if not isinstance(raw, dict):
        return raw

    # Unwrap { metadata, workflow } if present
    if "workflow" in raw and isinstance(raw.get("workflow"), dict):
        workflow = raw["workflow"]
        # Keep metadata attached (optional) but orchestrator likely expects workflow root
        # You can reattach metadata if your orchestrator uses it; we prioritize correct execution.
    else:
        workflow = raw

    steps = workflow.get("steps")
    if not isinstance(steps, list):
        return workflow

    for st in steps:
        if not isinstance(st, dict):
            continue
        if st.get("action") != "fill_google_form":
            continue

        args = st.get("args") or {}
        form_data = args.get("form_data")

        if isinstance(form_data, dict):
            new_form_data: Dict[str, Any] = {}
            for k, v in form_data.items():
                nk = _normalize_label(str(k))
                new_form_data[nk] = _coerce_singleton_lists(v)
            args["form_data"] = new_form_data
            st["args"] = args

    return workflow


# =============================================================================
# Workflow Validator
# =============================================================================
class WorkflowValidator:
    """Validates and normalizes workflows to match orchestrator expectations"""

    # Reference workflow template
    REFERENCE_WORKFLOW = {
        "description": "Fill Google Form, submit, then save returned details to TextEdit",
        "steps": [
            {
                "id": "step_1",
                "action": "goto",
                "args": {"url": "https://forms.gle/EXAMPLE"},
                "agent": "browser",
                "description": "Open the Google Form",
                "retries": 2,
                "delay_after": 1.5,
            },
            {
                "id": "step_2",
                "action": "fill_google_form",
                "args": {
                    "passes": 2,
                    "form_data": {
                        "Full Name": "John Doe",
                        "Already filled": "No",
                    },
                },
                "agent": "browser",
                "description": "Fill the form fields (multi-pass with required validation)",
                "retries": 0,
                "delay_after": 0.8,
            },
            {
                "id": "step_3",
                "action": "submit_form",
                "args": {
                    "require_no_missing_required": True,
                    "return_filled_details": True,
                    "return_form_id": True,
                },
                "agent": "browser",
                "description": "Submit the form and capture {form_id, filled_details, fill_results}",
                "retries": 2,
                "delay_after": 0.8,
            },
            {
                "id": "step_4",
                "action": "launch_fullscreen",
                "args": {"app_name": "TextEdit"},
                "agent": "desktop",
                "description": "Launch TextEdit",
                "retries": 2,
                "delay_after": 1.0,
            },
            {
                "id": "step_5",
                "action": "click_element",
                "args": {"element_description": "New Document button"},
                "agent": "desktop",
                "description": "Create a new document",
                "retries": 3,
                "delay_after": 0.5,
            },
            {
                "id": "step_6",
                "action": "type_text",
                "args": {
                    "text_from_ctx": "form_submission_text",
                    "fallback_text": "No form submission text found.\n",
                },
                "agent": "desktop",
                "description": "Type the captured form submission details into TextEdit",
                "retries": 1,
                "delay_after": 0.4,
            },
            {
                "id": "step_7",
                "action": "hotkey",
                "args": {"keys": ["CMD", "S"]},
                "agent": "desktop",
                "description": "Save (CMD+S)",
                "retries": 1,
                "delay_after": 0.5,
            },
            {
                "id": "step_8",
                "action": "type_text",
                "args": {"text": "google_form_submission.txt"},
                "agent": "desktop",
                "description": "Enter filename",
                "retries": 1,
                "delay_after": 0.3,
            },
            {
                "id": "step_9",
                "action": "hotkey",
                "args": {"keys": ["ENTER"]},
                "agent": "desktop",
                "description": "Confirm save",
                "retries": 1,
                "delay_after": 0.5,
            },
        ],
    }

    def __init__(self, anthropic_api_key: str):
        """Initialize with Anthropic API key"""
        self.client = Anthropic(api_key=anthropic_api_key)

    def validate_and_normalize(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate workflow and return normalized version

        Args:
            workflow: Input workflow to validate

        Returns:
            Dict with validation results and corrected workflow
        """
        logger.info(f"Validating workflow: {workflow.get('description', workflow.get('user_intent', 'No description'))}")

        prompt = f"""You are a workflow validation + normalization expert.
You will be given:
1) A REFERENCE WORKFLOW describing the orchestrator-compatible structure.
2) An INPUT WORKFLOW that may have missing/extra steps or wrong values.
3) Voice notes attached in step.metadata.voice_note.transcript which describe what values should be used when filling the Google Form.

REFERENCE WORKFLOW (ideal structure):
{json.dumps(self.REFERENCE_WORKFLOW, indent=2)}

INPUT WORKFLOW (to validate and normalize):
{json.dumps(workflow, indent=2)}

GOAL:
Return a corrected_workflow that is orchestrator-compatible AND updates Google Form values based on voice notes.

============================================================
CORE STRUCTURE RULES (Orchestrator Format)
============================================================
A) The corrected workflow MUST have these phases in this order:

BROWSER PHASE (in order):
1) goto (browser) with args.url
2) OPTIONAL: extract_form_id (browser) with args={{}} (best-effort)
3) fill_google_form (browser) with args.passes (int) and args.form_data (object)
4) submit_form (browser) with args:
- require_no_missing_required (boolean)
- return_filled_details (boolean)
- return_form_id (boolean)

DESKTOP PHASE (in order):
5) launch_fullscreen (desktop) args.app_name="TextEdit"
6) click_element (desktop) args.element_description (string)
7) type_text (desktop) args.text_from_ctx MUST be "form_submission_text"
8) hotkey (desktop) args.keys=["CMD","S"]
9) type_text (desktop) args.text (filename string)
10) hotkey (desktop) args.keys=["ENTER"]

B) Step IDs must be sequential: "step_1", "step_2", ...

C) Each step MUST have: id, action, args(object), agent, retries(int), delay_after(number).
time_ms and metadata are allowed and should be preserved if present.

D) IMPORTANT: total steps may be 9 or 10 depending on whether extract_form_id exists.
- If extract_form_id exists in INPUT, keep it and output 10 steps.
- If it does not exist, output 9 steps (skip it).
Do NOT include any other extra steps.

============================================================
VOICE NOTE RULES (How to modify form_data)
============================================================
E) Use ALL voice_note transcripts across the workflow steps as instructions for how to fill the form.

F) When voice notes conflict with existing form_data values:
- Prefer the voice notes.
- If multiple voice notes conflict, prefer the latest one by time_ms (or by voice_note.start_ms if time_ms ties).

G) NEVER change the question labels (keys) unless fixing obvious whitespace/quote artifacts.

============================================================
PRESERVATION RULES
============================================================
H) Preserve from INPUT wherever compatible:
- goto.args.url
- fill_google_form.args.passes (if missing, default to 2)
- submit_form args flags (if missing, default to require_no_missing_required=true, return_filled_details=true, return_form_id=true)
- Keep metadata.voice_note, time_ms, and other metadata fields if they exist.

I) Desktop:
- Step 7 MUST type from ctx "form_submission_text".
- Filename: if provided, preserve it. If missing, generate "google_form_submission.txt".

============================================================
CRITICAL OUTPUT FORMAT (JSON ONLY)
============================================================
Return ONLY a valid JSON object with these fields:
- is_valid: boolean
- issues: array of strings
- corrected_workflow: object with "description" and "steps" (ARRAY)

No markdown. No code fences. No extra explanation. JSON only.
"""

        logger.info("Calling Claude API for validation...")

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text.strip()
            logger.debug(f"Raw API response: {response_text[:500]}...")

            # Strip code fences if present
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()

            result = json.loads(response_text)

            if not isinstance(result, dict):
                raise ValueError("Result is not a dictionary")
            if "corrected_workflow" not in result:
                raise ValueError("Result missing 'corrected_workflow' field")

            corrected = result["corrected_workflow"]
            if not isinstance(corrected, dict):
                raise ValueError("corrected_workflow is not a dictionary")
            if "steps" not in corrected:
                raise ValueError("corrected_workflow missing 'steps' field")
            if not isinstance(corrected["steps"], list):
                logger.error(f"Steps field is not a list, it's: {type(corrected['steps'])}")
                raise ValueError("corrected_workflow.steps is not a list")

            if result.get("issues"):
                logger.warning(f"Issues found ({len(result['issues'])}):")
                for i, issue in enumerate(result["issues"], 1):
                    logger.warning(f"  {i}. {issue}")
            else:
                logger.info("No issues found")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            return {
                "is_valid": False,
                "issues": [f"Validation failed - JSON parse error: {str(e)}"],
                "corrected_workflow": workflow,
            }
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            return {
                "is_valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "corrected_workflow": workflow,
            }


# =============================================================================
# Validation / Execution functions
# =============================================================================
def validate_workflow_file(workflow_file: str, api_key: str, save_corrected: bool = True) -> Dict[str, Any]:
    logger.info(f"Loading workflow from: {workflow_file}")
    with open(workflow_file, "r") as f:
        workflow = json.load(f)

    validator = WorkflowValidator(api_key)
    result = validator.validate_and_normalize(workflow)

    corrected_workflow = result["corrected_workflow"]

    # Save corrected workflow if requested
    if save_corrected:
        corrected_file = workflow_file.replace(".json", "_corrected.json") if "_corrected" not in workflow_file else workflow_file
        with open(corrected_file, "w") as f:
            json.dump(corrected_workflow, f, indent=2)
        logger.info(f"Corrected workflow saved to: {corrected_file}")

    return corrected_workflow


def execute_validated_workflow(
    workflow_file: str,
    anthropic_api_key: str,
    browser_headless: bool = False,
    auto_cleanup: bool = True,
    stop_on_error: bool = True,
    save_result: bool = True,
    skip_validation: bool = False,
):
    logger.info(f"Loading workflow from: {workflow_file}")
    with open(workflow_file, "r") as f:
        original_workflow = json.load(f)

    # Validate and normalize workflow (unless skipped)
    if skip_validation:
        logger.warning("Validation skipped - using original workflow")
        workflow_raw = original_workflow
    else:
        validator = WorkflowValidator(anthropic_api_key)
        validation_result = validator.validate_and_normalize(original_workflow)
        workflow_raw = validation_result.get("corrected_workflow", original_workflow)

        corrected_file = workflow_file.replace(".json", "_corrected.json")
        try:
            with open(corrected_file, "w") as f:
                json.dump(workflow_raw, f, indent=2)
            logger.info(f"Corrected workflow saved to: {corrected_file}")
        except Exception as e:
            logger.warning(f"Failed to save corrected workflow: {e}")

        if not validation_result.get("is_valid", False):
            logger.warning("Workflow required corrections - using validated version")
        else:
            logger.info("Workflow validated cleanly - using validated version")

    # -----------------------------
    # EXECUTION-TIME NORMALIZATION
    # -----------------------------
    workflow = normalize_workflow_for_execution(workflow_raw)

    # Basic sanity check
    if not isinstance(workflow, dict) or not isinstance(workflow.get("steps"), list):
        logger.error("Workflow is not executable (missing steps list). Falling back to original workflow.")
        workflow = normalize_workflow_for_execution(original_workflow)

    orchestrator = WorkflowOrchestrator(
        anthropic_api_key=anthropic_api_key,
        browser_headless=browser_headless,
        auto_cleanup=auto_cleanup,
    )

    logger.info(f"Executing workflow: {workflow.get('description')}")
    logger.info(f"Total steps: {len(workflow.get('steps', []))}")

    result = orchestrator.execute_workflow(
        workflow,
        workflow_id="validated_workflow",
        stop_on_error=stop_on_error,
        save_result=save_result,
    )

    logger.info(f"Status: {result.status}")
    logger.info(f"Completed: {result.completed_steps}/{result.total_steps} steps")

    if result.status == "completed":
        logger.info("Workflow completed successfully!")
    elif result.status == "failed":
        logger.error(f"Workflow failed at step {result.completed_steps + 1}")
        if hasattr(result, "error") and result.error:
            logger.error(f"Error: {result.error}")
    else:
        logger.warning(f"Workflow ended with status: {result.status}")

    return result


# =============================================================================
# Recording helpers
# =============================================================================
def get_latest_recording_folder(base_dir="recordings"):
    """Find the latest recording folder based on creation time"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Recordings directory '{base_dir}' not found")

    folders = [f for f in Path(base_dir).iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No recording folders found in '{base_dir}'")

    latest_folder = max(folders, key=lambda x: x.stat().st_mtime)
    return latest_folder


def find_workflow_file(folder_path):
    """Find the workflow file in the given folder"""
    folder = Path(folder_path)

    workflow_patterns = [
        "events_workflow_voice_form.json",
    ]

    for pattern in workflow_patterns:
        matches = list(folder.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No workflow file found in {folder}")


# =============================================================================
# Main
# =============================================================================
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        logger.error("Please set it in your .env file")
        sys.exit(1)

    # Get workflow file from command line or use latest recording
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        workflow_file = sys.argv[1]
    else:
        try:
            latest_folder = get_latest_recording_folder()
            workflow_file = find_workflow_file(latest_folder)
            logger.info(f"Using latest recording: {latest_folder.name}")
            logger.info(f"Workflow file: {workflow_file.name}")
        except Exception as e:
            logger.error(f"Error finding latest workflow: {e}")
            sys.exit(1)

    if not os.path.exists(workflow_file):
        logger.error(f"Workflow file not found: {workflow_file}")
        sys.exit(1)

    validate_only = "--validate-only" in sys.argv
    skip_validation = "--skip-validation" in sys.argv
    headless = "--headless" in sys.argv

    try:
        if validate_only:
            logger.info("Running validation only (no execution)")
            _ = validate_workflow_file(str(workflow_file), api_key, save_corrected=True)
            logger.info("Validation complete - corrected workflow saved")
        else:
            _ = execute_validated_workflow(
                workflow_file=str(workflow_file),
                anthropic_api_key=api_key,
                browser_headless=headless,
                auto_cleanup=True,
                stop_on_error=True,
                save_result=True,
                skip_validation=skip_validation,
            )
            logger.info("Execution complete")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
