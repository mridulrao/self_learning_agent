"""
Entry point for computer use workflow.
RPC style communication between different services, repsonsible for orchestrating the workflow.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from domain import (
    DescriptionSpec,
    LocateSpec,
    StepAction,
    StepArtifact,
    VerificationSpec,
    WorkflowDefinition,
    WorkflowStepDefinition,
)
from env_config import load_env_file
from orchestrator import WorkflowOrchestrator
from services import LocatorService, UIAutomationService, VerificationService
from services.verification import create_vision_client

load_env_file()


SCRIPT_DIR = Path(__file__).resolve().parent
MACOS_INPUT_SCRIPT = SCRIPT_DIR / "macos_input.swift"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_WORKFLOW_OUTPUT_DIR = os.environ.get("COMPUTER_USE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
DEFAULT_QUERY = os.environ.get("COMPUTER_USE_QUERY", "Top stocks in the US")
DEFAULT_SEARCH_BOX_PROMPT = os.environ.get(
    "COMPUTER_USE_SEARCH_BOX_PROMPT",
    "the main search or address bar where a web query can be typed",
)
DEFAULT_FIRST_LINK_PROMPT = os.environ.get(
    "COMPUTER_USE_FIRST_LINK_PROMPT",
    "the first search result link relevant to top stocks in the US",
)
DEFAULT_NEW_NOTE_PROMPT = os.environ.get(
    "COMPUTER_USE_NEW_NOTE_PROMPT",
    "the New Note button in the Notes app toolbar",
)
DEFAULT_NOTE_PREFIX = os.environ.get("COMPUTER_USE_NOTE_PREFIX", "Top performing stocks")
DEFAULT_APP_LAUNCH_DELAY = float(os.environ.get("COMPUTER_USE_APP_LAUNCH_DELAY", "2.0"))
DEFAULT_PAGE_LOAD_DELAY = float(os.environ.get("COMPUTER_USE_PAGE_LOAD_DELAY", "3.0"))
DEFAULT_POST_ACTION_DELAY = float(os.environ.get("COMPUTER_USE_POST_ACTION_DELAY", "1.0"))
DEFAULT_LOG_LEVEL = os.environ.get("COMPUTER_USE_LOG_LEVEL", "INFO").upper()
COORDINATE_LOCATOR_ENDPOINT = os.environ.get(
    "COORDINATE_LOCATOR_ENDPOINT",
    "https://xmjgo3r2cn7lcj-8000.proxy.runpod.net/rpc",
)

LOGGER = logging.getLogger("computer_use_workflow")


def build_verification_prompt(expected_state: str) -> str:
    return (
        "You are verifying a computer-use workflow from a screenshot. "
        f"Answer with 'YES' if the screenshot matches this state: {expected_state}. "
        "If not, answer with 'NO' and one short reason."
    )


def build_content_prompt() -> str:
    return (
        "Read the visible webpage and extract the top-performing US stocks that are clearly shown. "
        "Return plain text only, with one stock per line."
    )


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic computer-use workflow on macOS.")
    parser.add_argument(
        "--locate-endpoint",
        default=COORDINATE_LOCATOR_ENDPOINT,
        help="LocateAnything JSON-RPC endpoint.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_WORKFLOW_OUTPUT_DIR, help="Directory for saved screenshots.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Text typed into the Safari search field.")
    parser.add_argument(
        "--search-box-prompt",
        default=DEFAULT_SEARCH_BOX_PROMPT,
        help="Prompt used to locate the Safari search box.",
    )
    parser.add_argument(
        "--first-link-prompt",
        default=DEFAULT_FIRST_LINK_PROMPT,
        help="Prompt used to locate the first result link.",
    )
    parser.add_argument(
        "--new-note-prompt",
        default=DEFAULT_NEW_NOTE_PROMPT,
        help="Prompt used to locate the Notes new note button.",
    )
    parser.add_argument("--note-prefix", default=DEFAULT_NOTE_PREFIX, help="Heading inserted into the new note.")
    parser.add_argument("--app-launch-delay", type=float, default=DEFAULT_APP_LAUNCH_DELAY)
    parser.add_argument("--page-load-delay", type=float, default=DEFAULT_PAGE_LOAD_DELAY)
    parser.add_argument("--post-action-delay", type=float, default=DEFAULT_POST_ACTION_DELAY)
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level, e.g. DEBUG, INFO, WARNING.")
    parser.add_argument("--fullscreen", action="store_true", help="Toggle Safari fullscreen after launch.")
    parser.add_argument("--verify", action="store_true", help="Run screenshot verification after major steps.")
    return parser.parse_args()


def build_workflow_definition(args: argparse.Namespace) -> WorkflowDefinition:
    steps = [
        WorkflowStepDefinition(
            step_name="step1_safari_open",
            before_actions=[
                StepAction(kind="open_application", app_name="Safari"),
                StepAction(kind="wait", seconds=args.app_launch_delay),
            ]
            + (
                [
                    StepAction(kind="set_safari_fullscreen"),
                    StepAction(kind="wait", seconds=args.post_action_delay),
                ]
                if args.fullscreen
                else []
            ),
            capture_screenshot=True,
            verification=VerificationSpec(
                prompt_template=build_verification_prompt("Safari is open and ready for input.")
            ),
        ),
        WorkflowStepDefinition(
            step_name="step2_search_box",
            locate=LocateSpec(
                prompt_template=args.search_box_prompt,
                screenshot_ref_step_name="step1_safari_open",
            ),
            action=StepAction(kind="click_located_target"),
            after_actions=[StepAction(kind="wait", seconds=args.post_action_delay)],
        ),
        WorkflowStepDefinition(
            step_name="step3_search_results",
            before_actions=[
                StepAction(kind="activate_application", app_name="Safari"),
                StepAction(kind="focus_address_bar"),
                StepAction(kind="wait", seconds=args.post_action_delay),
                StepAction(kind="paste_text", text_template="{query}"),
                StepAction(kind="press_return"),
                StepAction(kind="wait", seconds=args.page_load_delay),
            ],
            capture_screenshot=True,
            verification=VerificationSpec(
                prompt_template=build_verification_prompt(
                    "Google results for the query '{query}' are visible."
                )
            ),
        ),
        WorkflowStepDefinition(
            step_name="step4_first_link",
            locate=LocateSpec(
                prompt_template=args.first_link_prompt,
                screenshot_ref_step_name="step3_search_results",
            ),
            action=StepAction(kind="click_located_target"),
            after_actions=[StepAction(kind="wait", seconds=args.page_load_delay)],
        ),
        WorkflowStepDefinition(
            step_name="step5_page_loaded",
            capture_screenshot=True,
            verification=VerificationSpec(
                prompt_template=build_verification_prompt(
                    "The first search result page has loaded and contains stock-related content."
                )
            ),
            description=DescriptionSpec(
                prompt_template=build_content_prompt(),
                output_key="extracted_content",
            ),
        ),
        WorkflowStepDefinition(
            step_name="step6_notes_open",
            before_actions=[
                StepAction(kind="quit_application", app_name="Safari"),
                StepAction(kind="wait", seconds=args.post_action_delay),
                StepAction(kind="open_application", app_name="Notes"),
                StepAction(kind="wait", seconds=args.app_launch_delay),
                StepAction(kind="activate_application", app_name="Notes"),
                StepAction(kind="wait", seconds=args.post_action_delay),
            ],
            capture_screenshot=True,
            verification=VerificationSpec(
                prompt_template=build_verification_prompt("The Notes app is open and ready to create a note.")
            ),
        ),
        WorkflowStepDefinition(
            step_name="step7_new_note",
            locate=LocateSpec(
                prompt_template=args.new_note_prompt,
                screenshot_ref_step_name="step6_notes_open",
            ),
            action=StepAction(kind="create_new_note"),
            after_actions=[StepAction(kind="wait", seconds=args.post_action_delay)],
        ),
        WorkflowStepDefinition(
            step_name="step8_note_written",
            before_actions=[
                StepAction(kind="paste_text", text_template="{note_prefix}\n{extracted_content}"),
                StepAction(kind="wait", seconds=args.post_action_delay),
            ],
            capture_screenshot=True,
            verification=VerificationSpec(
                prompt_template=build_verification_prompt(
                    "A new note is open and contains the extracted stock list."
                )
            ),
            content_template="{note_prefix}\n{extracted_content}",
        ),
    ]

    return WorkflowDefinition(
        name="safari_to_notes",
        output_dir=Path(args.output_dir),
        verify=args.verify,
        context={
            "query": args.query,
            "note_prefix": args.note_prefix,
        },
        steps=steps,
    )


def create_orchestrator(args: argparse.Namespace) -> WorkflowOrchestrator:
    ui_service = UIAutomationService()
    locator_service = LocatorService(args.locate_endpoint)
    verification_service = VerificationService(create_vision_client())
    return WorkflowOrchestrator(ui_service, locator_service, verification_service)


def run_workflow(args: argparse.Namespace) -> list[StepArtifact]:
    orchestrator = create_orchestrator(args)
    return orchestrator.run(build_workflow_definition(args))


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        artifacts = run_workflow(args)
    except Exception:
        LOGGER.exception("Workflow failed.")
        raise
    summary = [
        {
            "step_name": artifact.step_name,
            "screenshot_path": str(artifact.screenshot_path) if artifact.screenshot_path else None,
            "debug_image_path": str(artifact.debug_image_path) if artifact.debug_image_path else None,
            "verification": artifact.verification,
            "content": artifact.content,
            "artifact_path": str(artifact.artifact_path) if artifact.artifact_path else None,
        }
        for artifact in artifacts
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
