import argparse
import json
import logging
import mimetypes
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_config import load_env_file

load_env_file()

try:
    from service.call_locateanything_rpc import (
        COORDINATE_LOCATOR_ENDPOINT,
        build_payload,
        call_rpc,
        encode_image_as_data_url,
    )
except ModuleNotFoundError:
    from call_locateanything_rpc import (
        COORDINATE_LOCATOR_ENDPOINT,
        build_payload,
        call_rpc,
        encode_image_as_data_url,
    )


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

LOGGER = logging.getLogger("computer_use_workflow")


class WorkflowError(RuntimeError):
    pass


@dataclass
class StepArtifacts:
    step_name: str
    screenshot_path: Path | None = None
    coordinates: dict[str, int] | None = None
    bounding_box: dict[str, int] | None = None
    screen_coordinates: dict[str, int] | None = None
    verification: str | None = None
    content: str | None = None
    debug_image_path: Path | None = None
    artifact_path: Path | None = None


@dataclass
class DisplayGeometry:
    width_points: int
    height_points: int
    width_pixels: int
    height_pixels: int


@dataclass
class LocateTarget:
    bounding_box: dict[str, int]
    click_point: dict[str, int]


class LocateAnythingClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def locate_target(self, screenshot_path: Path, prompt: str) -> LocateTarget:
        LOGGER.info("Locating bounding box for prompt: %s", prompt)
        payload = build_payload(
            screenshot_base64=encode_image_as_data_url(screenshot_path),
            prompt=prompt,
            output_type="box",
        )
        response = call_rpc(self.endpoint, payload)
        coordinates = response.get("result", {}).get("coordinates", [])
        if not coordinates:
            raise WorkflowError(f"No coordinates returned for prompt: {prompt}")
        box = coordinates[0]
        if not all(key in box for key in ("x1", "y1", "x2", "y2")):
            raise WorkflowError(f"Bounding box response was malformed for prompt {prompt}: {box}")

        click_point = {
            "x": round((box["x1"] + box["x2"]) / 2),
            "y": round((box["y1"] + box["y2"]) / 2),
        }
        LOGGER.info("Locator returned bounding box: %s", box)
        LOGGER.info("Using bounding box center as click point: %s", click_point)
        return LocateTarget(bounding_box=box, click_point=click_point)


class VisionClient:
    def describe(self, screenshot_path: Path, prompt: str) -> str:
        raise NotImplementedError

    def verify(self, screenshot_path: Path, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleVisionClient(VisionClient):
    def __init__(self, base_url: str, api_key: str, content_model: str, verify_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.content_model = content_model
        self.verify_model = verify_model

    def describe(self, screenshot_path: Path, prompt: str) -> str:
        return self._call_model(self.content_model, screenshot_path, prompt)

    def verify(self, screenshot_path: Path, prompt: str) -> str:
        return self._call_model(self.verify_model, screenshot_path, prompt)

    def _call_model(self, model: str, screenshot_path: Path, prompt: str) -> str:
        image_data_url = encode_image_as_data_url(screenshot_path)
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
        }
        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()

        if isinstance(body.get("output_text"), str) and body["output_text"].strip():
            return body["output_text"].strip()

        outputs = body.get("output", [])
        for item in outputs:
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        raise WorkflowError("Vision model response did not include text output.")


class SimpleHttpVisionClient(VisionClient):
    def __init__(self, content_url: str, verify_url: str, api_key: str | None = None) -> None:
        self.content_url = content_url
        self.verify_url = verify_url
        self.api_key = api_key

    def describe(self, screenshot_path: Path, prompt: str) -> str:
        return self._call_endpoint(self.content_url, screenshot_path, prompt)

    def verify(self, screenshot_path: Path, prompt: str) -> str:
        return self._call_endpoint(self.verify_url, screenshot_path, prompt)

    def _call_endpoint(self, url: str, screenshot_path: Path, prompt: str) -> str:
        payload = {
            "prompt": prompt,
            "image_base64": encode_image_as_data_url(screenshot_path),
            "mime_type": mimetypes.guess_type(screenshot_path.name)[0] or "image/png",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        body = response.json()

        for candidate in (
            body.get("text"),
            body.get("output_text"),
            body.get("result", {}).get("text"),
            body.get("result", {}).get("output_text"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        raise WorkflowError(f"No text field found in response from {url}")


def create_vision_client() -> VisionClient:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    content_model = os.environ.get("CONTENT_VISION_MODEL")
    verify_model = os.environ.get("VERIFY_VISION_MODEL")
    if openai_api_key and content_model and verify_model:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        LOGGER.info("Using OpenAI-compatible vision client with base URL: %s", base_url)
        return OpenAICompatibleVisionClient(base_url, openai_api_key, content_model, verify_model)

    content_url = os.environ.get("VISION_CONTENT_URL")
    verify_url = os.environ.get("VISION_VERIFY_URL")
    if content_url and verify_url:
        LOGGER.info("Using custom HTTP vision client.")
        return SimpleHttpVisionClient(content_url, verify_url, os.environ.get("VISION_API_KEY"))

    raise WorkflowError(
        "No vision client is configured. Set OPENAI_API_KEY + CONTENT_VISION_MODEL + VERIFY_VISION_MODEL, "
        "or set VISION_CONTENT_URL + VISION_VERIFY_URL."
    )


def run_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    LOGGER.debug("Running command: %s", args)
    return subprocess.run(args, check=check, text=True, capture_output=True)


def run_command_bytes(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    LOGGER.debug("Running binary command: %s", args)
    return subprocess.run(args, check=check, capture_output=True)


def run_applescript(script: str) -> None:
    result = run_command(["osascript", "-e", script], check=False)
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or "AppleScript command failed.")


def run_applescript_with_output(script: str) -> str:
    result = run_command(["osascript", "-e", script], check=False)
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or "AppleScript command failed.")
    return result.stdout.strip()


def open_application(app_name: str) -> None:
    LOGGER.info("Opening application: %s", app_name)
    result = run_command(["open", "-a", app_name], check=False)
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or f"Failed to open {app_name}")


def quit_application(app_name: str) -> None:
    LOGGER.info("Quitting application: %s", app_name)
    run_applescript(f'tell application "{app_name}" to quit')


def activate_application(app_name: str) -> None:
    LOGGER.info("Activating application: %s", app_name)
    run_applescript(f'tell application "{app_name}" to activate')


def set_safari_fullscreen() -> None:
    LOGGER.info("Toggling Safari fullscreen mode.")
    script = """
    tell application "Safari" to activate
    delay 0.5
    tell application "System Events"
        keystroke "f" using {command down, control down}
    end tell
    """
    run_applescript(script)


def take_screenshot(step_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / f"{step_name}.png"
    LOGGER.info("Taking screenshot for %s at %s", step_name, screenshot_path)
    result = run_command(["screencapture", "-x", str(screenshot_path)], check=False)
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or "Failed to capture screenshot.")
    return screenshot_path


def read_png_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise WorkflowError(f"Unsupported screenshot format for size detection: {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def get_display_geometry(screenshot_path: Path) -> DisplayGeometry:
    width_pixels, height_pixels = read_png_size(screenshot_path)
    result = run_command(
        ["swift", str(MACOS_INPUT_SCRIPT), "screen-geometry"],
        check=False,
    )
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or "Failed to resolve screen geometry.")

    try:
        geometry_payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise WorkflowError(f"Failed to parse screen geometry: {result.stdout}") from error

    width_points = int(geometry_payload["width_points"])
    height_points = int(geometry_payload["height_points"])
    if width_points <= 0 or height_points <= 0:
        raise WorkflowError(f"Invalid display geometry: {geometry_payload}")

    geometry = DisplayGeometry(
        width_points=width_points,
        height_points=height_points,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
    )
    LOGGER.info(
        "Display geometry resolved: %spx x %spx mapped to %spt x %spt",
        geometry.width_pixels,
        geometry.height_pixels,
        geometry.width_points,
        geometry.height_points,
    )
    return geometry


def translate_coordinates_for_screen(
    screenshot_path: Path,
    coordinates: dict[str, int],
) -> tuple[int, int]:
    geometry = get_display_geometry(screenshot_path)
    screen_x = round(coordinates["x"] * geometry.width_points / geometry.width_pixels)
    screen_y = round(coordinates["y"] * geometry.height_points / geometry.height_pixels)
    LOGGER.info(
        "Translated screenshot coordinates (%s, %s) to screen coordinates (%s, %s) using screenshot size %spx x %spx and screen size %spt x %spt",
        coordinates["x"],
        coordinates["y"],
        screen_x,
        screen_y,
        geometry.width_pixels,
        geometry.height_pixels,
        geometry.width_points,
        geometry.height_points,
    )
    return screen_x, screen_y


def move_and_click(x: int, y: int) -> None:
    LOGGER.info("Moving cursor and clicking at screen coordinates (%s, %s)", x, y)
    result = run_command(
        ["swift", str(MACOS_INPUT_SCRIPT), "move-click", str(x), str(y)],
        check=False,
    )
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or "Failed to move and click.")


def press_return() -> None:
    LOGGER.info("Pressing Return.")
    script = """
    tell application "System Events"
        key code 36
    end tell
    """
    run_applescript(script)


def create_new_note() -> None:
    LOGGER.info("Creating a new note with the native keyboard shortcut.")
    script = """
    tell application "System Events"
        keystroke "n" using {command down}
    end tell
    """
    run_applescript(script)


def type_text(text: str) -> None:
    LOGGER.info("Typing text: %s", text)
    script = f'''
    tell application "System Events"
        keystroke {json.dumps(text)}
    end tell
    '''
    run_applescript(script)


def paste_text(text: str) -> None:
    LOGGER.info("Pasting note text (%s characters).", len(text))
    subprocess.run(["pbcopy"], input=text, text=True, check=True)
    script = """
    tell application "System Events"
        keystroke "v" using {command down}
    end tell
    """
    run_applescript(script)


def focus_address_bar() -> None:
    LOGGER.info("Focusing the browser address bar.")
    script = """
    tell application "System Events"
        keystroke "l" using {command down}
    end tell
    """
    run_applescript(script)


def wait_for(seconds: float) -> None:
    LOGGER.debug("Waiting for %.2f seconds.", seconds)
    time.sleep(seconds)


def verify_or_raise(
    vision_client: VisionClient,
    screenshot_path: Path,
    prompt: str,
    artifacts: StepArtifacts,
    *,
    required_tokens: tuple[str, ...] = ("yes",),
) -> None:
    LOGGER.info("Running verification for %s", artifacts.step_name)
    verification = vision_client.verify(screenshot_path, prompt)
    artifacts.verification = verification
    lowered = verification.lower()
    if not all(token in lowered for token in required_tokens):
        raise WorkflowError(f"Verification failed for {artifacts.step_name}: {verification}")
    LOGGER.info("Verification passed for %s: %s", artifacts.step_name, verification)


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


def annotate_click_target(step: StepArtifacts, output_dir: Path) -> Path | None:
    if not step.screenshot_path or not step.bounding_box or not step.coordinates:
        return None

    debug_image_path = output_dir / f"{step.step_name}_debug.png"
    with Image.open(step.screenshot_path) as image:
        annotated = image.convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        box = step.bounding_box
        click = step.coordinates
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        point_x, point_y = click["x"], click["y"]
        crosshair_radius = max(10, round(min(annotated.size) * 0.012))
        marker_radius = max(6, round(min(annotated.size) * 0.008))

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 64, 64, 255),
            width=5,
            fill=(255, 64, 64, 35),
        )
        draw.ellipse(
            [
                (point_x - marker_radius, point_y - marker_radius),
                (point_x + marker_radius, point_y + marker_radius),
            ],
            fill=(0, 200, 255, 255),
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.line(
            [(point_x - crosshair_radius, point_y), (point_x + crosshair_radius, point_y)],
            fill=(0, 200, 255, 255),
            width=3,
        )
        draw.line(
            [(point_x, point_y - crosshair_radius), (point_x, point_y + crosshair_radius)],
            fill=(0, 200, 255, 255),
            width=3,
        )

        annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
        annotated.save(debug_image_path)

    step.debug_image_path = debug_image_path
    LOGGER.info("Wrote debug screenshot with click target: %s", debug_image_path)
    return debug_image_path


def persist_step_artifact(step: StepArtifacts, output_dir: Path) -> Path:
    artifact_path = output_dir / f"{step.step_name}.json"
    annotate_click_target(step, output_dir)
    payload = {
        "step_name": step.step_name,
        "screenshot_path": str(step.screenshot_path) if step.screenshot_path else None,
        "debug_image_path": str(step.debug_image_path) if step.debug_image_path else None,
        "verification": step.verification,
        "content": step.content,
    }
    artifact_path.write_text(json.dumps(payload, indent=2))
    step.artifact_path = artifact_path
    LOGGER.info("Wrote step artifact: %s", artifact_path)
    return artifact_path


def add_step_artifact(artifacts: list[StepArtifacts], step: StepArtifacts, output_dir: Path) -> None:
    persist_step_artifact(step, output_dir)
    artifacts.append(step)


def run_workflow(args: argparse.Namespace) -> list[StepArtifacts]:
    LOGGER.info("Starting computer-use workflow.")
    LOGGER.info("Workflow query: %s", args.query)
    LOGGER.info("Workflow screenshots directory: %s", args.output_dir)
    locate_client = LocateAnythingClient(args.locate_endpoint)
    vision_client = create_vision_client()
    artifacts: list[StepArtifacts] = []
    output_dir = Path(args.output_dir)

    open_application("Safari")
    wait_for(args.app_launch_delay)
    if args.fullscreen:
        set_safari_fullscreen()
        wait_for(args.post_action_delay)

    step1 = StepArtifacts(step_name="step1_safari_open")
    step1.screenshot_path = take_screenshot(step1.step_name, output_dir)
    if args.verify:
        verify_or_raise(
            vision_client,
            step1.screenshot_path,
            build_verification_prompt("Safari is open and ready for input."),
            step1,
        )
    add_step_artifact(artifacts, step1, output_dir)

    LOGGER.info("Step 2: locate and click Safari search box.")
    search_target = locate_client.locate_target(step1.screenshot_path, args.search_box_prompt)
    step2 = StepArtifacts(
        step_name="step2_search_box",
        screenshot_path=step1.screenshot_path,
        coordinates=search_target.click_point,
        bounding_box=search_target.bounding_box,
    )
    screen_x, screen_y = translate_coordinates_for_screen(step1.screenshot_path, search_target.click_point)
    step2.screen_coordinates = {"x": screen_x, "y": screen_y}
    move_and_click(screen_x, screen_y)
    wait_for(args.post_action_delay)
    add_step_artifact(artifacts, step2, output_dir)

    LOGGER.info("Step 3: type query and wait for results page.")
    activate_application("Safari")
    focus_address_bar()
    wait_for(args.post_action_delay)
    paste_text(args.query)
    press_return()
    wait_for(args.page_load_delay)

    step3 = StepArtifacts(step_name="step3_search_results")
    step3.screenshot_path = take_screenshot(step3.step_name, output_dir)
    if args.verify:
        verify_or_raise(
            vision_client,
            step3.screenshot_path,
            build_verification_prompt(f"Google results for the query '{args.query}' are visible."),
            step3,
        )
    add_step_artifact(artifacts, step3, output_dir)

    LOGGER.info("Step 4: locate and click the first search result.")
    first_link_target = locate_client.locate_target(step3.screenshot_path, args.first_link_prompt)
    step4 = StepArtifacts(
        step_name="step4_first_link",
        screenshot_path=step3.screenshot_path,
        coordinates=first_link_target.click_point,
        bounding_box=first_link_target.bounding_box,
    )
    screen_x, screen_y = translate_coordinates_for_screen(step3.screenshot_path, first_link_target.click_point)
    step4.screen_coordinates = {"x": screen_x, "y": screen_y}
    move_and_click(screen_x, screen_y)
    wait_for(args.page_load_delay)
    add_step_artifact(artifacts, step4, output_dir)

    LOGGER.info("Step 5: analyze loaded page content.")
    step5 = StepArtifacts(step_name="step5_page_loaded")
    step5.screenshot_path = take_screenshot(step5.step_name, output_dir)
    if args.verify:
        verify_or_raise(
            vision_client,
            step5.screenshot_path,
            build_verification_prompt("The first search result page has loaded and contains stock-related content."),
            step5,
        )
    step5.content = vision_client.describe(step5.screenshot_path, build_content_prompt())
    LOGGER.info("Extracted page content (%s characters).", len(step5.content or ""))
    add_step_artifact(artifacts, step5, output_dir)

    LOGGER.info("Step 6: switch from Safari to Notes.")
    quit_application("Safari")
    wait_for(args.post_action_delay)
    open_application("Notes")
    wait_for(args.app_launch_delay)
    activate_application("Notes")
    wait_for(args.post_action_delay)

    step6 = StepArtifacts(step_name="step6_notes_open")
    step6.screenshot_path = take_screenshot(step6.step_name, output_dir)
    if args.verify:
        verify_or_raise(
            vision_client,
            step6.screenshot_path,
            build_verification_prompt("The Notes app is open and ready to create a note."),
            step6,
        )
    add_step_artifact(artifacts, step6, output_dir)

    LOGGER.info("Step 7: locate and click New Note.")
    new_note_target = locate_client.locate_target(step6.screenshot_path, args.new_note_prompt)
    step7 = StepArtifacts(
        step_name="step7_new_note",
        screenshot_path=step6.screenshot_path,
        coordinates=new_note_target.click_point,
        bounding_box=new_note_target.bounding_box,
    )
    screen_x, screen_y = translate_coordinates_for_screen(step6.screenshot_path, new_note_target.click_point)
    step7.screen_coordinates = {"x": screen_x, "y": screen_y}
    create_new_note()
    wait_for(args.post_action_delay)
    add_step_artifact(artifacts, step7, output_dir)

    LOGGER.info("Step 8: paste extracted content into the note.")
    note_text = args.note_prefix + "\n" + (step5.content or "")
    paste_text(note_text)
    wait_for(args.post_action_delay)

    step8 = StepArtifacts(step_name="step8_note_written", content=note_text)
    step8.screenshot_path = take_screenshot(step8.step_name, output_dir)
    if args.verify:
        verify_or_raise(
            vision_client,
            step8.screenshot_path,
            build_verification_prompt("A new note is open and contains the extracted stock list."),
            step8,
        )
    add_step_artifact(artifacts, step8, output_dir)

    LOGGER.info("Workflow completed successfully.")
    return artifacts


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
