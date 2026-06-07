"""
Service for UI automation on macOS. 
"""
import json
import logging
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from domain import Point, WorkflowError

SCRIPT_DIR = ROOT_DIR / "workflow_examples"
MACOS_INPUT_SCRIPT = SCRIPT_DIR / "macos_input.swift"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenshotRequest:
    step_name: str
    output_dir: Path


@dataclass(frozen=True)
class DisplayGeometry:
    width_points: int
    height_points: int
    screenshot_width_pixels: int
    screenshot_height_pixels: int


class UIAutomationService:
    def run_command(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        LOGGER.debug("Running command: %s", args)
        return subprocess.run(args, check=check, text=True, capture_output=True)

    def run_applescript(self, script: str) -> None:
        result = self.run_command(["osascript", "-e", script], check=False)
        if result.returncode != 0:
            raise WorkflowError(result.stderr.strip() or "AppleScript command failed.")

    def take_screenshot(self, request: ScreenshotRequest) -> Path:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = request.output_dir / f"{request.step_name}.png"
        LOGGER.info("Taking screenshot for %s at %s", request.step_name, screenshot_path)
        result = self.run_command(["screencapture", "-x", str(screenshot_path)], check=False)
        if result.returncode != 0:
            raise WorkflowError(result.stderr.strip() or "Failed to capture screenshot.")
        return screenshot_path

    def get_display_geometry(self, screenshot_path: Path) -> DisplayGeometry:
        screenshot_width_pixels, screenshot_height_pixels = self._read_png_size(screenshot_path)
        result = self.run_command(
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
            screenshot_width_pixels=screenshot_width_pixels,
            screenshot_height_pixels=screenshot_height_pixels,
        )
        LOGGER.info(
            "Display geometry resolved: screenshot %spx x %spx, logical display %spt x %spt",
            geometry.screenshot_width_pixels,
            geometry.screenshot_height_pixels,
            geometry.width_points,
            geometry.height_points,
        )
        return geometry

    def translate_coordinates_for_screen(self, screenshot_path: Path, coordinates: Point) -> Point:
        geometry = self.get_display_geometry(screenshot_path)
        screen_x = round(coordinates.x * geometry.width_points / geometry.screenshot_width_pixels)
        screen_y = round(coordinates.y * geometry.height_points / geometry.screenshot_height_pixels)
        LOGGER.info(
            "Translated screenshot coordinates (%s, %s) to screen coordinates (%s, %s) using screenshot size %spx x %spx and screen size %spt x %spt",
            coordinates.x,
            coordinates.y,
            screen_x,
            screen_y,
            geometry.screenshot_width_pixels,
            geometry.screenshot_height_pixels,
            geometry.width_points,
            geometry.height_points,
        )
        return Point(x=screen_x, y=screen_y)

    def move_and_click(self, point: Point) -> None:
        LOGGER.info("Moving cursor and clicking at screen coordinates (%s, %s)", point.x, point.y)
        result = self.run_command(
            ["swift", str(MACOS_INPUT_SCRIPT), "move-click", str(point.x), str(point.y)],
            check=False,
        )
        if result.returncode != 0:
            raise WorkflowError(result.stderr.strip() or "Failed to move and click.")

    def press_return(self) -> None:
        LOGGER.info("Pressing Return.")
        script = """
        tell application "System Events"
            key code 36
        end tell
        """
        self.run_applescript(script)

    def paste_text(self, text: str) -> None:
        LOGGER.info("Pasting note text (%s characters).", len(text))
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        script = """
        tell application "System Events"
            keystroke "v" using {command down}
        end tell
        """
        self.run_applescript(script)

    def wait_for(self, seconds: float) -> None:
        LOGGER.debug("Waiting for %.2f seconds.", seconds)
        time.sleep(seconds)

    def _read_png_size(self, image_path: Path) -> tuple[int, int]:
        with image_path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            raise WorkflowError(f"Unsupported screenshot format for size detection: {image_path}")
        width, height = struct.unpack(">II", header[16:24])
        return width, height
