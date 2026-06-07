
"""
Service for the LocateAnything model. It takes in the screenshot and prompt and returns the bounding box or point coordinates.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from domain import BoundingBox, LocateTarget, WorkflowError
from workflow_examples.call_locateanything_rpc import build_payload, call_rpc, encode_image_as_data_url

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocateRequest:
    screenshot_path: Path
    prompt: str


class LocatorService:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def locate(self, request: LocateRequest) -> LocateTarget:
        LOGGER.info("Locating bounding box for prompt: %s", request.prompt)
        payload = build_payload(
            screenshot_base64=encode_image_as_data_url(request.screenshot_path),
            prompt=request.prompt,
            output_type="box",
        )
        response = call_rpc(self.endpoint, payload)
        coordinates = response.get("result", {}).get("coordinates", [])
        if not coordinates:
            raise WorkflowError(f"No coordinates returned for prompt: {request.prompt}")

        raw_box = coordinates[0]
        if not all(key in raw_box for key in ("x1", "y1", "x2", "y2")):
            raise WorkflowError(f"Bounding box response was malformed for prompt {request.prompt}: {raw_box}")

        box = BoundingBox(
            x1=int(raw_box["x1"]),
            y1=int(raw_box["y1"]),
            x2=int(raw_box["x2"]),
            y2=int(raw_box["y2"]),
        )
        click_point = box.center()
        LOGGER.info("Locator returned bounding box: %s", raw_box)
        LOGGER.info("Using bounding box center as click point: %s", click_point)
        return LocateTarget(bounding_box=box, click_point=click_point)
