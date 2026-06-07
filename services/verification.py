
"""
Mainly a verification loop that uses the vision client to verify the workflow steps.
"""
import logging
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from domain import WorkflowError
from workflow_examples.call_locateanything_rpc import encode_image_as_data_url

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DescribeRequest:
    screenshot_path: Path
    prompt: str


@dataclass(frozen=True)
class VerificationRequest:
    screenshot_path: Path
    prompt: str
    required_tokens: tuple[str, ...] = ("yes",)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    message: str


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


class VerificationService:
    def __init__(self, vision_client: VisionClient) -> None:
        self.vision_client = vision_client

    def describe(self, request: DescribeRequest) -> str:
        return self.vision_client.describe(request.screenshot_path, request.prompt)

    def verify(self, request: VerificationRequest) -> VerificationResult:
        message = self.vision_client.verify(request.screenshot_path, request.prompt)
        lowered = message.lower()
        passed = all(token in lowered for token in request.required_tokens)
        return VerificationResult(passed=passed, message=message)
