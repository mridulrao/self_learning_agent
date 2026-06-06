import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib import error, request

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_config import load_env_file


load_env_file()
COORDINATE_LOCATOR_ENDPOINT = os.environ.get(
    "COORDINATE_LOCATOR_ENDPOINT",
    "https://xmjgo3r2cn7lcj-8000.proxy.runpod.net/rpc",
)


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(screenshot_base64: str, prompt: str, output_type: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "locate-request-1",
        "method": "locate",
        "params": {
            "screenshot_base64": screenshot_base64,
            "prompt": prompt,
            "output_type": output_type,
            "generation_mode": "hybrid",
            "max_new_tokens": 128,
            "temperature": 0.0,
        },
    }


def call_rpc(endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )

    with request.urlopen(http_request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the LocateAnything Runpod RPC endpoint.")
    parser.add_argument("image", type=Path, help="Path to the screenshot image.")
    parser.add_argument("prompt", help="Description of what to locate in the screenshot.")
    parser.add_argument("--endpoint", default=COORDINATE_LOCATOR_ENDPOINT, help="Full /rpc endpoint URL.")
    parser.add_argument("--output-type", choices=["box", "point"], default="box")
    args = parser.parse_args()

    screenshot_base64 = encode_image_as_data_url(args.image)
    payload = build_payload(screenshot_base64, args.prompt, args.output_type)

    try:
        response = call_rpc(args.endpoint, payload)
    except error.HTTPError as exc:
        print(exc.read().decode("utf-8"))
        raise SystemExit(f"Request failed with HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    print(json.dumps(response, indent=2))

    if "result" in response and "coordinates" in response["result"]:
        print("\nCoordinates:")
        print(json.dumps(response["result"]["coordinates"], indent=2))


if __name__ == "__main__":
    main()
