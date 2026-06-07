"""
Service for the LocateAnything model. It takes in the screenshot and prompt and returns the bounding box or point coordinates.
"""
import base64
import io
import math
import os
import re
import sys
from typing import Any
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

def _load_env_config() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "env_config.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            from env_config import load_env_file

            load_env_file()
            return


_load_env_config()


class LocateAnythingService:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.environ.get("MODEL_DIR", "/models/LocateAnything-3B")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.max_image_edge = int(os.environ.get("MAX_IMAGE_EDGE", "1600"))
        self.max_image_pixels = int(os.environ.get("MAX_IMAGE_PIXELS", "2560000"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device).eval()

    def locate(
        self,
        screenshot_base64: str,
        prompt: str,
        output_type: str = "box",
        generation_mode: str = "hybrid",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        original_image = self._decode_image(screenshot_base64)
        original_width, original_height = original_image.size
        image, resize_metadata = self._resize_image_for_inference(original_image)
        question = self._build_prompt(prompt, output_type)
        raw_answer = self._predict(
            image=image,
            question=question,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if output_type == "point":
            coordinates = self._parse_points(raw_answer, original_width, original_height)
        else:
            coordinates = self._parse_boxes(raw_answer, original_width, original_height)

        return {
            "coordinates": coordinates,
            "resized_relatively": resize_metadata["resized_relatively"],
            "original_size": resize_metadata["original_size"],
            "inference_size": resize_metadata["inference_size"],
            "resize_scale": resize_metadata["resize_scale"],
        }

    def _predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        image_grid_hws = inputs.get("image_grid_hws")

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )

        if isinstance(response, tuple):
            return str(response[0])
        return str(response)

    @staticmethod
    def _build_prompt(prompt: str, output_type: str) -> str:
        if output_type == "point":
            return f"Point to: {prompt}."
        return f"Locate the region that matches the following description: {prompt}."

    @staticmethod
    def _decode_image(screenshot_base64: str) -> Image.Image:
        payload = screenshot_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _resize_image_for_inference(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        width, height = image.size
        scale = min(
            1.0,
            self.max_image_edge / max(width, height),
            math.sqrt(self.max_image_pixels / float(width * height)),
        )
        if scale >= 1.0:
            return image, {
                "resized_relatively": False,
                "original_size": {"width": width, "height": height},
                "inference_size": {"width": width, "height": height},
                "resize_scale": 1.0,
            }

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, {
            "resized_relatively": True,
            "original_size": {"width": width, "height": height},
            "inference_size": {"width": new_width, "height": new_height},
            "resize_scale": scale,
        }

    @staticmethod
    def _parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict[str, int]]:
        boxes: list[dict[str, int]] = []
        for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = (int(value) for value in match.groups())
            boxes.append(
                {
                    "x1": round(x1 / 1000 * image_width),
                    "y1": round(y1 / 1000 * image_height),
                    "x2": round(x2 / 1000 * image_width),
                    "y2": round(y2 / 1000 * image_height),
                }
            )
        return boxes

    @staticmethod
    def _parse_points(answer: str, image_width: int, image_height: int) -> list[dict[str, int]]:
        points: list[dict[str, int]] = []
        for match in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = (int(value) for value in match.groups())
            points.append(
                {
                    "x": round(x / 1000 * image_width),
                    "y": round(y / 1000 * image_height),
                }
            )
        return points
