"""
FastAPI server for the LocateAnything model.
"""
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from model.locateanything import LocateAnythingService


class InferenceRequest(BaseModel):
    screenshot_base64: str = Field(..., description="Base64-encoded screenshot, optionally as a data URL.")
    prompt: str = Field(..., description="Natural-language description of what should be located.")
    output_type: Literal["box", "point"] = "box"
    generation_mode: Literal["fast", "slow", "hybrid"] = "hybrid"
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="LocateAnything Model Server", version="1.0.0")
service = LocateAnythingService()


def rpc_result(request_id: str | int | None, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def run_inference(request: InferenceRequest) -> dict[str, object]:
    return service.locate(
        screenshot_base64=request.screenshot_base64,
        prompt=request.prompt,
        output_type=request.output_type,
        generation_mode=request.generation_mode,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": "locateanything", "status": "ok"}


@app.get("/health/model")
async def model_health() -> dict[str, str]:
    return {"service": "locateanything", "component": "model", "status": "ok"}


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
async def infer(request: InferenceRequest) -> dict[str, object]:
    return run_inference(request)


@app.post("/rpc")
async def rpc(request: JsonRpcRequest) -> dict[str, Any]:
    if request.method != "locate":
        return rpc_error(request.id, -32601, f"Method not found: {request.method}")

    try:
        params = InferenceRequest.model_validate(request.params)
    except Exception as exc:
        return rpc_error(request.id, -32602, f"Invalid params: {exc}")

    try:
        result = run_inference(params)
    except Exception as exc:
        return rpc_error(request.id, -32000, f"Inference failed: {exc}")

    return rpc_result(request.id, result)
