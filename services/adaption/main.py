"""FastAPI bridge that exposes the official Adaption SDK to the Next.js app.

The Adaption SDK is Python-only, so this service is the single place where the
Python runtime is required. The Next.js server never imports Python: it calls
these endpoints over loopback HTTP.

Credential handling: `ADAPTION_API_KEY` is read from the process environment or
the repository `.env` file and used only to build the SDK client. It is never
echoed back in a response, a log line, or an error message.

Run it with:

    npm run adaption:serve
    # or
    python services/adaption/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

# Resolve sibling modules both when run as a script and when imported by uvicorn.
_SERVICE_DIR = Path(__file__).resolve().parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

import adaption_client  # noqa: E402 - must follow the sys.path bootstrap
from fastapi import FastAPI, Query, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from settings import get_settings  # noqa: E402

DEFAULT_WAIT_SECONDS = 120.0
MAX_ROWS = 5_000

DOWNLOAD_MEDIA_TYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "parquet": "application/octet-stream",
}

app = FastAPI(
    title="AnimeVerse Studio - Adaption bridge",
    version="1.0.0",
    description=(
        "Loopback-only bridge that wraps the official `adaption` Python SDK for "
        "AnimeVerse Studio. It exists because the SDK is Python-only."
    ),
)


# --- request models --------------------------------------------------------


class InventRequest(BaseModel):
    """Body of POST /datasets/invent (already validated by Zod in Next.js)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=120)
    dataset_prompt: str = Field(min_length=40, max_length=12_000)
    rows: int = Field(ge=1, le=MAX_ROWS)
    training_type: Literal["instruction_dataset", "preference_pairs"] = "instruction_dataset"
    estimate: bool = False
    idempotency_key: str | None = Field(default=None, max_length=200)


class WaitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=DEFAULT_WAIT_SECONDS, ge=5, le=900)


class GenerateRequest(BaseModel):
    """Body of POST /responses - one synchronous text generation call."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, max_length=120)
    instructions: str = Field(min_length=1, max_length=20_000)
    input: str = Field(min_length=1, max_length=20_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=8_000)


# --- error handling --------------------------------------------------------


def _error_response(error: adaption_client.BridgeError) -> JSONResponse:
    body: dict[str, Any] = {"error": error.message, "kind": error.kind}
    if error.upstream_status is not None:
        body["upstreamStatus"] = error.upstream_status
    if error.detail:
        body["detail"] = error.detail
    if error.details:
        body["details"] = error.details
    return JSONResponse(status_code=error.http_status, content=body)


@app.exception_handler(adaption_client.BridgeError)
async def _bridge_error_handler(_request: Request, exc: adaption_client.BridgeError) -> JSONResponse:
    return _error_response(exc)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    details: dict[str, list[str]] = {}
    for error in exc.errors():
        parts = [str(part) for part in error.get("loc", ()) if part not in ("body", "query")]
        details.setdefault(".".join(parts) or "request", []).append(
            str(error.get("msg", "Invalid value"))
        )
    return _error_response(
        adaption_client.BridgeError(
            "invalid_request",
            "Some fields need attention.",
            http_status=400,
            details=details,
        )
    )


# --- endpoints -------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Reports configuration and SDK availability. Never returns the key."""
    settings = get_settings()
    return {
        "ok": True,
        "configured": settings.configured,
        "baseUrl": settings.base_url,
        "sdkPackage": adaption_client.SDK_PACKAGE,
        "sdkVersion": adaption_client.sdk_version(),
        "sdkImportError": adaption_client.sdk_import_error(),
        "detail": None if settings.configured else "ADAPTION_API_KEY is not set.",
    }


@app.post("/datasets/invent")
def invent(payload: InventRequest) -> dict[str, Any]:
    """Price (estimate=true) or start generation of a dataset from a brief."""
    return adaption_client.invent_dataset(
        name=payload.name,
        dataset_prompt=payload.dataset_prompt,
        rows=payload.rows,
        training_type=payload.training_type,
        estimate=payload.estimate,
        idempotency_key=payload.idempotency_key,
    )


@app.get("/datasets")
def list_datasets(
    limit: int = Query(default=20, ge=1, le=100),
    q: str | None = Query(default=None, max_length=200),
    status: str | None = Query(default=None, max_length=40),
) -> dict[str, Any]:
    return adaption_client.list_datasets(limit=limit, q=q, status=status)


@app.get("/datasets/{dataset_id}")
def dataset_status(dataset_id: str) -> dict[str, Any]:
    return adaption_client.dataset_status(dataset_id)


@app.post("/datasets/{dataset_id}/wait")
def dataset_wait(dataset_id: str, payload: WaitRequest | None = None) -> dict[str, Any]:
    timeout_seconds = payload.timeout_seconds if payload else DEFAULT_WAIT_SECONDS
    return adaption_client.wait_for_dataset(dataset_id, timeout_seconds=timeout_seconds)


@app.get("/datasets/{dataset_id}/download")
def dataset_download(
    dataset_id: str,
    file_format: Literal["csv", "json", "jsonl", "parquet"] = Query(default="jsonl"),
) -> Response:
    payload = adaption_client.download_dataset(dataset_id, file_format=file_format)
    return Response(
        content=payload,
        media_type=DOWNLOAD_MEDIA_TYPES[file_format],
        headers={
            "content-disposition": f'attachment; filename="adaption-{dataset_id}.{file_format}"'
        },
    )


@app.get("/models")
def models() -> dict[str, Any]:
    """The models this key may call, so a model id never has to be guessed."""
    return adaption_client.list_models()


@app.post("/responses")
def create_response(payload: GenerateRequest) -> dict[str, Any]:
    """One synchronous text generation call, via the SDK's Responses API."""
    return adaption_client.generate_text(
        model=payload.model,
        instructions=payload.instructions,
        input_text=payload.input,
        temperature=payload.temperature,
        max_output_tokens=payload.max_output_tokens,
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    print(
        f"[adaption-bridge] sdk={adaption_client.sdk_version()} "
        f"configured={settings.configured} base_url={settings.base_url}"
    )
    if not settings.configured:
        print(
            "[adaption-bridge] ADAPTION_API_KEY is not set: dataset requests will be "
            "rejected with HTTP 503 until it is added to .env."
        )
    print(f"[adaption-bridge] listening on http://{settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
