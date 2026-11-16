"""Thin wrapper around the official `adaption` Python SDK.

One function per SDK call, so the HTTP layer in `main.py` stays a plain
request/response mapper. Nothing is simulated: when Adaption is unreachable or
unconfigured, the call fails and the failure is reported with a
machine-readable `kind` that the Next.js layer maps onto an HTTP status.

Verified against adaption 0.19.0:
  client.datasets.invent(...)                   -> DatasetInventResponse
  client.datasets.list(...)                     -> SyncCursor[DatasetListResponse]
  client.datasets.get_status(dataset_id)        -> DatasetGetStatusResponse
  client.datasets.wait_for_completion(...)      -> DatasetGetStatusResponse
  client.datasets.download(id, file_format=...) -> BinaryAPIResponse (bytes)
  client.models.list(type="hosted")             -> ModelListResponse
  client.responses.create(model, input, ...)    -> Response (has .output_text)
"""

from __future__ import annotations

import functools
from importlib import metadata
from typing import Any, Callable, TypeVar

from settings import get_settings

SDK_PACKAGE = "adaption"

_T = TypeVar("_T")

_sdk: Any | None = None
_sdk_import_error: str | None = None


class BridgeError(Exception):
    """A failure the HTTP layer knows how to translate into a response."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        http_status: int,
        upstream_status: int | None = None,
        detail: str | None = None,
        details: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.http_status = http_status
        self.upstream_status = upstream_status
        self.detail = detail
        self.details = details


def _load_sdk() -> Any | None:
    """Import the SDK once, recording the failure instead of crashing.

    The service must be able to start (and explain itself over `/health`) even
    when `adaption` is not installed in the active Python environment.
    """
    global _sdk, _sdk_import_error
    if _sdk is not None or _sdk_import_error is not None:
        return _sdk
    try:
        import adaption
    except Exception as exc:  # pragma: no cover - depends on the host environment
        _sdk_import_error = f"{type(exc).__name__}: {exc}"
        return None
    _sdk = adaption
    return _sdk


def sdk_import_error() -> str | None:
    _load_sdk()
    return _sdk_import_error


def sdk_version() -> str | None:
    if _load_sdk() is None:
        return None
    try:
        return metadata.version(SDK_PACKAGE)
    except metadata.PackageNotFoundError:  # pragma: no cover - unusual installs
        return getattr(_sdk, "__version__", None)


def _dump(value: Any) -> Any:
    """Convert SDK pydantic models into plain JSON-serialisable data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value


def _upstream_detail(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value is not None:
                return str(value)[:500]
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return None


def _classify(exc: BaseException) -> BridgeError:
    adaption = _load_sdk()
    if adaption is None:
        return BridgeError(
            "sdk_missing",
            "The `adaption` Python SDK is not available in the bridge's Python environment.",
            http_status=503,
            detail=_sdk_import_error,
        )

    detail = _upstream_detail(exc)

    if isinstance(exc, adaption.APITimeoutError):
        return BridgeError(
            "timeout",
            "Adaption did not respond before the request timeout.",
            http_status=504,
            detail=detail,
        )
    if isinstance(exc, adaption.APIConnectionError):
        return BridgeError(
            "network_error",
            "Could not reach the Adaption API.",
            http_status=502,
            detail=detail,
        )
    if isinstance(exc, adaption.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 401:
            return BridgeError(
                "authentication_failed",
                "Adaption rejected the configured API key (401). Check ADAPTION_API_KEY.",
                http_status=401,
                upstream_status=401,
                detail=detail,
            )
        if status == 403:
            return BridgeError(
                "permission_denied",
                "The Adaption API key is not permitted to perform this request (403).",
                http_status=403,
                upstream_status=403,
                detail=detail,
            )
        if status == 404:
            return BridgeError(
                "not_found",
                "Adaption has no resource with that id (404).",
                http_status=404,
                upstream_status=404,
                detail=detail,
            )
        if status in (400, 422):
            return BridgeError(
                "invalid_request",
                "Adaption rejected the request as invalid.",
                http_status=400,
                upstream_status=status,
                detail=detail,
            )
        if status == 409:
            return BridgeError(
                "conflict",
                "Adaption reported a conflict for this request (409).",
                http_status=409,
                upstream_status=409,
                detail=detail,
            )
        if status == 429:
            return BridgeError(
                "rate_limited",
                "The Adaption rate limit was reached. Wait, then retry.",
                http_status=429,
                upstream_status=429,
                detail=detail,
            )
        return BridgeError(
            "upstream_error",
            "Adaption returned an unsuccessful response.",
            http_status=502,
            upstream_status=status,
            detail=detail,
        )
    if isinstance(exc, TimeoutError):
        # The SDK's polling helpers raise DatasetTimeout / PollTimeout here.
        return BridgeError(
            "timeout",
            "Adaption did not finish within the requested wait budget.",
            http_status=504,
            detail=str(exc) or None,
        )
    if isinstance(exc, adaption.AdaptionError):
        return BridgeError(
            "upstream_error",
            "The Adaption SDK reported an error.",
            http_status=502,
            detail=detail or str(exc),
        )
    return BridgeError(
        "upstream_error",
        "Unexpected failure while talking to Adaption.",
        http_status=502,
        detail=f"{type(exc).__name__}: {exc}",
    )


def _call(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except BridgeError:
        raise
    except BaseException as exc:
        raise _classify(exc) from exc


@functools.lru_cache(maxsize=4)
def _build_client(api_key: str, base_url: str, timeout_seconds: float, max_retries: int) -> Any:
    # The official SDK entry point, imported exactly as the docs show.
    from adaption import Adaption

    return Adaption(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _get_client() -> Any:
    if _load_sdk() is None:
        raise BridgeError(
            "sdk_missing",
            "The `adaption` Python SDK is not available in the bridge's Python environment.",
            http_status=503,
            detail=_sdk_import_error,
        )

    settings = get_settings()
    if settings.api_key is None:
        raise BridgeError(
            "not_configured",
            "ADAPTION_API_KEY is not set, so no Adaption request can be made.",
            http_status=503,
            detail="Add ADAPTION_API_KEY to the repo .env file (see .env.example), then restart the bridge.",
        )

    try:
        return _build_client(
            settings.api_key,
            settings.base_url,
            settings.timeout_seconds,
            settings.max_retries,
        )
    except Exception as exc:
        raise _classify(exc) from exc


def invent_dataset(
    *,
    name: str,
    dataset_prompt: str,
    rows: int,
    training_type: str,
    estimate: bool,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create (or price) a generated dataset from a free-text brief."""
    client = _get_client()

    options: dict[str, Any] = {
        "name": name,
        "dataset_prompt": dataset_prompt,
        "rows": rows,
        "training_type": training_type,
    }
    if estimate:
        # Omitted rather than sent as False: the API treats absence as "generate".
        options["estimate"] = True
    if idempotency_key:
        options["idempotency_key"] = idempotency_key

    return _dump(_call(lambda: client.datasets.invent(**options)))


def list_datasets(
    *,
    limit: int = 20,
    q: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    client = _get_client()

    options: dict[str, Any] = {"limit": limit}
    if q:
        options["q"] = q
    if status:
        options["status"] = status

    page = _call(lambda: client.datasets.list(**options))
    return {
        "datasets": [_dump(item) for item in page.datasets],
        "nextCursor": getattr(page, "next_cursor", None),
    }


def dataset_status(dataset_id: str) -> dict[str, Any]:
    client = _get_client()
    return _dump(_call(lambda: client.datasets.get_status(dataset_id)))


def wait_for_dataset(dataset_id: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Poll using the SDK helper until the dataset reaches a terminal status."""
    client = _get_client()
    return _dump(
        _call(lambda: client.datasets.wait_for_completion(dataset_id, timeout=timeout_seconds))
    )


def download_dataset(dataset_id: str, *, file_format: str) -> bytes:
    client = _get_client()
    response = _call(lambda: client.datasets.download(dataset_id, file_format=file_format))
    return response.read()


def list_models() -> dict[str, Any]:
    """The models this key may call. Used to resolve a model instead of guessing."""
    client = _get_client()
    page = _call(lambda: client.models.list(type="hosted"))
    models = [_dump(item) for item in page.data]
    if not models:
        # Some accounts have no "hosted" models; fall back to the unfiltered list.
        page = _call(lambda: client.models.list())
        models = [_dump(item) for item in page.data]
    return {"models": models}


def _resolve_model() -> str:
    """First available model id. The id comes from Adaption, never invented."""
    for model in list_models()["models"]:
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id.strip():
            return model_id.strip()
    raise BridgeError(
        "no_model_available",
        "Adaption reported no available model for text generation.",
        http_status=502,
        detail="Check GET /models on the bridge, or set ADAPTION_MODEL explicitly.",
    )


def _incomplete_reason(response: Any) -> str | None:
    """The API's own reason when a response finished without producing text."""
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, dict):
        reason = details.get("reason")
    else:
        reason = getattr(details, "reason", None)
    return reason if isinstance(reason, str) and reason else None


def generate_text(
    *,
    model: str | None,
    instructions: str,
    input_text: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """One synchronous text generation call via the SDK's Responses API."""
    client = _get_client()

    options: dict[str, Any] = {
        # Caller wins; otherwise the bridge's own ADAPTION_MODEL; otherwise ask
        # Adaption for one of the account's models.
        "model": model or get_settings().model or _resolve_model(),
        "instructions": instructions,
        "input": input_text,
    }
    if temperature is not None:
        options["temperature"] = temperature
    if max_output_tokens is not None:
        options["max_output_tokens"] = max_output_tokens

    response = _call(lambda: client.responses.create(**options))
    text = (getattr(response, "output_text", None) or "").strip()
    if not text:
        status = getattr(response, "status", None)
        reason = _incomplete_reason(response)
        detail = f"status={status}"
        if reason:
            detail += f", reason={reason}"
        if reason == "max_output_tokens":
            detail += (
                ". The model spent its entire output budget before writing an "
                "answer (reasoning models do this). Set ADAPTION_MODEL to a "
                "non-reasoning model, or raise max_output_tokens."
            )
        raise BridgeError(
            "empty_response",
            "Adaption returned an empty text response.",
            http_status=502,
            detail=detail,
        )

    return {
        "text": text,
        "model": getattr(response, "model", None),
        "id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
    }
