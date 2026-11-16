"""Environment resolution for the Adaption bridge service.

The bridge is the only process that talks to the Adaption API, so it is the
only place the credential is read. Nothing sensitive is ever returned over
HTTP: callers can learn *whether* a key is present, never what it is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = REPO_ROOT / ".env"

DEFAULT_BASE_URL = "https://api.prod.adaptionlabs.ai"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_TIMEOUT_MS = 120_000
DEFAULT_MAX_RETRIES = 2

# Same sentinel the Next.js layer uses: "let the bridge pick a model".
AUTO_MODEL_SENTINEL = "auto"


def load_env_file(path: Path = ENV_FILE_PATH) -> None:
    """Minimal `.env` reader so the service needs no extra tooling.

    Values already present in the process environment always win, so an
    exported `ADAPTION_API_KEY` is never replaced by a placeholder from `.env`.
    """
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def read_int(name: str, fallback: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        parsed = int(float(raw))
    except ValueError:
        return fallback
    return max(minimum, min(maximum, parsed))


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str | None
    base_url: str
    timeout_ms: int
    max_retries: int
    host: str
    port: int

    @property
    def configured(self) -> bool:
        """True when a credential is available. The key itself is never logged."""
        return bool(self.api_key)

    @property
    def timeout_seconds(self) -> float:
        return self.timeout_ms / 1000


def get_settings() -> Settings:
    load_env_file()
    api_key = os.environ.get("ADAPTION_API_KEY", "").strip()
    # The bridge reads the same `.env` the Next.js layer does, so a creator's
    # ADAPTION_MODEL choice still applies even if the web process is running
    # with an older environment and sends no model id. "auto" (or unset) means
    # "resolve a model from the account".
    raw_model = os.environ.get("ADAPTION_MODEL", "").strip()
    model = None if raw_model in ("", AUTO_MODEL_SENTINEL) else raw_model
    return Settings(
        api_key=api_key or None,
        model=model,
        base_url=os.environ.get("ADAPTION_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        timeout_ms=read_int("ADAPTION_TIMEOUT_MS", DEFAULT_TIMEOUT_MS, 5_000, 900_000),
        max_retries=read_int("ADAPTION_MAX_RETRIES", DEFAULT_MAX_RETRIES, 0, 5),
        host=os.environ.get("ADAPTION_SERVICE_HOST", "").strip() or DEFAULT_HOST,
        port=read_int("ADAPTION_SERVICE_PORT", DEFAULT_PORT, 1024, 65_535),
    )
