from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List


def _get_bool(env: str, default: bool) -> bool:
    val = os.getenv(env, "").strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_list(env: str, default: List[str]) -> List[str]:
    raw = os.getenv(env)
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


@dataclass(frozen=True)
class Settings:
    # General
    app_name: str = os.getenv("APP_NAME", "Hip Torque Analysis API")
    environment: str = os.getenv("ENV", os.getenv("ENVIRONMENT", "production"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = _get_bool("DEBUG", False)

    # CORS
    allowed_origins: List[str] = tuple(_get_list("ALLOWED_ORIGINS", ["*"]))  # type: ignore[assignment]
    allow_credentials: bool = _get_bool("ALLOW_CREDENTIALS", True)
    allowed_methods: List[str] = tuple(_get_list("ALLOWED_METHODS", ["GET", "POST", "OPTIONS"]))  # type: ignore[assignment]
    allowed_headers: List[str] = tuple(_get_list("ALLOWED_HEADERS", ["*"]))  # type: ignore[assignment]
    # Host header protection
    allowed_hosts: List[str] = tuple(_get_list("ALLOWED_HOSTS", ["*"]))  # type: ignore[assignment]

    # Performance / limits
    gzip_min_size: int = int(os.getenv("GZIP_MIN_SIZE", "1024"))  # bytes
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))

    # API surface
    openapi_enabled: bool = _get_bool("OPENAPI_ENABLED", False)
    docs_enabled: bool = _get_bool("DOCS_ENABLED", False)


settings = Settings()
