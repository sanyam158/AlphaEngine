"""Application settings loaded from environment variables / ``.env`` (prefix ``ALPHAENGINE_``)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALPHAENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = Environment.DEV
    database_url: str = "sqlite:///./alphaengine.db"
    data_dir: Path = Path("./var/data")
    log_level: str = "INFO"
    log_json: bool = False

    # Notifications (Phase 6)
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # API auth (Phase 7)
    api_key: SecretStr | None = None

    # Research defaults
    initial_capital: float = Field(default=1_000_000.0, gt=0)

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {v!r}")
        return level

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
