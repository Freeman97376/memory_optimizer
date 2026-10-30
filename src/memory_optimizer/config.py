"""Explicit configuration; startup never loads unrelated provider credentials."""

import os
import re
from dataclasses import dataclass, field

from pydantic import SecretStr


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: SecretStr = field(repr=False)
    schema: str
    pool_size: int


@dataclass(frozen=True)
class Settings:
    database_url: SecretStr
    api_key: SecretStr | None = field(default=None, repr=False)
    schema: str = "memory_optimizer"
    allow_local_anonymous: bool = False
    max_body_bytes: int = 32 * 1024 * 1024
    chunk_chars: int = 3000
    chunk_overlap: int = 200
    pool_size: int = 4

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", self.schema):
            raise ValueError("invalid database schema name")
        if not 1 <= self.chunk_chars <= 16_000:
            raise ValueError("chunk_chars must be between 1 and 16000")
        if not 0 <= self.chunk_overlap < self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        if self.max_body_bytes <= 0 or not 1 <= self.pool_size <= 32:
            raise ValueError("invalid body or connection pool limit")
        if not self.database_url.get_secret_value():
            raise ValueError("MEMORY_DATABASE_URL is required")
        if self.api_key is not None and not self.api_key.get_secret_value():
            raise ValueError("API key cannot be empty")

    def validate_service(self):
        if self.api_key is None and not self.allow_local_anonymous:
            raise ValueError("MEMORY_API_KEY is required unless local anonymous mode is explicit")

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(self.database_url, self.schema, self.pool_size)

    @classmethod
    def from_env(cls):
        database_url = os.environ.get("MEMORY_DATABASE_URL")
        if not database_url:
            raise ValueError("MEMORY_DATABASE_URL is required")
        api_key = os.environ.get("MEMORY_API_KEY")
        local = os.environ.get("MEMORY_ALLOW_LOCAL_ANONYMOUS", "0")
        if local not in {"0", "1"}:
            raise ValueError("MEMORY_ALLOW_LOCAL_ANONYMOUS must be 0 or 1")
        return cls(
            database_url=SecretStr(database_url),
            api_key=SecretStr(api_key) if api_key else None,
            schema=os.environ.get("MEMORY_DATABASE_SCHEMA", "memory_optimizer"),
            allow_local_anonymous=local == "1",
            max_body_bytes=int(os.environ.get("MEMORY_MAX_BODY_BYTES", 32 * 1024 * 1024)),
        )
