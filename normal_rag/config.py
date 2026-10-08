from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "openai/gpt-4o-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    provider: str = ""
    temperature: float = 0.1
    max_tokens: int = 256
    timeout_seconds: float = 60.0
    ollama_url: str = "http://127.0.0.1:11434/api/embeddings"
    embedding_timeout_seconds: float = 30.0

    def __post_init__(self):
        if not self.api_key or self.api_key in ("your-key-here", "sk-..."):
            raise ValueError("Set OPENAI_API_KEY for the shared benchmark or OPENROUTER_API_KEY for the standalone pilot")
        if not self.model or not self.provider:
            raise ValueError("Set the model and OPENROUTER_PROVIDER in the selected runner's settings")
        if any(c in self.api_key for c in "\r\n"):
            raise ValueError("OPENROUTER_API_KEY must be a single line")
        remote = urlsplit(self.base_url)
        if (remote.scheme != "https" or not remote.hostname or remote.username
                or remote.password or remote.query or remote.fragment):
            raise ValueError("OPENROUTER_BASE_URL must be an HTTPS base URL without credentials or query")
        local = urlsplit(self.ollama_url)
        if (local.scheme not in ("http", "https")
                or local.hostname not in ("localhost", "127.0.0.1", "::1")
                or local.username or local.password or local.query or local.fragment):
            raise ValueError("OLLAMA_URL must point to a local loopback server")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("MAX_TOKENS must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.timeout_seconds, self.embedding_timeout_seconds)):
            raise ValueError("Timeouts must be positive finite seconds")

    def public(self) -> dict:
        return {
            "model": self.model, "base_url": self.base_url,
            "provider": self.provider, "allow_fallbacks": False,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds, "ollama_url": self.ollama_url,
            "embedding_model": "nomic-embed-text",
            "embedding_timeout_seconds": self.embedding_timeout_seconds,
            "top_k": 4, "retries": 0,
        }


def load_settings(path: Path) -> Settings:
    """Read the standalone pilot's .env; shared runs use rule-chain-depth settings."""
    if not path.is_file():
        raise ValueError("Create the standalone pilot's .env from .env.example and configure the key and provider")
    values = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or key in values:
            raise ValueError(f"Invalid or duplicate .env setting on line {line_number}")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed .env quote on line {line_number}")
            value = value[1:-1]
        values[key] = value
    try:
        return Settings(
            api_key=values.get("OPENROUTER_API_KEY", ""),
            model=values.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=values.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            provider=values.get("OPENROUTER_PROVIDER", ""),
            temperature=float(values.get("TEMPERATURE", "0.1")),
            max_tokens=int(values.get("MAX_TOKENS", "256")),
            timeout_seconds=float(values.get("TIMEOUT_SECONDS", "60")),
            ollama_url=values.get("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings"),
            embedding_timeout_seconds=float(values.get("EMBEDDING_TIMEOUT_SECONDS", "30")),
        )
    except (ValueError, OverflowError) as exc:
        # Numeric conversion exceptions can contain the input; never echo it.
        if str(exc).startswith(("could not convert", "invalid literal")):
            raise ValueError("A numeric setting in .env is invalid") from None
        raise
