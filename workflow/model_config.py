"""Shared OpenAI-compatible model configuration for low-latency graph nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_REPAIR_MODEL = "qwen-flash"


def load_chat_model(
    *,
    model_env: str,
    default_model: str,
    model_override: str | None = None,
    timeout_env: str,
    default_timeout: float,
    retries_env: str,
    default_retries: int = 0,
    tokens_env: str | None = None,
    default_tokens: int | None = None,
    json_mode: bool = False,
):
    """Create a non-thinking ChatOpenAI client from shared credentials."""

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for model-backed nodes")

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model_override or os.getenv(model_env, default_model),
        "temperature": 0,
        "timeout": float(os.getenv(timeout_env, str(default_timeout))),
        "max_retries": int(os.getenv(retries_env, str(default_retries))),
        # Qwen and DeepSeek models on the configured Model Studio endpoint are
        # hybrid-thinking models. Structured workflow nodes should answer
        # directly unless a future task explicitly opts into reasoning.
        "extra_body": {"enable_thinking": False},
    }
    if tokens_env and default_tokens is not None:
        kwargs["max_tokens"] = max(
            256, int(os.getenv(tokens_env, str(default_tokens)))
        )
    if json_mode:
        kwargs["model_kwargs"] = {
            "response_format": {"type": "json_object"}
        }
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def load_json_repair_model():
    """Load the small structured-output fallback used after malformed JSON."""

    return load_chat_model(
        model_env="JSON_REPAIR_MODEL",
        default_model=DEFAULT_JSON_REPAIR_MODEL,
        timeout_env="JSON_REPAIR_TIMEOUT_SECONDS",
        default_timeout=20,
        retries_env="JSON_REPAIR_MAX_RETRIES",
        default_retries=0,
        tokens_env="JSON_REPAIR_MAX_TOKENS",
        default_tokens=4000,
        json_mode=True,
    )
