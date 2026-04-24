"""HTTP client for chat inference (``/v1/chat/completions`` — vLLM, local gateways, etc.)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv()

# Default inference service root (no ``/v1`` suffix). Set ``INFERENCE_BASE_URL`` in ``.env`` or pass ``base_url=``.
_INFERENCE_FALLBACK = "http://192.168.86.179:30180"
INFERENCE_BASE_URL = (
    (os.getenv("INFERENCE_BASE_URL") or _INFERENCE_FALLBACK).strip().rstrip("/")
)
DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def normalize_chat_base_url(url: str) -> str:
    """Accept ``http://host`` or ``http://host/v1``; return root without ``/v1``."""
    u = url.rstrip("/")
    if u.endswith("/v1"):
        return u[:-3].rstrip("/")
    return u


def chat_completions(
    *,
    messages: list[dict[str, Any]],
    base_url: str = INFERENCE_BASE_URL,
    model: str = DEFAULT_CHAT_MODEL,
    max_tokens: int | None = 50,
    temperature: float | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
    extra_headers: dict[str, str] | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    POST /v1/chat/completions.

    ``base_url`` is the API root (default: ``INFERENCE_BASE_URL``), not including ``/v1``.
    """
    url = f"{normalize_chat_base_url(base_url)}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format

    def _do(c: httpx.Client) -> dict[str, Any]:
        r = c.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    if client is not None:
        return _do(client)
    with httpx.Client() as c:
        return _do(c)


def chat_completion_text(
    *,
    user_content: str,
    base_url: str = INFERENCE_BASE_URL,
    model: str = DEFAULT_CHAT_MODEL,
    system_content: str | None = None,
    max_tokens: int | None = 50,
    **kwargs: Any,
) -> str:
    """One-shot user message; returns assistant message content string."""
    messages: list[dict[str, Any]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    data = chat_completions(
        messages=messages,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected chat response shape: {data!r}") from e
