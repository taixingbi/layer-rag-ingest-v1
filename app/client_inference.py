"""HTTP client for chat inference (``/v1/chat/completions`` — vLLM, local gateways, etc.)."""

from __future__ import annotations

import asyncio
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


def _build_headers(
    *,
    api_key: str | None,
    extra_headers: dict[str, str] | None,
) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _build_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float | None,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


async def async_chat_completions(
    *,
    messages: list[dict[str, Any]],
    base_url: str = INFERENCE_BASE_URL,
    model: str = DEFAULT_CHAT_MODEL,
    max_tokens: int | None = 50,
    temperature: float | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    client: httpx.AsyncClient | None = None,
    extra_headers: dict[str, str] | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Async POST /v1/chat/completions."""
    url = f"{normalize_chat_base_url(base_url)}/v1/chat/completions"
    headers = _build_headers(api_key=api_key, extra_headers=extra_headers)
    payload = _build_payload(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )

    async def _do(c: httpx.AsyncClient) -> dict[str, Any]:
        r = await c.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    if client is not None:
        return await _do(client)
    async with httpx.AsyncClient() as c:
        return await _do(c)


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
    if client is not None:
        url = f"{normalize_chat_base_url(base_url)}/v1/chat/completions"
        headers = _build_headers(api_key=api_key, extra_headers=extra_headers)
        payload = _build_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        r = client.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            async_chat_completions(
                messages=messages,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
                timeout=timeout,
                extra_headers=extra_headers,
                response_format=response_format,
            )
        )
    raise RuntimeError(
        "chat_completions() cannot be used inside an active event loop without "
        "a sync client; use async_chat_completions() instead."
    )


async def async_chat_completion_text(
    *,
    user_content: str,
    base_url: str = INFERENCE_BASE_URL,
    model: str = DEFAULT_CHAT_MODEL,
    system_content: str | None = None,
    max_tokens: int | None = 50,
    **kwargs: Any,
) -> str:
    """Async one-shot user message; returns assistant message content string."""
    messages: list[dict[str, Any]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    data = await async_chat_completions(
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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            async_chat_completion_text(
                user_content=user_content,
                base_url=base_url,
                model=model,
                system_content=system_content,
                max_tokens=max_tokens,
                **kwargs,
            )
        )
    raise RuntimeError(
        "chat_completion_text() cannot be used inside an active event loop; "
        "use async_chat_completion_text() instead."
    )
