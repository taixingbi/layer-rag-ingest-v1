"""HTTP client for ``/v1/embeddings`` (local embedding gateways, etc.)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv()

# Default embeddings service root (no ``/v1`` suffix). Set ``EMBEDDINGS_BASE_URL`` in ``.env`` or pass ``base_url=``.
_EMBEDDINGS_FALLBACK = "http://192.168.86.179:30181"
EMBEDDINGS_BASE_URL = (
    (os.getenv("EMBEDDINGS_BASE_URL") or _EMBEDDINGS_FALLBACK).strip().rstrip("/")
)


def _order_embeddings(items: list[dict[str, Any]], n: int) -> list[list[float]]:
    if len(items) != n:
        return [e["embedding"] for e in sorted(items, key=lambda x: x["index"])]
    try:
        ordered: list[list[float] | None] = [None] * n
        for e in items:
            ordered[e["index"]] = e["embedding"]
        if any(x is None for x in ordered):
            raise ValueError
        return ordered  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError, IndexError):
        return [e["embedding"] for e in sorted(items, key=lambda x: x["index"])]


def embeddings(
    *,
    base_url: str = EMBEDDINGS_BASE_URL,
    model: str,
    input_text: str | list[str],
    api_key: str | None = None,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    POST /v1/embeddings.

    ``base_url`` is the API root (default: ``EMBEDDINGS_BASE_URL``), not including ``/v1``.
    """
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    if isinstance(input_text, str):
        payload: dict[str, Any] = {"model": model, "input": input_text}
    else:
        if not input_text:
            return {"data": [], "model": model}
        payload = {"model": model, "input": input_text}

    def _do(c: httpx.Client) -> dict[str, Any]:
        r = c.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    if client is not None:
        data = _do(client)
    else:
        with httpx.Client() as c:
            data = _do(c)

    items = data.get("data")
    if not isinstance(items, list):
        raise ValueError(f"Unexpected embeddings response: {data!r}")
    if len(items) > 1:
        data["data"] = sorted(items, key=lambda x: int(x.get("index", 0)))
    return data


def embed_text(
    *,
    base_url: str = EMBEDDINGS_BASE_URL,
    model: str,
    text: str,
    **kwargs: Any,
) -> list[float]:
    """Embed a single string; returns the embedding vector."""
    data = embeddings(base_url=base_url, model=model, input_text=text, **kwargs)
    items = data["data"]
    if len(items) != 1:
        raise ValueError(f"Expected one embedding, got {len(items)}")
    return list(items[0]["embedding"])


def embed_texts(
    *,
    base_url: str = EMBEDDINGS_BASE_URL,
    model: str,
    texts: list[str],
    **kwargs: Any,
) -> list[list[float]]:
    """Embed multiple strings; returns vectors in input order."""
    data = embeddings(base_url=base_url, model=model, input_text=texts, **kwargs)
    items = data["data"]
    return _order_embeddings(items, len(texts))
