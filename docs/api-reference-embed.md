# API reference: chat completions and embeddings

One-shot examples against local HTTP endpoints (``/v1/chat/completions``, ``/v1/embeddings``).

## Chat completions (inference)

```bash
curl http://192.168.86.179:30180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "where is jersey city"}
    ],
    "max_tokens": 50
  }'
```

## Embeddings

```bash
curl -X POST http://192.168.86.179:30181/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":"hello world"}'
```

## Python wrappers (`app/`)

Use these from the repo root (or add the repo to `PYTHONPATH`). `base_url` is the service root **without** `/v1` (the clients append `/v1/chat/completions` and `/v1/embeddings`).

Defaults: set `INFERENCE_BASE_URL` and `EMBEDDINGS_BASE_URL` in `.env` (see `env.example`), or rely on built-in fallbacks in `app/client_inference.py` and `app/client_embeddings.py`. Omit `base_url` on calls to use those defaults; pass `base_url=` to override.

### Chat completions — `app/client_inference.py`

```python
from app.client_inference import chat_completions, chat_completion_text

# Full response JSON (base_url optional — uses INFERENCE_BASE_URL from .env or code fallback)
out = chat_completions(
    messages=[{"role": "user", "content": "where is jersey city"}],
    model="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=50,
)

# Assistant text only
text = chat_completion_text(
    user_content="where is jersey city",
    model="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=50,
)
```

Async equivalents:

```python
from app.client_inference import async_chat_completions, async_chat_completion_text

out = await async_chat_completions(
    messages=[{"role": "user", "content": "where is jersey city"}],
    model="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=50,
)

text = await async_chat_completion_text(
    user_content="where is jersey city",
    model="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=50,
)
```

### Embeddings — `app/client_embeddings.py`

```python
from app.client_embeddings import embed_text, embed_texts, embeddings

vec = embed_text(model="BAAI/bge-m3", text="hello world")

vecs = embed_texts(model="BAAI/bge-m3", texts=["hello", "world"])

raw = embeddings(model="BAAI/bge-m3", input_text="hello world")
```

Optional: pass `api_key="..."`, `timeout=60.0`, `client=httpx.Client(...)`, or `extra_headers={...}` on either module for auth or custom gateways.
