# data2 — GitHub / Markdown docs runbook

Repo docs: list GitHub `/tree/` URLs, download as text exports, chunk with Markdown-aware splitter. Outputs land in `data2/processed/`.

**Data2** (GitHub / markdown pipeline) — from repo root:

```bash
./scripts/data2.sh
```

Synthetic questions and smoke validation run by default; use **`RUN_SYNTHETIC_QUESTIONS=0`** and/or **`RUN_SMOKE_VALIDATE=0`** to skip stages.

## Prerequisites

Same as [data1.md](data1.md): venv, `pip install -r requirements.txt`, `.env` with `QDRANT_URL`, `EMBEDDINGS_BASE_URL`, `COLLECTION_NAME`.

For GitHub downloads, a token helps rate limits:

- `GITHUB_TOKEN` or `GH_TOKEN` in `.env`

For optional synthetic questions:

- `INFERENCE_BASE_URL` and/or `CHAT_*` vars (see [README.md](README.md))

## Layout

- URL list: `data2/raw/repo.txt` (one GitHub tree URL per line)
- Raw exports: `data2/raw/*.txt` (from `github_tree_to_txt.py`, or your chosen `--out-dir`)
- Chunks: `data2/processed/chunks_<stem>.json`
- Points: `data2/processed/points_<stem>.json`

Use **`--source-prefix repo`** so `payload.source` values look like `repo_<doc_type>` (filter vs `personal_*` from `data1`).

## Full pipeline

Run from repo root:

```bash
python3 app/github_tree_to_txt.py --repo-list data2/raw/repo.txt --out-dir data2/raw
python3 app/markdown_to_chunks.py data2
python3 app/prepare_payloads.py \
  --data-dir data2/processed \
  --output-dir data2/processed \
  --pattern "chunks_*.json" \
  --source-prefix repo
```

### Optional: synthetic questions (chat API)

```bash
python3 app/synthetic_questions.py --data-dir data2/processed --questions-per-chunk 3
```

Re-run **`upsert_qdrant.py`** after this if you already upserted old vectors (embed text changed).

Audit file directory:

`data2/processed/reports/synthetic_questions_failed_<timestamp>_EST.json`

Chunk size tuning on `markdown_to_chunks.py`: `--min-chunk-chars`, `--max-chunk-chars`.

### Upsert to Qdrant

Dry run:

```bash
python3 app/upsert_qdrant.py --data-dir data2/processed --pattern "points_*.json" --dry-run --skip-embedding
```

Embed + upsert:

```bash
python3 app/upsert_qdrant.py --data-dir data2/processed --pattern "points_*.json"
```

Run smoke validation explicitly:

```bash
# warning-only default
python3 app/smoke_validate.py --data-dir data2/processed --pattern "points_*.json"

# strict mode
python3 app/smoke_validate.py --data-dir data2/processed --pattern "points_*.json" --strict
```

Useful flags: `--threshold`, `--max-probes`, `--report-path`.

## Single-file chunking

Example:

```bash
python3 app/markdown_to_chunks.py \
  data2/raw/layer-gateway-embed-v1__design.md.txt \
  data2/processed/chunks_layer-gateway-embed-v1__design.md.json
```

Then run `prepare_payloads.py` on `data2/processed` as above.

## Do not use for data2

`plain_text_chunks.py` is for plain prose-style text. GitHub-exported `.txt` from this repo still contains Markdown — use **`markdown_to_chunks.py`** for `data2`.

## Retrieval hint

Filter repository context by `payload.source` prefix `repo_` when querying the shared collection.
