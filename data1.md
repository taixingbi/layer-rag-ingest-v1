# data1 — personal corpus runbook

Personal-style text (resume, Q&A, profile) under `data1/raw/*.txt`. Outputs land in `data1/processed/`.

**Data1** (plain text pipeline) — from repo root:

```bash
./scripts/data1.sh
```

Synthetic questions and smoke validation run by default; use **`RUN_SYNTHETIC_QUESTIONS=0`** and/or **`RUN_SMOKE_VALIDATE=0`** to skip stages.

## Prerequisites

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (see `env.example` and [README.md](README.md)). For ingest you need at least:

- `QDRANT_URL`
- `EMBEDDINGS_BASE_URL` (for upsert embedding)
- `COLLECTION_NAME` (for example `taixing_knowledge`)

Optional:

- `ENV=dev|qa|prod` → collection resolves to `<COLLECTION_NAME>_<env>`
- `INFERENCE_BASE_URL`, `CHAT_*` → only if you run `synthetic_questions.py`

## Layout

- Input: `data1/raw/*.txt`
- Chunks: `data1/processed/chunks_<stem>.json`
- Points: `data1/processed/points_<stem>.json`
- Summary: `data1/processed/ingest_prepare_summary.json`

Use **`--source-prefix personal`** so `payload.source` values look like `personal_<doc_type>` (easy to filter vs `repo_*` from `data2`).

## Full pipeline

Run from repo root:

```bash
python3 app/plain_text_chunks.py data1
python3 app/prepare_payloads.py \
  --data-dir data1/processed \
  --output-dir data1/processed \
  --pattern "chunks_*.json" \
  --source-prefix personal
```

### Optional: synthetic questions (chat API)

Regenerates `payload.synthetic_questions` and **`embed_text`** / **`embed_token_count`**. Re-run upsert afterward so vectors match the new embed text.

```bash
python3 app/synthetic_questions.py --data-dir data1/processed --questions-per-chunk 3
```

Failed-row audit (even when empty) is written under:

`data1/processed/reports/synthetic_questions_failed_<timestamp>_EST.json`

Useful flags: `--max-concurrency`, `--retry-max-attempts`, `--retry-base-delay`, `--failed-report-path`, `--dry-run`, `--skip-existing`.

### Upsert to Qdrant

Dry run (parse + validate; no write):

```bash
python3 app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json" --dry-run --skip-embedding
```

Embed + upsert:

```bash
python3 app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json"
```

Run smoke validation explicitly:

```bash
# warning-only default
python3 app/smoke_validate.py --data-dir data1/processed --pattern "points_*.json"

# strict mode
python3 app/smoke_validate.py --data-dir data1/processed --pattern "points_*.json" --strict
```

Useful flags: `--threshold`, `--max-probes`, `--report-path`.

Embedding gateway auth uses standard `--embedding-api-key` / `EMBEDDING_API_KEY` when required.

## Single-file chunking

If you only want one source:

```bash
python3 app/plain_text_chunks.py data1/raw/profile.txt data1/processed/chunks_profile.json
```

Then run `prepare_payloads.py` on `data1/processed` as above (it globs all `chunks_*.json`).

## Retrieval hint

Filter personal context by `payload.source` prefix `personal_` when querying the shared collection.
