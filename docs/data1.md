# data1 — personal corpus runbook

Personal-style text (resume, Q&A, profile) under `data_<env>/data1/raw/*.txt`. Outputs land in `data_<env>/data1/processed/`.

**Data1** (plain text pipeline) — from repo root:

```bash
./scripts/data1.sh dev
./scripts/data1.sh qa
./scripts/data1.sh prod
```

Synthetic questions and smoke validation run by default; use **`RUN_SYNTHETIC_QUESTIONS=0`** and/or **`RUN_SMOKE_VALIDATE=0`** to skip stages.

## Prerequisites

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create environment file(s) (see `env.example` and [README.md](README.md)):

- `.env.dev`
- `.env.qa`
- `.env.prod`

For ingest you need at least:

- `QDRANT_URL`
- `EMBEDDINGS_BASE_URL` (for upsert embedding)
- `COLLECTION_NAME` (for example `taixing_knowledge`)

Optional:

- `ENV=dev|qa|prod` → collection resolves to `<COLLECTION_NAME>_<env>`
- `INFERENCE_BASE_URL`, `CHAT_*` → only if you run `synthetic_questions.py`

## Layout

- Input: `data_<env>/data1/raw/*.txt`
- Chunks: `data_<env>/data1/processed/chunks_<stem>.json`
- Points: `data_<env>/data1/processed/points_<stem>.json`
- Summary: `data_<env>/data1/processed/ingest_prepare_summary.json`
- Manifest: `data_<env>/data1/processed/ingest_manifest_<ingest_run_id>.json` and `data_<env>/data1/processed/ingest_manifest_latest.json`

For manual commands in this doc, use:

```bash
export DATA_ROOT=data_dev
# export DATA_ROOT=data_qa
# export DATA_ROOT=data_prod
```

Path shorthand in remaining examples: `data1/...` means `"${DATA_ROOT}/data1/..."`.

ID contract (default v3): point id is UUID5 of  
`v3|source=<source>|document_id=<document_id>|document_version=<document_version>|chunk_version=<chunk_version>|embedding_version=<embedding_version>|chunk_id=<chunk_id>`.

Compatibility: pass `--id-key-version v2` to keep the old identity key  
`v2|source=<source>|document_id=<document_id>|chunk_id=<chunk_id>`.

Use **`--source-prefix personal`** so `payload.source` values look like `personal_<doc_type>` (easy to filter vs `repo_*` from `data2`).

## Full pipeline

Run from repo root:

```bash
python3 app/plain_text_chunks.py "${DATA_ROOT}/data1"
python3 app/prepare_payloads.py \
  --data-dir "${DATA_ROOT}/data1/processed" \
  --output-dir "${DATA_ROOT}/data1/processed" \
  --pattern "chunks_*.json" \
  --source-prefix personal
```

Optional explicit version tags (recommended for immutable coexistence):

```bash
python3 app/prepare_payloads.py \
  --data-dir data1/processed \
  --output-dir data1/processed \
  --pattern "chunks_*.json" \
  --source-prefix personal \
  --document-version "2026-04-29" \
  --chunk-version "plain_text_v1" \
  --embedding-version "bge-m3@1024"
```

Optional role mapping (adds `payload.profile.role` only for mapped sources/documents):

```bash
python3 app/prepare_payloads.py \
  --data-dir data1/processed \
  --output-dir data1/processed \
  --pattern "chunks_*.json" \
  --source-prefix personal \
  --profile-role-map '{"personal_profile":"AI Infrastructure Engineer"}'
```

You can also load mappings from a file:

```bash
python3 app/prepare_payloads.py \
  --data-dir data1/processed \
  --output-dir data1/processed \
  --pattern "chunks_*.json" \
  --source-prefix personal \
  --profile-role-map-file data1/processed/profile_roles.json
```

Key lookup order for role resolution:

- `source:document_id` (most specific)
- `source`
- `document_id` (fallback)
- default literal `role` when no mapping key matches

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

## Lifecycle reconcile / purge / rollback

```bash
# preview stale candidates (no mutation)
python3 app/reconcile_qdrant.py \
  --manifest-path data1/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --dry-run

# apply soft-delete for stale points
python3 app/reconcile_qdrant.py \
  --manifest-path data1/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --delete-mode soft \
  --apply-soft-delete

# purge old tombstones
python3 app/reconcile_qdrant.py \
  --manifest-path data1/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --delete-mode hard \
  --retention-days 30 \
  --apply-hard-delete

# rollback to one ingest run
python3 app/rollback_ingest_run.py \
  --target-run-id run_20260425_180000_EST \
  --manifest-dir data1/processed \
  --scope-key collection \
  --dry-run
```

## Single-file chunking

If you only want one source:

```bash
python3 app/plain_text_chunks.py data1/raw/profile.txt data1/processed/chunks_profile.json
```

Then run `prepare_payloads.py` on `data1/processed` as above (it globs all `chunks_*.json`).

## Retrieval hint

Filter personal context by `payload.source` prefix `personal_` when querying the shared collection.

## Gold dataset export

Generate gold QA JSONL from all env folders (`data_dev`, `data_qa`, `data_prod`). For per-env split files only (no single consolidated file next to `data_dev/`):

```bash
python3 app/rag_gold_eval/generate_gold_dataset.py \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset
```

Or run with defaults (writes `gold_dataset.jsonl` in the current directory plus splits beside it); see `docs/gold-dataset.md`.

To evaluate those rows against the live RAG API, use `app/rag_gold_eval/run_eval.py` (see the **RAG evaluation** section in `docs/gold-dataset.md`).

Each output row is one question-answer pair derived from `points_*.json`:
- `question` from `payload.synthetic_questions[]`
- `answer` from `payload.text`
- standard metadata: `env`, `source_file`, `id`, `source`, `doc_type`, `section`, `chunk_id`, `text`
