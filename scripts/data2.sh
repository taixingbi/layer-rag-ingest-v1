#!/usr/bin/env bash
# data2 pipeline: GitHub tree -> raw txt -> markdown chunks -> (default synthetic Q) -> Qdrant upsert -> smoke validate
# Run from anywhere; repo root is the parent of this scripts/ directory.
# Synthetic questions run by default unless RUN_SYNTHETIC_QUESTIONS=0 (skips chat API).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATA_ENV="${DATA_ENV:-dev}"
DATA_ROOT="${DATA_ROOT:-data_${DATA_ENV}}"
DATASET_ROOT="${DATA_ROOT}/data2"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "error: dataset root not found: $DATASET_ROOT"
  echo "set DATA_ENV=dev|qa|prod (or DATA_ROOT) to select the environment data folder."
  exit 1
fi

echo "==> data2 (${DATA_ENV}): download GitHub trees from repo list"
"$PYTHON" app/github_tree_to_txt.py --repo-list "$DATASET_ROOT/raw/repo.txt" --out-dir "$DATASET_ROOT/raw"

echo "==> data2 (${DATA_ENV}): chunk Markdown -> processed/chunks_*.json"
"$PYTHON" app/markdown_to_chunks.py "$DATASET_ROOT"

echo "==> data2 (${DATA_ENV}): chunks -> points (source prefix: repo)"
"$PYTHON" app/prepare_payloads.py \
  --data-dir "$DATASET_ROOT/processed" \
  --output-dir "$DATASET_ROOT/processed" \
  --pattern "chunks_*.json" \
  --source-prefix repo \
  --access-control-file "$DATASET_ROOT/raw/access_control.json"

if [[ "${RUN_SYNTHETIC_QUESTIONS:-1}" == "0" ]]; then
  echo "==> data2 (${DATA_ENV}): skipping synthetic questions (RUN_SYNTHETIC_QUESTIONS=0)"
else
  echo "==> data2 (${DATA_ENV}): synthetic questions (requires chat / inference env)"
  "$PYTHON" app/synthetic_questions.py --data-dir "$DATASET_ROOT/processed" --questions-per-chunk 3
fi

echo "==> data2 (${DATA_ENV}): embed + upsert to Qdrant"
"$PYTHON" app/upsert_qdrant.py --data-dir "$DATASET_ROOT/processed" --pattern "points_*.json"

if [[ "${RUN_SMOKE_VALIDATE:-1}" == "0" ]]; then
  echo "==> data2 (${DATA_ENV}): skipping smoke validation (RUN_SMOKE_VALIDATE=0)"
else
  if [[ "${RUN_SMOKE_JUDGE:-1}" == "0" ]]; then
    echo "==> data2 (${DATA_ENV}): post-upsert smoke validation (judge disabled via RUN_SMOKE_JUDGE=0)"
    "$PYTHON" app/smoke_validate.py --data-dir "$DATASET_ROOT/processed" --pattern "points_*.json"
  else
    echo "==> data2 (${DATA_ENV}): post-upsert smoke validation (LLM judge enabled)"
    "$PYTHON" app/smoke_validate.py \
      --data-dir "$DATASET_ROOT/processed" \
      --pattern "points_*.json" \
      --judge-enabled \
      --judge-rescue-floor "${SMOKE_JUDGE_RESCUE_FLOOR:-0.58}"
  fi
fi

if [[ "${RUN_RECONCILE:-0}" == "1" ]]; then
  echo "==> data2 (${DATA_ENV}): lifecycle reconcile"
  RECONCILE_MANIFEST="${RECONCILE_MANIFEST:-$DATASET_ROOT/processed/ingest_manifest_latest.json}"
  if [[ "${RECONCILE_APPLY_SOFT_DELETE:-0}" == "1" ]]; then
    "$PYTHON" app/reconcile_qdrant.py \
      --manifest-path "$RECONCILE_MANIFEST" \
      --scope-key "${RECONCILE_SCOPE_KEY:-collection}" \
      --scope-value "${RECONCILE_SCOPE_VALUE:-}" \
      --delete-mode soft \
      --apply-soft-delete
  else
    "$PYTHON" app/reconcile_qdrant.py \
      --manifest-path "$RECONCILE_MANIFEST" \
      --scope-key "${RECONCILE_SCOPE_KEY:-collection}" \
      --scope-value "${RECONCILE_SCOPE_VALUE:-}" \
      --dry-run
  fi
fi

echo "==> data2 (${DATA_ENV}) pipeline finished"
