#!/usr/bin/env bash
# data2 pipeline: GitHub tree -> raw txt -> markdown chunks -> (default synthetic Q) -> Qdrant upsert -> smoke validate
# Run from anywhere; repo root is the parent of this scripts/ directory.
# Synthetic questions run by default unless RUN_SYNTHETIC_QUESTIONS=0 (skips chat API).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> data2: download GitHub trees from repo list"
"$PYTHON" app/github_tree_to_txt.py --repo-list data2/raw/repo.txt --out-dir data2/raw

echo "==> data2: chunk Markdown -> processed/chunks_*.json"
"$PYTHON" app/markdown_to_chunks.py data2

echo "==> data2: chunks -> points (source prefix: repo)"
"$PYTHON" app/prepare_payloads.py \
  --data-dir data2/processed \
  --output-dir data2/processed \
  --pattern "chunks_*.json" \
  --source-prefix repo

if [[ "${RUN_SYNTHETIC_QUESTIONS:-1}" == "0" ]]; then
  echo "==> data2: skipping synthetic questions (RUN_SYNTHETIC_QUESTIONS=0)"
else
  echo "==> data2: synthetic questions (requires chat / inference env)"
  "$PYTHON" app/synthetic_questions.py --data-dir data2/processed --questions-per-chunk 3
fi

echo "==> data2: embed + upsert to Qdrant"
"$PYTHON" app/upsert_qdrant.py --data-dir data2/processed --pattern "points_*.json"

if [[ "${RUN_SMOKE_VALIDATE:-1}" == "0" ]]; then
  echo "==> data2: skipping smoke validation (RUN_SMOKE_VALIDATE=0)"
else
  if [[ "${RUN_SMOKE_JUDGE:-1}" == "0" ]]; then
    echo "==> data2: post-upsert smoke validation (judge disabled via RUN_SMOKE_JUDGE=0)"
    "$PYTHON" app/smoke_validate.py --data-dir data2/processed --pattern "points_*.json"
  else
    echo "==> data2: post-upsert smoke validation (LLM judge enabled)"
    "$PYTHON" app/smoke_validate.py \
      --data-dir data2/processed \
      --pattern "points_*.json" \
      --judge-enabled \
      --judge-rescue-floor "${SMOKE_JUDGE_RESCUE_FLOOR:-0.58}"
  fi
fi

if [[ "${RUN_RECONCILE:-0}" == "1" ]]; then
  echo "==> data2: lifecycle reconcile"
  RECONCILE_MANIFEST="${RECONCILE_MANIFEST:-data2/processed/ingest_manifest_latest.json}"
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

echo "==> data2 pipeline finished"
