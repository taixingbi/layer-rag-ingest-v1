#!/usr/bin/env bash
# data1 pipeline: plain text -> chunks -> points -> (optional synthetic Q) -> Qdrant upsert
# Run from anywhere; repo root is the parent of this scripts/ directory.
# Synthetic questions run by default unless RUN_SYNTHETIC_QUESTIONS=0 (skips chat API).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> data1: chunk raw/*.txt -> processed/chunks_*.json"
"$PYTHON" app/plain_text_chunks.py data1

echo "==> data1: chunks -> points (source prefix: personal)"
"$PYTHON" app/prepare_payloads.py \
  --data-dir data1/processed \
  --output-dir data1/processed \
  --pattern "chunks_*.json" \
  --source-prefix personal

if [[ "${RUN_SYNTHETIC_QUESTIONS:-1}" == "0" ]]; then
  echo "==> data1: skipping synthetic questions (RUN_SYNTHETIC_QUESTIONS=0)"
else
  echo "==> data1: synthetic questions (requires chat / inference env)"
  "$PYTHON" app/synthetic_questions.py --data-dir data1/processed --questions-per-chunk 3
fi

echo "==> data1: embed + upsert to Qdrant"
"$PYTHON" app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json"

echo "==> data1 pipeline finished"
