# Lifecycle Operations Runbook

This runbook covers lifecycle maintenance commands for:

- reconcile (stale detection)
- soft-delete apply
- hard purge (retention-gated)
- rollback preview/apply

All operations should follow **dry-run first**.

## Prerequisites

- env file configured (`.env.dev`, `.env.qa`, or `.env.prod`; copy to `.env` when needed)
- latest manifest exists under `data_<env>/data1/processed/ingest_manifest_latest.json`

For manual commands in this doc, use:

```bash
export DATA_ROOT=data_dev
# export DATA_ROOT=data_qa
# export DATA_ROOT=data_prod
```

## 1) Reconcile dry-run (no mutation)

Use this to preview stale IDs in scope.

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path "${DATA_ROOT}/data1/processed/ingest_manifest_latest.json" \
  --scope-key collection \
  --dry-run
```

Output artifacts:

- `stale_candidates_<run_id>.json`
- `delete_actions_<run_id>.json` (action summary; zero applies in dry-run)

## 2) Apply soft-delete for stale points

Only run after reviewing dry-run output.

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path "${DATA_ROOT}/data1/processed/ingest_manifest_latest.json" \
  --scope-key collection \
  --delete-mode soft \
  --apply-soft-delete
```

Notes:

- updates payload lifecycle fields on stale points
- does not hard-delete vectors immediately

## 3) Hard purge tombstones by retention

Use after retention window has passed.

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path "${DATA_ROOT}/data1/processed/ingest_manifest_latest.json" \
  --scope-key collection \
  --delete-mode hard \
  --retention-days 30 \
  --apply-hard-delete
```

Recommended flow:

1. run with `--dry-run` first
2. verify hard-delete candidate count
3. rerun with `--apply-hard-delete`

## 4) Scope-specific operations

To limit changes by source or doc type:

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path "${DATA_ROOT}/data1/processed/ingest_manifest_latest.json" \
  --scope-key source \
  --scope-value personal_profile \
  --dry-run
```

Supported scope keys:

- `collection`
- `source`
- `doc_type`

## 5) Rollback preview and apply

Rollback restores lifecycle status to a target ingest run within scope.

Preview:

```bash
python3 app/rollback_ingest_run.py \
  --target-run-id run_20260425_180000_EST \
  --manifest-dir "${DATA_ROOT}/data1/processed" \
  --scope-key collection \
  --dry-run
```

Apply:

```bash
python3 app/rollback_ingest_run.py \
  --target-run-id run_20260425_180000_EST \
  --manifest-dir "${DATA_ROOT}/data1/processed" \
  --scope-key collection \
  --apply
```

Output artifact:

- `rollback_actions_<run_id>.json`

## 6) Safety checklist

- run preview (`--dry-run`) before all mutating operations
- verify scope (`collection/source/doc_type`) and manifest path
- keep reports/artifacts for audit trail
- snapshot collection before large cleanup waves
- avoid apply actions during active ingest jobs

## 7) GitHub Actions mapping

Repo workflows for lifecycle operations:

- `.github/workflows/lifecycle-reconcile.yml`
  - nightly dry-run reconcile
  - manual soft-delete apply (approval + confirm input)
- `.github/workflows/lifecycle-phase2.yml`
  - manual purge dry-run/apply
  - manual rollback dry-run/apply

Use workflows when you need approval-gated, auditable operations in CI.
