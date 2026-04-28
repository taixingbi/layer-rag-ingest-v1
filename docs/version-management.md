# Version Management

This document defines practical version-management rules for the ingest pipeline.

## Scope

Versioning in this repo applies to:

- payload/data contract changes
- identity key contract changes
- script CLI behavior changes
- workflow/runbook changes

## Principles

- Prefer backward-compatible changes by default.
- Version contracts, not just code.
- Keep roll-forward and rollback paths explicit.
- Use dry-run checks before mutating operations.

## 1) Data Contract Versioning

When payload shape changes:

1. Add new fields as optional first.
2. Keep existing readers tolerant of missing new fields.
3. Only remove fields after docs + runbooks are updated and old artifacts are retired.

Recommended compatibility approach:

- writer: can emit new fields
- reader: must not fail when field is absent

## 2) Identity Key Versioning

Point IDs are deterministic and computed from canonical identity input.  
If canonical key format changes, treat it as a contract change.

Rules:

- preserve deterministic ID generation within a given contract
- document canonical key format in docs
- define cutover plan before changing identity logic

Current canonical identity input:

- `source`
- `document_id`
- `chunk_id`

## 3) CLI and Script Versioning

For `app/*.py` and `scripts/*.sh` changes:

- additive flags are preferred over breaking flag renames
- if behavior changes materially, document before/after examples
- keep defaults safe (`dry-run` where possible for destructive flows)

## 4) Workflow/Operations Versioning

For `.github/workflows` and operational commands:

- require explicit apply confirmations for mutating paths
- keep environment-gated approvals for apply jobs
- store artifacts for traceability (`manifest`, `stale_candidates`, `delete_actions`, `rollback_actions`)

## 5) Documentation Version Discipline

When behavior changes, update in same change set:

- `README.md`
- `docs/design.md`
- `docs/plan.md`
- runbook files (`data1.md`, `data2.md`)

Avoid “code says one thing, docs say another.”

## 6) Change Categories

Use this simple classification:

- **Patch**: bug fix, no contract change.
- **Minor**: additive fields/flags, backward compatible.
- **Major**: identity/schema/behavior changes requiring migration or cutover.

## 7) Release Checklist

Before applying to production-like collections:

1. Run syntax/lint checks.
2. Run ingest dry-run and reconcile dry-run.
3. Verify manifest output and stale preview.
4. Confirm rollback path (snapshot + action reports).
5. Update docs/runbooks.

## 8) Recommended Naming

For run artifacts:

- `ingest_manifest_<run_id>.json` (immutable run artifact)
- `ingest_manifest_latest.json` (operational pointer)
- `stale_candidates_<run_id>.json`
- `delete_actions_<run_id>.json`
- `rollback_actions_<run_id>.json`

## 9) Anti-Patterns

- changing identity key logic without a cutover plan
- mutating live data without dry-run preview
- removing fields immediately without compatibility window
- shipping behavior changes without runbook/doc updates
