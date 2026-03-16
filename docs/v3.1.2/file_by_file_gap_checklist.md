# v3.1.2 File-by-File Gap Checklist

Current status after the first fact-vs-hypothesis, analysis-mode, outcome-evidence, durable-maturity, and ledger pass.

Status after the follow-up implementation pass:
- `closed` means the previously identified gap has been implemented in the current code.
- `kept` means the original requested behavior remains intentionally backward-compatible.

## `src/v3_1/world/blackboard.py`

- `implemented`
  - top-level observed/hypothesized stores exist
  - legacy combined stores still exist for backward-compatible reads
  - explicit `observed_view()`, `hypothesized_view()`, and `combined_view()` accessors now exist
- `kept`
  - legacy combined state is still materialized for backward compatibility

## `src/v3_1/world/merge.py`

- `implemented`
  - incoming rows are classified into observed vs hypothesized stores
  - provenance fields are stamped at merge time
  - factual rows now prefer explicit analyzer markers when present
  - rebuilt indexes expose evidence-tier row views for the main planner-facing structures

## `src/v3_1/planning/planner_service.py`

- `implemented`
  - observed-first / hypothesized-backfill prioritization exists in the service
  - hypothesis-seeded candidates receive trace fields and penalties
  - explicit planning preparation views are injected into belief for trace/debug visibility

## `src/v3_1/analysis/episode_analysis.py`

- `implemented`
  - `analysis_mode` is required and carried through summaries/deltas
  - probe vs `directed_outcome` now changes POI selection, consequence selection, topology selection, and priority metadata

## `src/v3_1/agents/analysis_worker.py`

- `implemented`
  - validates and forwards `analysis_mode`
- `open gap`
  - none in this file relative to the requested scope

## `src/v3_1/execution/outcomes.py`

- `implemented`
  - `outcome_evidence` block exists
  - summary labels are derived from that block
  - unknown / unavailable evidence now remains `None` instead of being forced to observed false

## `src/v3_1/execution/env_worker.py`

- `implemented`
  - per-step telemetry is richer and passed through raw execution artifacts
  - probe/directed telemetry now goes through one normalized helper
- `kept`
  - some fields remain execution-derived because the env does not expose them natively

## `src/v3_1/memory/reconcile.py`

- `implemented`
  - durable rows now carry maturity/evidence fields
  - evidence basis and support counts are now filled more explicitly per family
- `kept`
  - maturity is still computed in reconcile, as requested, rather than in storage

## `src/v3_1/agents/memory_agent.py`

- `implemented`
  - durable flush gating uses maturity/evidence fields
  - durable eligibility is enforced before request construction

## `src/v3_1/storage/persistent_memory.py`

- `implemented`
  - new maturity/evidence columns are persisted
  - storage validates presence instead of recomputing them
  - read paths preserve compatibility with the added columns

## `src/v3_1/runtime/session_ledger.py`

- `implemented`
  - append-only typed ledger record exists
  - append helpers exist for all requested event types
  - event payload dataclasses now exist for round, plan, execution, analysis, merge, reconcile, flush, and stop events

## `src/v3_1/runtime/round_runner.py`

- `implemented`
  - ledger events are appended after the authoritative stage transitions in the current multi-trial loop
  - stage payloads are now normalized through typed event payloads

## `src/v3_1/runtime/orchestrator.py`

- `implemented`
  - run-scoped ledger ownership exists
  - flush/stop events are appended at orchestrator level
  - flush and stop payloads are now normalized through typed event payloads

## `src/v3_1/runtime/postrun_exports.py`

- `implemented`
  - ledger is consulted first for chronology/linkage/version-transition views
  - ledger is now also persisted as its own post-run artifact
- `kept`
  - `episodes` and `round_records` remain compatibility sources for heatmaps and memory summaries where the ledger intentionally does not yet duplicate those payloads
