# v3.1.3 Gaps Addressed And Current Implementation

This document describes how the previously identified partial implementations were addressed and what the current code does now. It is a current-state description, not a redesign.

Code remains authoritative where this document is summarizing behavior.

## Scope

The changes covered here are in:

- `src/v3_1/world/blackboard.py`
- `src/v3_1/world/merge.py`
- `src/v3_1/planning/planner_service.py`
- `src/v3_1/analysis/episode_analysis.py`
- `src/v3_1/agents/analysis_worker.py`
- `src/v3_1/execution/outcomes.py`
- `src/v3_1/execution/env_worker.py`
- `src/v3_1/memory/reconcile.py`
- `src/v3_1/agents/memory_agent.py`
- `src/v3_1/storage/persistent_memory.py`
- `src/v3_1/runtime/session_ledger.py`
- `src/v3_1/runtime/round_runner.py`
- `src/v3_1/runtime/orchestrator.py`
- `src/v3_1/runtime/postrun_exports.py`

## 1. Blackboard fact-vs-hypothesis split

### Gap that existed

The blackboard had only combined stores like `entities`, `consequences`, `trigger_zones`, `topology_nodes`, and `topology_edges`. That meant observed facts and inferred rows were mixed in one state surface.

### How it was addressed

`src/v3_1/world/blackboard.py` now initializes separate top-level stores:

- `observed_entities`
- `hypothesized_entities`
- `observed_consequences`
- `hypothesized_consequences`
- `observed_trigger_zones`
- `hypothesized_trigger_zones`
- `observed_topology`
- `hypothesized_topology`

It also now exposes explicit read helpers:

- `BlackboardState.observed_view()`
- `BlackboardState.hypothesized_view()`
- `BlackboardState.combined_view()`

### Current implementation details

Backward compatibility is still preserved:

- legacy combined stores still exist in `state`
- merge rebuilds those combined stores from the split stores
- snapshots still expose the combined `state` and `indexes`

So the strict split exists internally, but callers that still use legacy combined fields continue to work.

## 2. Merge-time classification and provenance

### Gap that existed

Merge previously accepted incoming rows directly into combined stores and did not force explicit evidence tiering or provenance fields.

### How it was addressed

`src/v3_1/world/merge.py` now classifies every incoming row before merge.

Current classification path:

- rows are normalized through `_classify_row(...)`
- rows are stamped with:
  - `evidence_tier`
  - `source_stage`
  - `source_pass_id`
  - `source_episode_id`
  - `inference_method`
  - `confidence`

### Current classification rule

Observed rows are only produced when the row is treated as a direct fact:

- explicit `factual_observation`
- or direct observation style inference methods such as `direct_observation`

Otherwise rows are merged into hypothesized stores.

The merge path does not mutate a hypothesized row into an observed row in place. Instead:

- observed and hypothesized stores remain separate
- combined views are rebuilt afterward

### Current implementation details

Merged stores are rebuilt in this order:

1. areas
2. observed entities
3. hypothesized entities
4. observed consequences
5. hypothesized consequences
6. observed topology
7. hypothesized topology
8. trigger proposals and merge into observed/hypothesized trigger stores
9. legacy combined views
10. indexes

`build_indexes(...)` is still used for the legacy combined view, then `merge.py` enriches the resulting planner-facing index rows with `evidence_tier`.

## 3. Planner observed-first preparation

### Gap that existed

The planner still consumed the old combined blackboard shape, even after the blackboard split existed. That meant observed-first behavior was only implicit.

### How it was addressed

`src/v3_1/planning/planner_service.py` now creates a prioritized planning blackboard before belief building.

Current behavior:

- observed entities, consequences, trigger zones, and topology override hypothesized rows in the planner-facing combined snapshot
- hypothesized rows remain available as backfill

### Candidate seed evidence handling

Candidates are now annotated with:

- `seed_evidence_tier`
- `seed_source_count`
- `seed_is_fallback_from_hypothesis`

The planner also adds a hypothesized-only penalty when reranking scored rows if the seed support came only from hypothesis state.

### Current implementation details

The planner injects explicit planning-preparation sections into belief:

- `planning_input_priority`
- `planning_observed_state`
- `planning_hypothesized_backfill`

This does not replace the existing belief-builder output. It adds planner-visible context on top of it.

## 4. Analysis mode split

### Gap that existed

`analyze_episode(...)` had one shared flow with no explicit contract difference between probe analysis and directed-outcome analysis.

### How it was addressed

`src/v3_1/analysis/episode_analysis.py` now requires:

- `analysis_mode="probe"`
- or `analysis_mode="directed_outcome"`

`src/v3_1/agents/analysis_worker.py` validates and forwards the mode. Missing or invalid mode raises an error.

### Current mode behavior

Shared low-level utilities are still reused:

- observation normalization
- observation summaries
- area assignment
- avatar tracking
- motion summarization
- POI detection

The high-level contract now branches in mode-specific selectors:

- `_mode_select_pois(...)`
- `_mode_select_consequences(...)`
- `_mode_select_topology(...)`

Probe mode currently prioritizes:

- broader POI recall
- topology growth
- general discovery retention

Directed outcome mode currently prioritizes:

- result-bearing consequences
- contact/effect-related POIs
- route-progress/topology relevant to directed outcome

### Current implementation details

The returned `AnalyzedEpisode` now carries `analysis_mode` in:

- `summary`
- `metadata`
- each `step_row`

The produced `BlackboardDelta` now carries `analysis_mode` and provenance on the rows it emits.

## 5. Outcome evidence refactor

### Gap that existed

Outcome summarization mixed raw inference logic with final labels such as blocked/stalled/success, and several facts were inferred ad hoc.

### How it was addressed

`src/v3_1/execution/outcomes.py` now constructs a separate `outcome_evidence` block first, then derives summary labels from it.

Current `outcome_evidence` fields:

- `objective_contact_observed`
- `target_presence_observed`
- `avatar_target_distance_before`
- `avatar_target_distance_after`
- `route_progress_observed`
- `terminal_action_executed`
- `terminal_reward_observed`
- `done_observed`
- `blocked_by_boundary_observed`
- `blocked_by_unavailable_action_observed`
- `effect_region_observed`
- `effect_changed_cells_observed`
- `success_certainty`

### Current implementation details

The summarizer now keeps uncertainty lower fidelity where telemetry does not prove a fact:

- some fields are `None` when evidence is absent
- certainty is accumulated conservatively
- success labels are still derived, but now from one structured evidence block instead of scattered logic

This does not make the env oracle-like. It only centralizes the evidence handling.

## 6. Richer env-worker telemetry

### Gap that existed

The env worker was not emitting enough raw execution telemetry to support a stronger typed outcome-evidence layer.

### How it was addressed

`src/v3_1/execution/env_worker.py` now records richer per-step telemetry in `RawStep.info`.

Current fields emitted where available:

- chosen action
- action availability at decision time
- avatar cell before
- avatar cell after
- target cell before
- target cell after
- boundary hit
- invalid move
- reward observed
- done observed
- truncated observed
- terminal stop reason
- route instruction id
- terminal action marker
- effect region
- effect changed cells

### Current implementation details

Probe and directed execution now both use `_step_telemetry(...)` so the shape is normalized.

Some fields remain execution-derived rather than env-native:

- `boundary_hit`
- `route_instruction_id`
- `terminal_action_marker`
- sometimes target cells

That is because the current env adapters do not expose all of those natively.

## 7. Durable-memory maturity typing

### Gap that existed

Durable rows were family-shaped and carried only broad support metadata. Maturity/evidence quality was not explicit enough.

### How it was addressed

`src/v3_1/memory/reconcile.py` now annotates durable rows with:

- `mechanic_type`
- `maturity_stage`
- `evidence_basis`
- `observed_support_count`
- `hypothesis_support_count`
- `contradiction_count`
- `cross_round_stability`
- `last_evidence_tier`

Current stages:

- `speculative`
- `repeatable`
- `stable`
- `durable_ready`

### Current implementation details

The maturity model is still computed in reconcile. Storage does not recompute it.

Evidence basis is now more explicit per family:

- directed execution rows use directed-supported basis
- probe-only rows remain probe-only
- world-derived priors use observed-world or hypothesis-world basis
- mechanic hypotheses are marked mixed inference

This makes later flush gating stricter without changing tactical/session learning.

## 8. Durable flush eligibility tightening

### Gap that existed

Durable flush gating was not yet fully based on maturity and evidence basis.

### How it was addressed

`src/v3_1/agents/memory_agent.py` now filters flush rows before constructing the persistent flush request.

Current durable eligibility requires:

- `maturity_stage == "durable_ready"`
- `observed_support_count > 0`
- `contradiction_count <= 0`
- `cross_round_stability >= 2`
- `evidence_basis != "probe_only"`

### Current implementation details

If all rows are filtered out, `build_flush_request(...)` returns `None`.

This keeps:

- tactical session learning active during probe/direct execution
- durable writes restricted to mature, observed-backed rows

## 9. Persistent storage schema extension

### Gap that existed

SQLite persistence did not yet store the new maturity/evidence fields.

### How it was addressed

`src/v3_1/storage/persistent_memory.py` now:

- adds the new maturity/evidence columns compatibly
- validates durable rows before writing
- persists the fields as provided by memory
- does not recompute maturity

Current added columns:

- `mechanic_type`
- `maturity_stage`
- `evidence_basis`
- `observed_support_count`
- `hypothesis_support_count`
- `contradiction_count`
- `cross_round_stability`
- `last_evidence_tier`

### Current implementation details

Compatibility is preserved by:

- keeping old columns
- adding new columns with `ALTER TABLE ... ADD COLUMN` where missing

Read helpers now also return the new fields for durable prior consumers.

## 10. Typed session ledger

### Gap that existed

The ledger existed only as a plan item. There was no append-only runtime event stream, and later the first pass still used generic payload dictionaries.

### How it was addressed

`src/v3_1/runtime/session_ledger.py` now defines:

- `SessionLedgerRecord`
- `SessionLedger`

and typed payload records such as:

- `RoundStartPayload`
- `PlanSelectedPayload`
- `EpisodeExecutedPayload`
- `AnalysisCompletedPayload`
- `MergeCompletedPayload`
- `MemoryReconcilePayload`
- `DurableFlushPayload`
- `StopDecisionPayload`

### Current implementation details

The ledger is append-only.

Each record includes:

- `session_id`
- `round_id`
- `pass_id`
- `event_type`
- `blackboard_version`
- `memory_version`
- `plan_context_id`
- `episode_id`
- `decision_id`
- `outcome_id`
- `timestamp`
- `payload`

The payload is stored as a dictionary in the final record, but it is created from typed dataclass payloads where the runtime now uses the new helpers.

## 11. Round-runner ledger integration

### Gap that existed

The active round loop did not produce a new authoritative event stream.

### How it was addressed

`src/v3_1/runtime/round_runner.py` now appends ledger records after the authoritative transitions in the current multi-trial round loop.

Current stage events appended:

- round start
- probe plan selected
- probe episode executed
- probe analysis completed
- probe blackboard merge completed
- probe memory reconcile completed
- directed plan selected
- directed episode executed
- directed analysis completed
- directed blackboard merge completed
- directed memory reconcile completed

### Current implementation details

The current runner still has multi-trial logic:

- probe branches
- directed branches
- branch winner selection

Ledger appends happen against that live structure, not a simplified single-episode loop.

## 12. Orchestrator ledger ownership

### Gap that existed

The orchestrator did not own a run-scoped ledger or append flush/stop events.

### How it was addressed

`src/v3_1/runtime/orchestrator.py` now:

- creates `SessionLedger` at session start
- passes it into `RoundRunner`
- appends:
  - `durable flush requested`
  - `durable flush completed`
  - `stop decision made`
- keeps the ledger alive through post-run export

### Current implementation details

Orchestrator still remains control-flow-only in the current architecture:

- it owns the run loop
- it owns stop/flush timing
- it does not replace authoritative blackboard/memory/planner/storage ownership

## 13. Ledger-first post-run chronology

### Gap that existed

Post-run export still depended primarily on orchestrator-accumulated `round_records` and `episodes`.

### How it was addressed

`src/v3_1/runtime/postrun_exports.py` now derives ledger-first chronology views:

- per-round stage ordering
- decision/outcome linkage
- blackboard/memory version transitions
- durable flush chronology
- stop reason chronology

It also now persists the ledger as a standalone artifact:

- `session_ledger.json`

### Current implementation details

The ledger is now the first source for chronology/linkage/version views.

Compatibility fallback remains in place:

- heatmaps still use `episodes`
- memory summary still uses `round_records`

This is intentional because the current ledger does not yet duplicate every high-volume export payload.

## Current implementation by file

## `src/v3_1/world/blackboard.py`

Current implementation:

- owns split world-state stores plus legacy combined stores
- exposes snapshot over current combined `state`
- exposes explicit observed/hypothesized/combined read views

Authoritative writer:

- blackboard agent through merge

## `src/v3_1/world/merge.py`

Current implementation:

- classifies all incoming delta rows
- maintains split stores
- rebuilds legacy combined stores
- rebuilds planner-facing indexes with evidence-tier row views

Authoritative writer:

- blackboard merge path

## `src/v3_1/planning/planner_service.py`

Current implementation:

- builds a planner-facing observed-first blackboard
- injects planning-preparation trace context into belief
- annotates candidates with seed evidence tier/source information
- applies hypothesis-only penalty

Authoritative writer:

- planner decision packaging remains centralized here

## `src/v3_1/analysis/episode_analysis.py`

Current implementation:

- shared low-level analysis utilities
- explicit high-level analysis modes
- mode-specific POI/consequence/topology selection
- stamps `analysis_mode` and provenance into the returned outputs

Authoritative writer:

- analysis worker / task path returning `AnalyzedEpisode`

## `src/v3_1/agents/analysis_worker.py`

Current implementation:

- strict `analysis_mode` validation
- forwards only valid modes into `analyze_episode(...)`

## `src/v3_1/execution/env_worker.py`

Current implementation:

- owns one live env instance
- runs probe and directed episodes
- emits richer raw per-step execution telemetry

Authoritative writer:

- env worker owns raw episode construction

## `src/v3_1/execution/outcomes.py`

Current implementation:

- derives typed `outcome_evidence`
- derives final labels from the evidence block
- keeps unknown cases lower certainty where telemetry is absent

Authoritative writer:

- execution summarizer

## `src/v3_1/memory/reconcile.py`

Current implementation:

- builds durable update batches from working memory, priors, blackboard, decision, and outcome
- annotates rows with maturity/evidence fields

Authoritative writer:

- memory reconcile path

## `src/v3_1/agents/memory_agent.py`

Current implementation:

- keeps working memory mutable in process
- loads durable priors at session start
- filters durable rows before flush request construction

Authoritative writer:

- memory agent for working memory

## `src/v3_1/storage/persistent_memory.py`

Current implementation:

- owns SQLite durable schema
- persists durable rows transactionally
- validates presence of maturity/evidence fields
- keeps older columns intact

Authoritative writer:

- storage agent via persistent memory store

## `src/v3_1/runtime/session_ledger.py`

Current implementation:

- append-only runtime event ledger
- typed payload helper records
- no export logic inside the ledger module

## `src/v3_1/runtime/round_runner.py`

Current implementation:

- active multi-trial probe/direct round execution
- authoritative stage calls
- ledger append after stage transitions

## `src/v3_1/runtime/orchestrator.py`

Current implementation:

- run-scoped ledger ownership
- session-start durable prior load
- periodic/final durable flush decisions
- stop-policy application
- post-run export call

## `src/v3_1/runtime/postrun_exports.py`

Current implementation:

- session summary assembly
- memory summary assembly
- heatmap payload assembly
- ledger-first chronology views
- session ledger artifact persistence

## Remaining compatibility boundaries

These are current-state boundaries, not open redesign items:

- legacy combined blackboard stores remain present for compatibility
- planner still receives a combined belief structure, but now with observed-first preparation and explicit trace context
- post-run still uses `episodes` and `round_records` for heatmaps and memory summaries where the ledger intentionally does not replicate those heavy payloads
- some execution telemetry remains derived because the env does not expose it directly

## Validation status for this implementation pass

The touched modules compile successfully with:

- `python -m py_compile`

No additional behavior claims are made here beyond the code changes already present in the listed files.
