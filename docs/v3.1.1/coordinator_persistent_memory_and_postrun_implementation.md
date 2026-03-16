# Coordinator, Persistent Long-Term Memory Updates, and Post-Run Exports

This document describes the current `v3_1` implementation of:

- the coordinator/orchestrator run loop
- durable long-term memory load and flush behavior
- post-run artifact and visualization export behavior

It reflects the actual code in:

- `src/v3_1/runtime/bootstrap.py`
- `src/v3_1/runtime/orchestrator.py`
- `src/v3_1/agents/memory_agent.py`
- `src/v3_1/agents/storage_agent.py`
- `src/v3_1/memory/skill_memory.py`
- `src/v3_1/memory/reconcile.py`
- `src/v3_1/storage/persistent_memory.py`
- `src/v3_1/storage/artifact_store.py`
- `src/v3_1/runtime/postrun_exports.py`

## 1. Coordinator implementation

The coordinator is implemented by `Orchestrator` in `src/v3_1/runtime/orchestrator.py`.

Its responsibilities are:

- load persistent priors once at session start
- initialize the authoritative blackboard and working memory snapshots
- run the per-round probe -> analyze -> merge -> reconcile -> plan -> execute -> analyze -> merge -> reconcile loop
- coordinate helper workers when enabled
- persist round artifacts through the storage agent
- trigger periodic and end-of-session durable memory flushes
- trigger post-run reports and PNG/JSON exports

### 1.1 Service bootstrap and ownership

`bootstrap_services(...)` in `src/v3_1/runtime/bootstrap.py` creates the Ray-native runtime graph.

It initializes:

- `BlackboardAgent`
- `MemoryAgent`
- `PlannerAgent`
- optional `RankerAgent`
- `StorageAgent`
- pooled `EnvWorkerAgent`s
- pooled `AnalysisWorker`s
- pooled `PlanningHelperWorker`s

The bootstrap layer also computes the durable SQLite path:

- from `config.storage.persistent_memory_db_path_override`, if present
- otherwise from `get_persistent_memory_db_path(config.storage.root_dir)`

That durable path is passed into `StorageAgent`, not `MemoryAgent`.

This keeps ownership split correctly:

- memory agent owns live mutable session memory
- storage agent owns durable SQLite writes

### 1.2 Session start

At the start of `Orchestrator.run(...)`, the coordinator does:

1. `_load_persistent_priors()`
2. blackboard initial snapshot
3. memory initial reconcile
4. snapshot registration for both initial snapshots

Persistent priors are loaded only once at session start.

The load path is:

1. coordinator builds `PersistentMemoryLoadRequest`
2. storage agent loads priors from `PersistentMemoryStore`
3. memory agent receives `PersistentMemoryLoadResult`
4. memory state merges priors into `durable_priors`
5. skill library is rebuilt using those priors

The relevant call chain is:

- `Orchestrator._load_persistent_priors()`
- `StorageAgent.load_persistent_memory(...)`
- `PersistentMemoryStore.load_priors(...)`
- `MemoryAgent.load_persistent_priors(...)`
- `SkillMemoryState.load_persistent_priors(...)`

### 1.3 Per-round stage order

The implemented round order is:

1. build probe planning context
2. ask planner for a probe decision
3. build probe executor request
4. run probe env episode
5. analyze probe episode
6. merge probe deltas into blackboard
7. reconcile memory from probe outcome
8. build directed planning context
9. invalidate stale helper/task state if versions changed
10. optionally run helper workers
11. ask planner for final directed decision
12. build directed executor request
13. run directed env episode
14. analyze directed episode
15. merge directed deltas into blackboard
16. reconcile memory from directed outcome
17. persist round snapshots and reports
18. render round debug heatmaps
19. flush durable memory periodically if configured
20. evaluate stop conditions

This is all done in-process at the coordinator level, but execution, planning, analysis, storage, and helper work are sent to Ray actors/workers.

### 1.4 Planning contexts and versioning

The coordinator constructs a `PlanningContext` from:

- blackboard snapshot handle/version
- memory snapshot handle/version
- policy version
- ranker version

This happens in `_planning_context(...)`.

Every planning pass therefore has an explicit compatibility stamp:

- `plan_context_id`
- `blackboard_version`
- `memory_version`
- `policy_version`
- `ranker_version`

When the directed planning pass begins, the coordinator compares the previous stamp against the new one and calls:

- `invalidate_if_needed(...)`

This prevents helper outputs or tracked tasks from surviving across incompatible state versions.

### 1.5 Helpers inside the coordinator

When helper workers are enabled, the coordinator does a two-phase planning pass:

1. seed planner decision without helpers
2. helper dispatch using seed planner trace and memory/blackboard slices
3. final planner decision with helper results

Helper dispatch uses:

- generated candidates from the seed trace
- selected belief slices
- local context
- trigger support
- consequence support
- topology facts
- route facts
- recent local failure patterns
- durable priors

The coordinator records helper telemetry:

- remote success count
- local fallback count
- per-helper latency
- per-helper contribution rate

These are stored through `TaskRegistry` as helper summary state and passed into planner trace/decision metadata.

### 1.6 Snapshot handling

The coordinator uses `SnapshotRegistry` for immutable snapshot access by handle.

After every successful blackboard merge or memory reconcile, the resulting snapshot is registered:

- blackboard snapshots after probe and directed merge
- memory snapshots after probe and directed reconcile

Planner contexts then refer to these handles rather than passing large mutable state objects as ownership objects.

### 1.7 Round persistence

At the end of each round, the coordinator persists:

- `blackboard_pass1_roundNNN.json`
- `memory_pass1_roundNNN.json`
- `decision_roundNNN.json`
- `analysis_summary.json`
- round debug PNGs:
  - `visit_heatmap_debug.png`
  - `poi_heatmap_debug.png`

All of these writes go through the storage agent.

### 1.8 Stop logic

The current coordinator stop logic is:

- stop on true env win if `config.runtime.stop_on_win`
- stop on `no_progress_budget`
- otherwise stop at `max_rounds`

Importantly, win is currently tied to:

- `exec_outcome.episode.won`

not route success.

## 2. Persistent long-term memory updates

Persistent long-term memory is implemented as a SQLite-backed store in:

- `src/v3_1/storage/persistent_memory.py`

This durable store is distinct from live working memory.

### 2.1 Ownership split

The ownership model is:

- `SkillMemoryState`
  - owns mutable working/session memory
  - owns loaded durable priors as advisory state
  - accumulates pending durable update batches

- `MemoryAgent`
  - owns one live `SkillMemoryState`
  - performs reconcile
  - loads priors at session start
  - builds flush requests by draining pending durable updates

- `StorageAgent`
  - owns `PersistentMemoryStore`
  - applies persistence flags
  - writes durable updates transactionally into SQLite
  - records manifest/checkpoint metadata

The memory agent does not write SQLite directly.

### 2.2 Working memory vs durable priors

`SkillMemoryState` maintains:

- `working_memory`
- `durable_priors`
- `pending_durable_updates`

`working_memory` holds tactical in-session state such as:

- skill library
- plan memory
- cooldowns
- retries
- exhaustion
- memory telemetry

`durable_priors` holds cross-session learned priors such as:

- skill stats
- candidate outcomes
- failure patterns
- recovery patterns
- POI patterns
- trigger patterns
- consequence patterns
- entity signatures
- area signatures
- mechanic hypotheses
- ranker state

These priors are advisory and persist across sessions.

### 2.3 How durable updates are created

Durable updates are created inside memory reconciliation.

The path is:

1. `MemoryAgent.reconcile(...)`
2. `SkillMemoryState.reconcile(...)`
3. `build_durable_update_batch(...)` in `src/v3_1/memory/reconcile.py`
4. resulting `DurableMemoryUpdateBatch` appended to `pending_durable_updates`

The durable update batch is built from:

- working memory
- durable priors
- current blackboard state
- latest decision
- latest outcome

It includes at least:

- `skills`
- `skill_stats`
- `candidate_outcomes`
- `failure_patterns`
- `recovery_patterns`
- `poi_patterns`
- `trigger_patterns`
- `consequence_patterns`
- `entity_signatures`
- `area_signatures`
- `mechanic_hypotheses`
- `ranker_state`

This means durable memory is derived from reconciled session state, not written incrementally from arbitrary runtime sites.

### 2.4 What gets written into SQLite

`PersistentMemoryStore` owns the schema and upsert logic.

The schema currently creates and maintains:

- `sessions`
- `games`
- `memory_snapshots`
- `skills`
- `skill_stats`
- `candidate_outcomes`
- `failure_patterns`
- `recovery_patterns`
- `poi_patterns`
- `trigger_patterns`
- `consequence_patterns`
- `entity_signatures`
- `area_signatures`
- `mechanic_hypotheses`
- `ranker_state`

Flush behavior is transactional per request.

The durable DB path is stable per storage root:

- one SQLite DB per v3.1 storage root

### 2.5 Flush cadence

The coordinator does not write durable updates every step.

Current flush behavior:

- optional periodic flush every `persistent_memory_flush_every_n_rounds`
- unconditional end-of-session flush

Periodic flush path:

- after round persistence
- only if enabled and cadence is positive

End-of-session flush path:

- after post-run exports
- using the latest round memory snapshot path

The flush path is:

1. coordinator asks memory agent to `build_flush_request(...)`
2. memory agent drains pending durable batches
3. coordinator submits flush request to storage agent
4. storage agent applies persistence flags and writes SQLite
5. storage agent records flush manifest metadata

### 2.6 Persistence flags

The storage agent can suppress whole durable families using config flags.

Before writing, `StorageAgent.flush_persistent_memory(...)` filters the batch according to flags such as:

- `persist_skill_stats`
- `persist_candidate_outcomes`
- `persist_failure_patterns`
- `persist_recovery_patterns`
- `persist_poi_patterns`
- `persist_trigger_patterns`
- `persist_consequence_patterns`
- `persist_entity_signatures`
- `persist_area_signatures`
- `persist_mechanic_hypotheses`
- `persist_ranker_state`

This means the memory layer always emits the same durable batch shape, while the storage layer decides what is actually persisted.

### 2.7 Session memory snapshots vs durable memory

Working/session memory snapshots are still exported as JSON:

- `memory_pass1_roundNNN.json`

Those remain session artifacts.

They are not the cross-run learning store.

The durable cross-run learning store is the SQLite DB managed by `PersistentMemoryStore`.

The storage layer links these two worlds by recording snapshot references in:

- session manifests
- SQLite `memory_snapshots`

## 3. Post-run exports

Post-run export is implemented in:

- `src/v3_1/runtime/postrun_exports.py`

The coordinator calls:

- `export_postrun(...)`

once the round loop ends.

### 3.1 Inputs

`export_postrun(...)` currently receives:

- session id
- final round id
- game id
- accumulated analyzed episodes in export form
- final blackboard state
- final `won` flag
- final blackboard version
- final memory version
- visualization width/height
- `export_png`
- first captured observation
- selected target entity ids across the session
- per-round records

That means post-run exports are based on:

- final blackboard
- full session episode history accumulated by the orchestrator
- full round records for memory/report summaries

### 3.2 Session-level reports

The current session-level reports include:

- `summary.json`
- `memory_events.jsonl`
- `memory_summary.json`

`summary.json` is written at the session root, not the final round directory.

It includes:

- rounds completed
- won
- latest blackboard version
- latest memory version
- unique target entity ids
- total number of entities
- `percentage_targets_solved`
- `percentage_targets_with_effect`
- `average_effect_strength`
- movement/interact/click effect split metrics

`memory_summary.json` is derived from `round_records` and direct memory telemetry, not from a second run over SQLite.

### 3.3 Round-level post-run artifacts

The current post-run function still writes final-round JSON heatmap artifacts:

- `visit_heatmap.json`
- `poi_heatmap.json`
- `visit_heatmap_debug.json`

These remain in the final round directory.

### 3.4 Visualization outputs

If the first observation was captured, post-run export writes:

- `<game_id>.png`

to the session visualization directory.

If `export_png=True`, it also writes:

- `visit_heatmap.png`
- `poi_heatmap.png`
- `visit_heatmap_debug.png`
- `poi_heatmap_debug.png`

These are session-level visualization artifacts, not round-scoped ones.

### 3.5 Heatmap source data

Visit heatmap source:

- all per-step avatar cells from all exported analyzed episodes across all rounds

POI heatmap source:

- final blackboard POIs only
- post-run POI export acceptance rules, not planner `active` state alone

The heatmap code itself lives in:

- `src/v3_1/visualization/heatmaps.py`

The post-run module only builds payloads and calls storage persistence helpers.

### 3.6 Storage path behavior

The storage agent exposes four relevant persistence paths:

- round JSON
- round bytes
- session JSON
- session bytes
- session visualization bytes

These are implemented in:

- `ArtifactStore.write_json(...)`
- `ArtifactStore.write_bytes(...)`
- `ArtifactStore.write_session_json(...)`
- `ArtifactStore.write_session_bytes(...)`
- `ArtifactStore.write_visualization_bytes(...)`

So the export split is:

- round artifacts under `round_NNN/`
- session reports under `session_id/`
- visualization PNGs under `session_id/visualization/`

### 3.7 Persistent memory metadata in final result

After `export_postrun(...)` returns, the coordinator performs the end-of-session durable flush.

If a flush occurred, the final run result exports:

- `persistent_memory_flush`
- `persistent_memory_db_path`

This makes the final result bundle explicitly tie together:

- session artifacts
- final post-run exports
- durable memory flush metadata

## 4. Current strengths

The current implementation has a few strong properties:

- single-writer ownership is maintained
- durable SQLite writes are isolated to storage
- working memory remains mutable and in-process only
- persistent priors load once at session start
- durable writes do not block every step
- post-run export is separate from active planning/execution
- JSON session snapshots and durable SQLite memory are clearly distinct

## 5. Current limitations

The current implementation also has clear limitations:

- post-run export is still a fairly large monolithic function
- coordinator round logic is still long and centralizes many policy details
- durable update batching is broad and heuristic, not yet strongly typed by learning family beyond the batch schema
- periodic durable flush is round-based only, not checkpoint-policy aware beyond cadence
- post-run exports currently consume orchestrator-built episode rows rather than querying a richer persisted episode store
- the coordinator still owns some export-shaping logic such as `_episode_export_row(...)` and round analysis summaries

## 6. Practical mental model

The most accurate mental model of the current implementation is:

- coordinator owns session control and stage ordering
- blackboard owns cumulative world state
- memory owns mutable tactical state plus loaded durable priors
- planner consumes current blackboard + current memory + helper proposals
- executor/env worker produces typed raw episode outcomes
- analysis converts episodes into blackboard-ready deltas
- storage owns all durable writes:
  - JSON artifacts
  - manifests
  - SQLite long-term memory
- post-run export is a sink that summarizes the completed session after the active control loop ends

