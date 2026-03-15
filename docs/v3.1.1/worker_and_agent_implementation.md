# v3.1.1 Worker and Agent Implementation

This note describes the current implementation of the Environment worker, Analysis worker, Blackboard agent, and Memory agent in `src/v3_1`.

## Environment worker

Code:
- `src/v3_1/agents/env_worker_agent.py`
- `src/v3_1/execution/env_worker.py`

### Structure

`EnvWorkerAgent` is a thin Ray actor wrapper around `EnvWorker`.

- The actor constructor builds one long-lived worker-owned environment through `EnvWorker.from_config(...)`.
- The actor exposes one method, `execute(request)`, which delegates to `EnvWorker.run(request)`.

`EnvWorker` owns:
- one `NormalizedEnvAdapter` instance
- the worker id
- the latest observation/info
- a reset counter

### Execution modes

`EnvWorker.run(request)` dispatches by `request.mode`:

- `probe` -> `_run_probe(...)`
- anything else -> `_run_directed(...)`

### Probe execution

`_run_probe(...)`:

- resets the environment using the request seed
- repeatedly reads `available_actions`
- picks an exploratory action with `choose_probe_action(...)`
- normalizes the emitted env action via `normalize_action_lookup(...)`
- steps the environment
- appends one `RawStep` per interaction
- stops on `done`, `truncated`, or step budget exhaustion

Each `RawStep` stores:
- raw observation
- raw action
- normalized `action_id`
- normalized `action_name`
- normalized `action_family`
- reward / done / truncated
- env `info`

### Directed execution

`_run_directed(...)`:

- resets the environment using the request seed
- on each step asks `route_instruction(...)` for the next routed intent
- if routing fails or stops, records the routed failure and exits
- otherwise selects the concrete env action with `_select_directed_action(...)`
- normalizes the emitted action with `normalize_action_lookup(...)`
- steps the environment
- records one `RawStep`
- stops when:
  - the routed instruction is terminal
  - the env returns `done` or `truncated`
  - movement stalls

### Action semantics

`_select_directed_action(...)` respects `request.required_action_family`.

- For `click_at`, it finds a click-capable env action and injects coordinates.
- For other families, it prefers:
  - the routed `desired_action_name`
  - otherwise an action whose normalized `action_family` matches the required family at terminal time

The worker therefore emits actual env actions rather than planner-only symbolic actions.

### Output

Both probe and directed paths end in `_build_outcome(...)`, which returns `ExecutorOutcome`.

That outcome contains:
- the full `RawEpisode`
- success / termination reason
- reward delta
- an `outcome` summary from `summarize_outcome(...)`

The worker does not mutate blackboard, memory, or storage directly.

## Analysis worker

Code:
- `src/v3_1/agents/analysis_worker.py`
- `src/v3_1/analysis/episode_analysis.py`

### Structure

`AnalysisWorker` is a minimal Ray actor with one method:

- `analyze(raw_episode)` -> `analyze_episode(raw_episode)`

The real implementation lives in `analyze_episode(...)`.

### Analysis pipeline

For each raw episode, `analyze_episode(...)` does the following:

1. Normalizes every observation with `normalize_observation(...)`
2. Builds per-step observation summaries with `summarize_observation(...)`
3. Assigns areas with `assign_area(...)`
4. Tracks the avatar with `track_avatar(...)`
5. Applies a fallback avatar-track reconstruction if the track degenerates
6. Computes motion summaries with `summarize_motion(...)`
7. Builds normalized `step_rows`
8. Builds topology nodes/edges from `step_rows`
9. Builds consequence rows from motion + step rows
10. Detects POIs with `detect_pois(...)`
11. Packages all of that into one `AnalyzedEpisode`

### Step-level normalized transport

Each `step_row` includes:
- `step_idx`
- raw `action`
- normalized `action_id`
- normalized `action_name`
- normalized `action_family`
- backward-compatible `action_type`
- `target_entity_id`
- `target_coordinates`
- `reward`
- `done`
- `area_id`
- state hash
- avatar centroid / avatar cell
- changed-cell statistics

This makes analysis the canonical bridge from executed env actions into the symbolic world model.

### Blackboard delta emission

`analyze_episode(...)` emits one `BlackboardDelta` inside `AnalyzedEpisode.blackboard_deltas`.

That delta contains:
- `areas`
- `entities` derived from detected POIs
- normalized `consequences`
- `topology_nodes`
- `topology_edges`
- `evidence`
- metadata including `step_rows`

The analysis worker does not write artifacts itself. It only returns analyzed objects to the orchestrator.

## Blackboard agent

Code:
- `src/v3_1/agents/blackboard_agent.py`
- `src/v3_1/world/blackboard.py`
- `src/v3_1/world/merge.py`

### Structure

`BlackboardAgent` is a Ray actor that owns one `BlackboardState`.

It exposes:
- `merge(round_id, pass_id, deltas)`
- `snapshot(round_id, pass_id, material_change)`
- `get_state()`

### Responsibility

The blackboard agent is the owner of cumulative world state for the current session/game.

It merges analysis deltas into persistent state rather than replacing state per episode.

### Merge behavior

At merge time the blackboard path:

1. Merges areas
2. Merges entities / POIs
3. Validates or reconstructs normalized consequences
4. Merges topology nodes and edges
5. Recomputes reachability for entities
6. Proposes and merges trigger zones
7. Rebuilds indexes

`apply_delta(...)` in `world/merge.py` is the core reducer.

### State shape

The blackboard state keeps:
- `areas`
- `entities`
- `consequences`
- `trigger_zones`
- `topology_nodes`
- `topology_edges`
- `indexes`

In current `v3_1`, `entities` means stable merged planner-facing entities, not all raw objects seen during analysis.

Concretely:
- analysis emits POI-like entities into the delta
- blackboard merges those into stable session entities
- raw per-step object detections remain in analyzed episode payloads and are not inserted wholesale into blackboard state

Reachability is not a separate top-level section. It is stored on entities as fields such as:
- `reachable_now`
- `reachable_later`
- `access_profile`

### Trigger zones

Trigger zones are lightweight merged hypotheses derived from reachable POI entities plus consequence evidence.

The current trigger-zone row stores:
- `trigger_zone_id`
- `entity_id`
- `centroid`
- `bbox`
- `confidence`
- `observations`
- `evidence_refs`

The current implementation does not store:
- explicit contradiction markers
- explicit decay markers
- explicit merge lineage
- explicit support history beyond cumulative `observations`

So the code supports source evidence refs and merged confidence/observation counts, but not the richer contradiction/decay/lineage model.

### Consequences

The intended normal path is for analysis to emit normalized consequence rows directly in `BlackboardDelta.consequences`.

Blackboard-side reconstruction exists as a fallback validation path:
- if incoming consequence rows already contain normalized action transport and `evidence_refs`, merge uses them directly
- if they are incomplete, merge rebuilds them from `extract_consequence_records(delta)`

So reconstruction is fallback-only behavior in the implementation, not the normal preferred path.

### Output

`snapshot(...)` returns a `BlackboardSnapshot` containing:
- a snapshot handle
- blackboard version
- created round / pass ids
- material-change flag
- the full merged state

The blackboard agent is pure session state management. It does not execute env steps or perform episode analysis.

## Memory agent

Code:
- `src/v3_1/agents/memory_agent.py`
- `src/v3_1/memory/skill_memory.py`

### Structure

`MemoryAgent` is a Ray actor that owns one `SkillMemoryState`.

It exposes:
- `reconcile(...)`
- `load_persistent_priors(...)`
- `build_flush_request(...)`
- `get_state()`

### Responsibility

The memory agent owns planner-facing working memory and session-to-session durable update staging.

It is updated after blackboard state changes and directed outcomes.

### Reconcile behavior

`SkillMemoryState.reconcile(...)` is the core implementation.

For each reconcile call it:

1. Reads the selected candidate from the planner decision
2. Reads the execution outcome
3. Ages cooldowns on directed/pass-1 reconcile
4. Updates retry ledgers
5. Applies failure cooldowns when the directed outcome failed
6. Updates tactical plan memory
7. Rebuilds the skill library from current entities / trigger zones
8. Updates skill execution stats
9. Recomputes exhaustion from retries
10. Emits direct memory telemetry for the write operations
11. Builds a new `MemorySnapshot`
12. Stages one durable-memory batch for later flush

### Working memory contents

The working memory currently includes:
- `skill_library`
- `plan_memory`
- `cooldowns`
- `retries`
- `exhausted`
- `exhaustion_map`
- `memory_telemetry`

### plan_memory schema

`plan_memory` is a session-scoped tactical memory map updated on each reconcile.

Current schema:
- `history`
- `repeated_failures`
- `movement_memory`
- `recovery_memory`
- `blocked_patterns`
- `route_patterns`
- `candidate_class_performance`
- `no_progress_rounds`

Current keying strategy:
- `history`
  - bounded append-only list of recent decision/outcome entries
- `repeated_failures`
  - keyed by `target_entity_id`
- `movement_memory`
  - keyed by `target_area_id`
- `recovery_memory`
  - keyed by `target_area_id` or fallback target/global key
- `blocked_patterns`
  - keyed as `<candidate_class>:<termination_reason>:<target_area_id|target_entity_id|global>`
- `route_patterns`
  - keyed as `<candidate_class>:<target_area_id|target_entity_id|global>`
- `candidate_class_performance`
  - keyed by `candidate_class`

Lifetime:
- it lives in working memory for the duration of the session
- it is serialized into memory snapshots
- it is updated incrementally across rounds

Aging / retention rules:
- `history` is truncated to the last 80 entries
- `no_progress_rounds` increments or resets based on progress
- the other maps currently do not have explicit time-decay or age-out rules; they accumulate within the session unless overwritten/reset by normal update logic

### Skill library

The skill library is rebuilt on every reconcile from current blackboard entities and trigger zones, then merged with the prior in-memory library.

Skill identity is stable only to the extent that upstream ids stay stable.

Current deterministic id rules:
- inspect skill id:
  - `skill:inspect:{stable_digest({'entity_id': entity_id})}`
- recover skill id:
  - `skill:recover:{stable_digest({'entity_id': entity_id, 'area': area_id})}`
- trigger skill id:
  - `skill:trigger:{stable_digest({'trigger_id': trigger_id, 'target': target_entity_id})}`

Execution stats survive rebuilds when the rebuilt skill resolves to the same `skill_id`.

Continuity implications:
- if the entity id stays stable, inspect/recover skills keep continuity
- if trigger-zone identity changes, the trigger skill id can change and continuity can break
- if entity identity changes upstream, the rebuilt skill is treated as a new skill row

So the invariant is deterministic continuity from stable upstream ids, not guaranteed semantic continuity under arbitrary entity/trigger re-identification

`memory_telemetry` records direct write events such as:
- retry increments
- cooldown set/clear
- exhaustion set/clear
- route-failure writes
- recovery-history writes
- skill-stat updates
- touched memory keys per directed outcome

### Durable memory staging

The memory agent does not flush to SQLite directly.

Instead:
- `reconcile(...)` appends durable update batches to `pending_durable_updates`
- `build_flush_request(...)` drains those updates into a `PersistentMemoryFlushRequest`
- the storage agent performs the actual durable flush

### Output

`reconcile(...)` returns a `MemorySnapshot` containing:
- snapshot handle
- memory version
- created round / pass ids
- the composed state

The memory agent therefore owns session memory mutation, while storage owns durable persistence.

### Snapshot pairing

The planner reads exactly one blackboard snapshot handle/version and one memory snapshot handle/version per planning call through `PlanningContext`.

That pair is coordinator-selected by the orchestrator:
- it takes a concrete blackboard snapshot
- takes a concrete memory snapshot
- builds one `PlanningContext` containing both handles and both versions

What is guaranteed in code:
- within one `decide(...)` call, the planner sees one explicit blackboard snapshot version and one explicit memory snapshot version
- those two snapshots were paired by the orchestrator when the context was built

What is not provided by separate transactional machinery:
- there is no stronger cross-actor transactional snapshot system beyond the coordinator choosing the pair in order
- if versions change later, invalidation logic handles stale contexts rather than retroactively changing the pair
