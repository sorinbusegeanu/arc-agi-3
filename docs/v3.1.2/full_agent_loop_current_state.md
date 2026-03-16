# Stage 1: Observe

- Owner component(s): [RoundRunner](/home/zodrak/zod/src/v3_1/runtime/round_runner.py), [EnvWorkerAgent](/home/zodrak/zod/src/v3_1/agents/env_worker_agent.py), [EnvWorker](/home/zodrak/zod/src/v3_1/execution/env_worker.py), [executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py)
- Main inputs: `PlanningContext`, `PlannerDecision` for directed mode, `ExecutorRequest`, env adapter from [env_factory.py](/home/zodrak/zod/src/v3_1/execution/env_factory.py)
- Main outputs: `ExecutorOutcome`, `RawEpisode`, per-step `RawStep` rows
- State mutations: env worker mutates only its own live env instance, `reset_counter`, `last_observation`, `last_info`
- Authoritative writer: env worker is the only writer of raw episode execution data
- Snapshot/version dependencies: uses the selected `plan_context_id`; does not mutate blackboard or memory versions directly
- Key files / functions / classes implementing it:
  - [Orchestrator.run](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)
  - [RoundRunner.run_round](/home/zodrak/zod/src/v3_1/runtime/round_runner.py)
  - [build_executor_request](/home/zodrak/zod/src/v3_1/execution/executor_service.py)
  - [EnvWorker.run](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
  - [EnvWorker._run_probe](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
  - [EnvWorker._run_directed](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
- What is complete: probe and directed execution are separate; the executor request is typed; the env worker is the only execution owner; probe uses only available actions; directed mode is strict and does not intentionally degrade into probing
- What is heuristic / partial / indirect:
  - directed routing uses `route_instruction(...)` plus bounded local reroute logic, but it still depends on observation-derived avatar/target localization
  - probe policy is weighted random with anti-repeat heuristics, not model-based exploration
  - env success is still partly inferred from episode reward/done shaping inside outcome summarization
- Known limitations / bug-prone areas:
  - the worker resets at the start of every probe and directed episode, so env continuity is process-persistent but episode-reset-based
  - missing avatar/target can still end the directed episode after a limited micro-recovery attempt
  - terminal rendering happens inside Ray workers, not the coordinator terminal

# Stage 2: Analyze

- Owner component(s): [AnalysisWorker](/home/zodrak/zod/src/v3_1/agents/analysis_worker.py), [analyze_episode](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py)
- Main inputs: `RawEpisode`
- Main outputs: `AnalyzedEpisode`, `BlackboardDelta`, `summary`, POIs, avatar tracks, motion rows, area sequence
- State mutations: none outside the analysis call; analysis worker is stateless/light-state
- Authoritative writer: no cumulative state writer here; analysis emits proposals/deltas only
- Snapshot/version dependencies: consumes episode metadata and step observations; no blackboard or memory snapshot mutation
- Key files / functions / classes implementing it:
  - [AnalysisWorker.analyze](/home/zodrak/zod/src/v3_1/agents/analysis_worker.py)
  - [analyze_episode](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py)
  - [summarize_observation](/home/zodrak/zod/src/v3_1/analysis/observation_summary.py)
  - [track_avatar](/home/zodrak/zod/src/v3_1/analysis/avatar_tracking.py)
  - [detect_pois](/home/zodrak/zod/src/v3_1/analysis/poi_detection.py)
  - [summarize_motion](/home/zodrak/zod/src/v3_1/analysis/motion_analysis.py)
  - [assign_area](/home/zodrak/zod/src/v3_1/analysis/area_assignment.py)
- What is complete: analysis produces structured `AnalyzedEpisode`; step rows include `avatar_cell`, action normalization, changed-cell counts, area ids, and state hashes; blackboard-ready deltas are emitted directly
- What is heuristic / partial / indirect:
  - avatar tracking still uses fallback reconstruction when the main track degenerates
  - consequences are reconstructed from changed-cell summaries and reward/done signals, not direct env telemetry
  - topology is reconstructed from per-step avatar cells and normalized actions
  - POI detection is heuristic and filter-based
- Known limitations / bug-prone areas:
  - current tests still show sessions/fixtures where POI detection returns zero POIs and downstream entity state is empty
  - active/change-region analysis is inferred from observation diffs
  - outcome-side effect strength is still an inferred metric, not direct game mechanic telemetry

# Stage 3: Merge world

- Owner component(s): [BlackboardAgent](/home/zodrak/zod/src/v3_1/agents/blackboard_agent.py), [BlackboardState](/home/zodrak/zod/src/v3_1/world/blackboard.py), [apply_delta](/home/zodrak/zod/src/v3_1/world/merge.py)
- Main inputs: `BlackboardDelta` rows from analysis
- Main outputs: `BlackboardSnapshot`, rebuilt indexes, updated cumulative world state
- State mutations: cumulative blackboard dictionaries for `areas`, `entities`, `consequences`, `trigger_zones`, `topology_nodes`, `topology_edges`, `indexes`
- Authoritative writer: blackboard agent only
- Snapshot/version dependencies: merges against the current in-memory blackboard state, then emits a new `blackboard_version`
- Key files / functions / classes implementing it:
  - [BlackboardAgent.merge](/home/zodrak/zod/src/v3_1/agents/blackboard_agent.py)
  - [BlackboardState.merge](/home/zodrak/zod/src/v3_1/world/blackboard.py)
  - [apply_delta](/home/zodrak/zod/src/v3_1/world/merge.py)
- What is complete: blackboard merge is cumulative; indexes are rebuilt; reachability, topology, consequences, trigger-zone proposals, and area metadata are folded into one authoritative state
- What is heuristic / partial / indirect:
  - consequence extraction may come from `extract_consequence_records(...)` if the transport row is not complete
  - trigger zones can be proposed from merged entity/consequence patterns rather than direct env trigger events
  - reachability is world-derived, not env-certified
- Known limitations / bug-prone areas:
  - if analysis emits weak or empty POI/entity state, blackboard entities remain empty and planner seeds collapse
  - trigger-zone inference is proposal-based, not direct mechanic instrumentation

# Stage 4: Update memory

- Owner component(s): [MemoryAgent](/home/zodrak/zod/src/v3_1/agents/memory_agent.py), [SkillMemoryState](/home/zodrak/zod/src/v3_1/memory/skill_memory.py)
- Main inputs: latest blackboard state, latest outcome, optional decision, `pass_id`
- Main outputs: `MemorySnapshot`, updated working memory, pending durable update batches
- State mutations:
  - `working_memory`
  - `durable_priors` only at session start load
  - `pending_durable_updates`
  - `revision`
- Authoritative writer: memory agent only
- Snapshot/version dependencies: reconcile runs against the current memory state plus current blackboard state and emits a new `memory_version`
- Key files / functions / classes implementing it:
  - [MemoryAgent.reconcile](/home/zodrak/zod/src/v3_1/agents/memory_agent.py)
  - [SkillMemoryState.reconcile](/home/zodrak/zod/src/v3_1/memory/skill_memory.py)
  - [update_plan_memory](/home/zodrak/zod/src/v3_1/memory/plan_memory.py)
  - [update_retry_ledgers](/home/zodrak/zod/src/v3_1/memory/retries.py)
  - [apply_failure_cooldowns](/home/zodrak/zod/src/v3_1/memory/cooldowns.py)
  - [exhaustion_snapshot](/home/zodrak/zod/src/v3_1/memory/exhaustion.py)
- What is complete: memory reconcile is versioned; working memory and durable priors are separated; probe and directed reconciliation are now semantically different
- What is heuristic / partial / indirect:
  - probe reconcile updates `plan_memory` in a weaker exploration-only way
  - directed reconcile updates retries, cooldowns, exhaustion, and stronger outcome history
  - telemetry about memory writes is reconstructed in `_reconcile_telemetry(...)`
- Known limitations / bug-prone areas:
  - some learning signals are still derived from compacted decision/outcome rows rather than a dedicated typed session ledger
  - repeated-failure and route-pattern logic is heuristic-count-based

# Stage 5: Plan

- Owner component(s): [PlannerAgent](/home/zodrak/zod/src/v3_1/agents/planner_agent.py), [plan](/home/zodrak/zod/src/v3_1/planning/planner_service.py), optional [PlanningHelperWorker](/home/zodrak/zod/src/v3_1/agents/planning_helper_worker.py), [HelperCoordinator](/home/zodrak/zod/src/v3_1/runtime/helper_coordinator.py)
- Main inputs: `PlanningContext`, blackboard snapshot state, memory snapshot state, optional helper results
- Main outputs: `PlannerDecision`, planner trace, ranked candidates, selected candidate/action
- State mutations: planner itself does not own cumulative mutable state; helper summary is written into `TaskRegistry`
- Authoritative writer: planner is the only final decision authority; helper workers are non-authoritative
- Snapshot/version dependencies:
  - planner consumes one `PlanningContext`
  - helper requests are stamped with `plan_context_id`, blackboard version, memory version, policy version, ranker version
  - invalidation is version-based before final directed planning
- Key files / functions / classes implementing it:
  - [plan](/home/zodrak/zod/src/v3_1/planning/planner_service.py)
  - [build_belief](/home/zodrak/zod/src/v3_1/planning/belief_builder.py)
  - [generate_candidates](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py)
  - [filter_candidates](/home/zodrak/zod/src/v3_1/planning/candidate_filters.py)
  - [score_candidates](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)
  - [rerank_candidates](/home/zodrak/zod/src/v3_1/planning/reranking.py)
  - [fallback_candidates](/home/zodrak/zod/src/v3_1/planning/fallbacks.py)
  - [run_helper_mode](/home/zodrak/zod/src/v3_1/planning/helper_modes.py)
- What is complete: the planner is a real multi-stage pipeline; belief is versioned; helper workers are advisory only; reranking is deterministic; fallback candidates remain within the same exported schema
- What is heuristic / partial / indirect:
  - belief contains many derived views and aliases for compatibility
  - scoring is heuristic-weight driven
  - helper outputs are still score/risk/evidence proposals, not hard model outputs
  - route features are planner-derived, not direct env path telemetry
- Known limitations / bug-prone areas:
  - current docs in [planner_implementation.md](/home/zodrak/zod/docs/v3.1.1/planner_implementation.md) still mention older labels such as `target`; current code uses `target_interaction` / `click_target` as derived candidate classes, so code is authoritative
  - if blackboard entities are empty, candidate generation collapses toward `route_probe` and fallback

# Stage 6: Execute

- Owner component(s): [executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py), [EnvWorkerAgent](/home/zodrak/zod/src/v3_1/agents/env_worker_agent.py), [EnvWorker](/home/zodrak/zod/src/v3_1/execution/env_worker.py), [option_execution.py](/home/zodrak/zod/src/v3_1/execution/option_execution.py), [route_execution.py](/home/zodrak/zod/src/v3_1/execution/route_execution.py)
- Main inputs: `PlannerDecision`
- Main outputs: `ExecutorRequest`, then `ExecutorOutcome`
- State mutations: env worker internal env state only; no blackboard/memory mutation
- Authoritative writer: env worker only for execution trace and `ExecutorOutcome`
- Snapshot/version dependencies: executor request carries the selected `plan_context_id`; directed execution depends on the planner-selected contract fields in the request
- Key files / functions / classes implementing it:
  - [build_executor_request](/home/zodrak/zod/src/v3_1/execution/executor_service.py)
  - [choose_probe_action](/home/zodrak/zod/src/v3_1/execution/option_execution.py)
  - [choose_directed_action](/home/zodrak/zod/src/v3_1/execution/option_execution.py)
  - [route_instruction](/home/zodrak/zod/src/v3_1/execution/route_execution.py)
  - [summarize_outcome](/home/zodrak/zod/src/v3_1/execution/outcomes.py)
- What is complete: execution request now has explicit `objective`, `navigation`, `terminal_action`, `constraints`, and `stop_conditions`; probe and directed modes are separate; directed execution is the live path
- What is heuristic / partial / indirect:
  - route selection is still observation-derived and bounded-search-based
  - objective success is summarized after the episode, not directly confirmed by env semantics
  - partial success and route progress are inferred in `summarize_outcome(...)`
- Known limitations / bug-prone areas:
  - reset semantics are explicit as `reset_each_episode`
  - unavailable action mapping still fails fast
  - some success/failure classes still depend on heuristic summary terms instead of game-native telemetry

# Stage 7: Analyze outcome

- Owner component(s): [AnalysisWorker](/home/zodrak/zod/src/v3_1/agents/analysis_worker.py), [analyze_episode](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py)
- Main inputs: directed `RawEpisode`
- Main outputs: directed `AnalyzedEpisode`, second-pass `BlackboardDelta`
- State mutations: none in analysis itself
- Authoritative writer: none; still proposal-only
- Snapshot/version dependencies: uses the executed episode; no direct snapshot mutation
- Key files / functions / classes implementing it: same analysis files as Stage 2
- What is complete: the same analysis pipeline is reused for directed outcome analysis
- What is heuristic / partial / indirect:
  - outcome analysis still reconstructs changes from observation deltas
  - effect-family detection is inferred from step rows in `actual_effect_mode(...)`
- Known limitations / bug-prone areas:
  - this stage and Stage 2 share the same analysis machinery, so probe and directed analysis quality are coupled

# Stage 8: Merge again

- Owner component(s): [BlackboardAgent](/home/zodrak/zod/src/v3_1/agents/blackboard_agent.py), [BlackboardState](/home/zodrak/zod/src/v3_1/world/blackboard.py)
- Main inputs: directed-pass `BlackboardDelta`
- Main outputs: updated post-execution `BlackboardSnapshot`
- State mutations: authoritative cumulative blackboard state
- Authoritative writer: blackboard agent only
- Snapshot/version dependencies: second merge in the same round advances `blackboard_version` again
- Key files / functions / classes implementing it: same blackboard/merge files as Stage 3
- What is complete: the loop is explicitly double-pass per round: one merge after probe analysis and one merge after directed outcome analysis
- What is heuristic / partial / indirect: same merge heuristics as Stage 3
- Known limitations / bug-prone areas: if directed analysis is weak, the second merge may add little beyond topology/consequence noise

# Stage 9: Learn

- Owner component(s): [MemoryAgent](/home/zodrak/zod/src/v3_1/agents/memory_agent.py), [SkillMemoryState](/home/zodrak/zod/src/v3_1/memory/skill_memory.py), [build_durable_update_batch](/home/zodrak/zod/src/v3_1/memory/reconcile.py), [StorageAgent](/home/zodrak/zod/src/v3_1/agents/storage_agent.py), [PersistentMemoryStore](/home/zodrak/zod/src/v3_1/storage/persistent_memory.py)
- Main inputs:
  - tactical/session learning: latest blackboard state, decision, outcome, pass id
  - durable learning: drained `DurableMemoryUpdateBatch`
- Main outputs:
  - tactical/session: updated `working_memory`, new `MemorySnapshot`
  - durable/persistent: `PersistentMemoryFlushRequest`, `PersistentMemoryFlushResult`, SQLite row updates
- State mutations:
  - tactical/session: `skill_library`, `plan_memory`, `cooldowns`, `retries`, `exhausted`, `exhaustion_map`, `memory_telemetry`
  - durable/persistent: SQLite tables in [persistent_memory.py](/home/zodrak/zod/src/v3_1/storage/persistent_memory.py)
- Authoritative writer:
  - tactical/session memory: memory agent only
  - durable/persistent priors: storage agent only
- Snapshot/version dependencies:
  - tactical reconcile emits new `memory_version`
  - durable flush uses pending batches built from those memory versions
  - flush/load are gated by storage config and flush policy
- Key files / functions / classes implementing it:
  - [SkillMemoryState.reconcile](/home/zodrak/zod/src/v3_1/memory/skill_memory.py)
  - [MemoryAgent.build_flush_request](/home/zodrak/zod/src/v3_1/agents/memory_agent.py)
  - [StorageAgent.flush_persistent_memory](/home/zodrak/zod/src/v3_1/agents/storage_agent.py)
  - [PersistentMemoryStore.flush](/home/zodrak/zod/src/v3_1/storage/persistent_memory.py)
- What is complete:
  - tactical/session learning happens every probe and directed reconcile
  - durable batches are accumulated in memory and flushed later
  - durable rows now carry `source_mode`, `support_count`, `confidence`, `stable_rounds`, `last_updated_round`, and `allowed_for_durable_write`
- What is heuristic / partial / indirect:
  - durable eligibility is metadata-based and inferred from row counts/confidence, not directly validated by game semantics
  - probe-generated durable rows are usually marked non-authoritative for durable write
  - memory summary/reporting later reconstructs behavior from telemetry and compact round records
- Known limitations / bug-prone areas:
  - durable batch contents are still broad and family-shaped rather than strongly typed by mechanic maturity
  - flush policy is improved but still policy-based rather than truly evidence-certified

# Stage 10: Repeat / session control

- Owner component(s): [Orchestrator](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py), [RoundRunner](/home/zodrak/zod/src/v3_1/runtime/round_runner.py), [FlushPolicy](/home/zodrak/zod/src/v3_1/runtime/flush_policy.py), [StopPolicy](/home/zodrak/zod/src/v3_1/runtime/stop_policy.py), [postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py)
- Main inputs: latest blackboard snapshot, latest memory snapshot, accumulated round records, selected targets, first observation
- Main outputs: session result dict, post-run reports, heatmaps, optional final durable flush metadata
- State mutations:
  - `SnapshotRegistry`
  - `TaskRegistry`
  - session artifact files
  - visualization files
  - final durable SQLite flush
- Authoritative writer:
  - coordinator is authoritative for control flow only
  - storage agent is authoritative for durable artifacts and SQLite writes
- Snapshot/version dependencies:
  - every planning pass uses a `PlanningContext`
  - helper invalidation is version-based
  - periodic and end-of-session durable flush decisions depend on pending durable status
- Key files / functions / classes implementing it:
  - [Orchestrator.run](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)
  - [RoundRunner.run_round](/home/zodrak/zod/src/v3_1/runtime/round_runner.py)
  - [export_postrun](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py)
- What is complete: the coordinator now delegates sequencing, helper dispatch, stop decisions, flush timing, and export shaping to narrower runtime components
- What is heuristic / partial / indirect:
  - stop policy uses no-progress, planner-starvation, repeated-route-failure, and no-new-evidence counters
  - post-run export still depends on orchestrator-accumulated `analyzed_rows` and `round_records`, not a separate session ledger
- Known limitations / bug-prone areas:
  - the loop is not a single simple pass; it is probe-pass reconcile plus directed-pass reconcile inside each round
  - post-run export remains outside the active loop and uses already-shaped session rows

# Actual end-to-end control flow in current implementation

1. [bootstrap_services](/home/zodrak/zod/src/v3_1/runtime/bootstrap.py) creates Ray actors/workers.
2. [Orchestrator.run](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py) loads persistent priors through storage -> memory.
3. The coordinator initializes blackboard and memory snapshots.
4. For each round, [RoundRunner.run_round](/home/zodrak/zod/src/v3_1/runtime/round_runner.py) executes:
   - probe plan
   - probe observe
   - probe analyze
   - probe merge
   - probe memory reconcile
   - directed plan context creation
   - invalidation check
   - optional helper dispatch
   - final directed plan
   - directed execute
   - directed analyze
   - directed merge
   - directed memory reconcile
   - round artifact persistence
   - round debug heatmaps
5. After each round, the coordinator optionally flushes durable memory according to [FlushPolicy](/home/zodrak/zod/src/v3_1/runtime/flush_policy.py).
6. [StopPolicy](/home/zodrak/zod/src/v3_1/runtime/stop_policy.py) decides whether to continue.
7. At session end, the coordinator performs the final durable flush first.
8. Then [export_postrun](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py) builds session summary, memory summary, heatmap payloads, and visualization outputs.

# Main deviations from the ideal full-agent loop

- The current loop is double-pass per round, not one simple linear pass:
  - probe observe/analyze/merge/update-memory
  - then directed plan/execute/analyze/merge/update-memory
- Planning happens twice in some rounds:
  - probe decision
  - final directed decision
  - plus optional seed decision for helpers
- “Learn” is split:
  - tactical/session updates happen inside every memory reconcile
  - durable/persistent updates are batched and flushed later
- Post-run export is not part of the active per-step loop; it is a final sink after the round loop ends

# Authoritative state owners and write boundaries

- Environment execution state: [EnvWorker](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
- Raw episode trace: env worker
- Analyzed episode and deltas: [AnalysisWorker](/home/zodrak/zod/src/v3_1/agents/analysis_worker.py) output only, non-authoritative
- Cumulative world state: [BlackboardAgent](/home/zodrak/zod/src/v3_1/agents/blackboard_agent.py)
- Tactical/session memory: [MemoryAgent](/home/zodrak/zod/src/v3_1/agents/memory_agent.py)
- Final plan decision: [PlannerAgent](/home/zodrak/zod/src/v3_1/agents/planner_agent.py)
- Helper proposals: [PlanningHelperWorker](/home/zodrak/zod/src/v3_1/agents/planning_helper_worker.py), non-authoritative
- Durable long-term memory: [StorageAgent](/home/zodrak/zod/src/v3_1/agents/storage_agent.py) writing [PersistentMemoryStore](/home/zodrak/zod/src/v3_1/storage/persistent_memory.py)
- Session/post-run artifacts: storage agent
- Session control: [Orchestrator](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)

# Current weak points / architectural risks

- Analysis and world merge still have known empty-entity / zero-POI cases, and those collapse planner seeds.
- Outcome and mechanic signals are still reconstructed from observation diffs, changed-cell counts, and compact summaries.
- Durable learning eligibility is still metadata-driven, not directly game-certified.
- Post-run export still depends on coordinator-accumulated export-shaped history.
- Probe and directed episodes both reset the env, so the control loop is round-based over fresh episodes rather than one continuous live game trajectory.

# Open ambiguities where code and docs do not fully match

- [coordinator_persistent_memory_and_postrun_implementation.md](/home/zodrak/zod/docs/v3.1.1/coordinator_persistent_memory_and_postrun_implementation.md) still describes helper dispatch and export shaping as coordinator-owned. Current code has moved those responsibilities into [RoundRunner](/home/zodrak/zod/src/v3_1/runtime/round_runner.py), [HelperCoordinator](/home/zodrak/zod/src/v3_1/runtime/helper_coordinator.py), [FlushPolicy](/home/zodrak/zod/src/v3_1/runtime/flush_policy.py), [StopPolicy](/home/zodrak/zod/src/v3_1/runtime/stop_policy.py), and [export_assembler.py](/home/zodrak/zod/src/v3_1/runtime/export_assembler.py). Code is authoritative.
- [planner_implementation.md](/home/zodrak/zod/docs/v3.1.1/planner_implementation.md) still mentions older candidate class labels like `target`. Current code in [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py) derives classes such as `target_interaction`, `click_target`, `frontier_move`, `route_probe`, `trigger_probe`, `recovery_move`, and `fallback_action`. Code is authoritative.
- [helper_workers_and_executor_implementation.md](/home/zodrak/zod/docs/v3.1.1/helper_workers_and_executor_implementation.md) still describes flatter helper nudges. Current code in [helper_modes.py](/home/zodrak/zod/src/v3_1/planning/helper_modes.py) returns structured proposals with confidence, warning codes, contradiction flags, support-strength adjustments, and evidence. Code is authoritative.

# ideal_stage | current_component_owner | implemented? | authoritative_writer | main artifact/state | main limitation

| ideal_stage | current_component_owner | implemented? | authoritative_writer | main artifact/state | main limitation |
| --- | --- | --- | --- | --- | --- |
| Observe | RoundRunner + EnvWorkerAgent + EnvWorker + executor service | yes | EnvWorker | `RawEpisode`, `ExecutorOutcome` | env resets every episode; route success still partly heuristic |
| Analyze | AnalysisWorker | yes | none; proposal-only | `AnalyzedEpisode`, `BlackboardDelta` | avatar/POI/effect inference is still heuristic |
| Merge world | BlackboardAgent | yes | BlackboardAgent | cumulative blackboard state + `BlackboardSnapshot` | weak analysis can leave entities empty |
| Update memory | MemoryAgent | yes | MemoryAgent | `working_memory`, `MemorySnapshot`, pending durable batches | probe vs directed semantics are cleaner now but still heuristic-count based |
| Plan | PlannerAgent + helper workers + HelperCoordinator | yes | PlannerAgent | `PlannerDecision`, planner trace | candidate quality collapses when world seeds are empty |
| Execute | EnvWorker | yes | EnvWorker | `ExecutorRequest` -> `ExecutorOutcome` | objective success still summarized indirectly |
| Analyze outcome | AnalysisWorker | yes | none; proposal-only | directed `AnalyzedEpisode` | same heuristic analysis stack as probe |
| Merge again | BlackboardAgent | yes | BlackboardAgent | second-pass blackboard snapshot | second merge quality depends on directed analysis quality |
| Learn | MemoryAgent + StorageAgent + PersistentMemoryStore | yes | MemoryAgent for tactical, StorageAgent for durable | tactical memory + durable SQLite updates | durable eligibility is metadata-gated, not direct telemetry |
| Repeat / session control | Orchestrator + RoundRunner + FlushPolicy + StopPolicy + postrun export | yes | Orchestrator for control, StorageAgent for artifacts | round loop, final exports, flush metadata | still double-pass and report-shaped rather than a single simple loop |
