# v3.1.6 Implementation

This document describes the current `src/v3_1` implementation as it exists in code on 2026-03-26. It is an implementation note, not a design spec. Where the runtime uses heuristics, gates, or partial fallbacks, those behaviors are called out directly.

## Scope

`v3_1` is a standalone symbolic runtime. The package is organized into these implementation surfaces:

- `runtime/`: session orchestration, round control, invalidation, helper coordination, export assembly, stop/flush policy, ledgering
- `agents/`: authoritative Ray actors and helper workers
- `planning/`: belief building, candidate generation/filtering/scoring/reranking, decision packaging, subgoal-chain logic
- `execution/`: environment worker loop, route execution, action execution, live avatar tracking, outcome summarization
- `analysis/`: observation normalization, spatial/object summarization, POI extraction, topology/consequence reconstruction, mechanic graph extraction
- `world/`: blackboard state, merges, queries, reachability, topology, consequences, trigger zones, mechanic graph storage and merge
- `mechanics/`: deterministic rule engine, hypothesis orchestration, LLM prompt/reason/validation, proposal registry and types
- `memory/`: working memory, plan memory, retries, cooldowns, exhaustion, durable update batching, reconcile
- `learning/`: optional ranker state, feature cache, score bonus service, update stubs
- `storage/`: artifact persistence, manifests, SQLite manifest index, session store, persistent memory database, checkpoint helpers
- `config/`: schema, defaults, loader, validation, resource sizing, feature flags, hypothesis-generation config
- `llm/`: local adapter interface, stub adapter, OpenAI-compatible adapter
- `visualization/`: run summaries, heatmaps, PNG rendering, export helpers
- `eval/`: offline metrics and evaluation entry points
- `cli/`: session runner, report export, blackboard print, hypothesis eval

## Architecture Rules

The implementation is constrained by [ARCHITECTURE_RULES.md](/home/zodrak/zod/src/v3_1/ARCHITECTURE_RULES.md):

- `v3_1` does not import runtime behavior from `src/v3` or `src/codex_baseline_v2`.
- Authoritative live state is owned only by native v3.1 agents.
- Files are export sinks only, never the runtime message bus.
- Blackboard, working memory, planner decision, storage, and durable memory each have explicit ownership boundaries.
- Helper workers and env workers produce proposals or telemetry, but never authoritatively mutate blackboard or memory.

In practice, the five long-lived authoritative state surfaces are:

1. environment state inside env workers
2. blackboard world state
3. working memory state
4. mechanic graph state
5. hypothesis registry state

## Bootstrap And Service Graph

The main CLI entry point is [run_autonomous_game.py](/home/zodrak/zod/src/v3_1/cli/run_autonomous_game.py). It loads config, builds a `RunContext`, bootstraps services, then invokes [Orchestrator](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py).

[bootstrap.py](/home/zodrak/zod/src/v3_1/runtime/bootstrap.py) initializes Ray if needed and constructs:

- `BlackboardAgent`
- `MemoryAgent`
- `MechanicGraphAgent`
- `PlannerAgent`
- optional `RankerAgent`
- `StorageAgent`
- a pool of `EnvWorkerAgent`s
- function tasks for analysis and helper-worker execution
- in-process `HypothesisRegistry`
- a local LLM adapter, either stub or OpenAI-compatible

The shipped config in [config_def.conf](/home/zodrak/zod/src/v3_1/config_def.conf) currently enables:

- Ray execution
- helper workers
- ranker actor
- deterministic hypotheses
- persistent memory
- SQLite manifest indexing

LLM hypothesis generation is configured but disabled by default in that file.

## Agent Ownership

The authoritative actors are thin wrappers around native stateful implementations:

- [blackboard_agent.py](/home/zodrak/zod/src/v3_1/agents/blackboard_agent.py): only cumulative world-state writer
- [memory_agent.py](/home/zodrak/zod/src/v3_1/agents/memory_agent.py): only working-memory writer and durable-batch builder
- [mechanic_graph_agent.py](/home/zodrak/zod/src/v3_1/agents/mechanic_graph_agent.py): only mechanic-graph writer and graph-driven hypothesis feedback updater
- [planner_agent.py](/home/zodrak/zod/src/v3_1/agents/planner_agent.py): final decision authority via `planning.planner_service.plan(...)`
- [storage_agent.py](/home/zodrak/zod/src/v3_1/agents/storage_agent.py): only durable artifact and persistent-memory writer
- [env_worker_agent.py](/home/zodrak/zod/src/v3_1/agents/env_worker_agent.py): execution sandbox; never mutates blackboard or memory directly
- [ranker_agent.py](/home/zodrak/zod/src/v3_1/agents/ranker_agent.py): optional score bonus actor

Helper surfaces are non-authoritative:

- [analysis_worker.py](/home/zodrak/zod/src/v3_1/agents/analysis_worker.py): episode analysis task
- [planning_helper_worker.py](/home/zodrak/zod/src/v3_1/agents/planning_helper_worker.py): candidate expansion, route-analysis, feature, hypothesis, and pruning proposals
- [coordinator_agent.py](/home/zodrak/zod/src/v3_1/agents/coordinator_agent.py): minimal wrapper, not the primary runtime driver

## Session Algorithm

[Orchestrator](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py) owns the session lifecycle.

At session start it:

1. loads persistent priors when enabled
2. snapshots the initial blackboard
3. reconciles initial working memory against the blackboard
4. initializes the mechanic graph
5. snapshots the hypothesis registry

For each round it then:

1. delegates to `RoundRunner.run_round(...)`
2. replaces latest blackboard, memory, mechanic-graph, and hypothesis snapshots
3. records round artifacts and selected targets
4. checks periodic persistent-memory flush policy
5. checks stop policy

At session end it optionally performs one final durable-memory flush and then calls [postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py).

The orchestrator also threads versioned planning contexts through the runtime:

- `blackboard_version`
- `memory_version`
- `mechanic_graph_version`
- `policy_version`
- `ranker_version`
- `plan_context_id`

## Round Algorithm

[round_runner.py](/home/zodrak/zod/src/v3_1/runtime/round_runner.py) implements the full per-round control loop.

The implemented round structure is:

1. reconstruct memory-side planning mode persistence and active subgoal-chain state
2. build a probe planning context from current snapshots
3. run probe planning
4. execute probe branches in env workers
5. analyze probe episodes
6. register deterministic and optional LLM hypothesis bundles
7. merge all probe blackboard deltas
8. merge all probe mechanic-graph deltas
9. reconcile memory from probe evidence and winner outcome
10. build a directed planning context from the updated snapshots
11. invalidate stale helper outputs if versions changed
12. optionally run helper-worker seeded planning, otherwise direct planning
13. materialize or continue runtime subgoal-chain state
14. execute directed branches
15. analyze directed episodes
16. pick a directed winner
17. merge all directed blackboard deltas
18. merge all directed mechanic-graph deltas
19. reconcile memory using the selected decision and directed winner outcome
20. persist round artifacts, memory snapshots, and ledger data

Important runtime properties:

- Probe and directed branching are capped by env-worker pool size.
- Merge stages use all branch evidence, but decision-visible outcome is based on the chosen winner.
- Compatibility invalidation is explicit through [invalidation.py](/home/zodrak/zod/src/v3_1/runtime/invalidation.py); in the current code this is structured stale-context detection rather than active task cancellation.
- Planning mode persistence is carried across rounds with hysteresis logic rather than recomputed statelessly every time.

## Planning Stack

The core planner is [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py).

The implemented planning flow is:

1. build a belief snapshot with [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py)
2. split world evidence into observed and hypothesized contracts
3. query mechanic-graph structure, unlock paths, trigger chains, exit readiness, and missing verification
4. generate candidates
5. hard/soft filter candidates
6. compute route features
7. score candidates
8. rerank candidates
9. apply service-level selection guards
10. package a `PlannerDecision`

Candidate generation in [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py) covers:

- frontier and exploration movement
- recovery and fallback actions
- target interaction and click candidates
- mechanic-graph path candidates
- deterministic and LLM-backed mechanic tests
- verification objectives such as `verify_trigger_contact`, `reobserve_remote_change`, `verify_panel_state`, and `verify_gate_match`
- chain objectives such as `trigger_then_target` and `unlock_then_exit`

Notable generation behavior:

- executable-family normalization is explicit: `move`, `interact`, `click_at`
- graph and hypothesis chains are first-step gated before the planner admits them as executable paths
- weak trigger evidence is downgraded toward probe/verification candidates instead of promoted directly to full unlock chains
- detector-backed POIs can spawn follow-up escalation candidates from `plan_memory.poi_followthrough`

Filtering, scoring, and reranking are split across:

- [candidate_filters.py](/home/zodrak/zod/src/v3_1/planning/candidate_filters.py)
- [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)
- [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)

The current scorer is still hand-authored and additive, not learned. It combines:

- utility, novelty, reachability, and route progress
- route cost/risk/uncertainty
- contradiction and support freshness
- durable priors and memory penalties
- graph support and chain coherence
- exit readiness and missing verification
- POI followthrough and probe-escalation bonuses

Planner-level guards then:

- demote low-readiness `unlock_then_exit` selections when stronger verification candidates exist
- penalize repeated stale route probes
- prefer escalation only when support, identity, or durable gain improved since earlier visits

### Subgoal Chains

Subgoal chains are now a first-class runtime structure, implemented in:

- [subgoal_chain.py](/home/zodrak/zod/src/v3_1/planning/subgoal_chain.py)
- [subgoal_chain_manager.py](/home/zodrak/zod/src/v3_1/planning/subgoal_chain_manager.py)

Each chain is a sequence of `SubgoalStep`s with:

- `step_kind`
- expected evidence
- success and failure conditions
- retry budgets
- verification points
- fallback targets
- dependency ordering

The chain builder enforces exit-chain ordering and can insert missing verification before `attempt_exit`. The manager tracks active chain state, step advancement, retries, rewrites, aborts, completions, and replan requests. Round-runner materializes chain runtime state into planner metadata and executor requests.

## Execution Stack

Execution is implemented in:

- [executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py)
- [env_worker.py](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
- [route_execution.py](/home/zodrak/zod/src/v3_1/execution/route_execution.py)
- [option_execution.py](/home/zodrak/zod/src/v3_1/execution/option_execution.py)
- [outcomes.py](/home/zodrak/zod/src/v3_1/execution/outcomes.py)
- [live_avatar_tracker.py](/home/zodrak/zod/src/v3_1/execution/live_avatar_tracker.py)
- [env_factory.py](/home/zodrak/zod/src/v3_1/execution/env_factory.py)

`build_executor_request(...)` converts the selected planner candidate into a structured execution contract containing:

- normalized action family
- navigation contract
- terminal-action contract
- stop conditions
- expected-effect contract
- active subgoal-chain and active-step metadata
- experiment and origin-hypothesis metadata

The runtime distinguishes:

- `move`: route only
- `interact`: route then terminal `ACTION5`
- `click_at`: terminal `ACTION6` at coordinates

Effect attribution is action-family aware. The runtime no longer assumes all meaningful effects are interact-style.

### Avatar Tracking

Live execution uses [live_avatar_tracker.py](/home/zodrak/zod/src/v3_1/execution/live_avatar_tracker.py), not raw env info alone. The tracker fuses:

1. motion-consistent continuation
2. action-conditioned prediction
3. validated env-provided avatar hints when available
4. static scan fallback as a low-confidence last resort

Execution telemetry exports `avatar_cell`, confidence, source, ambiguity flags, and localized failure reasons. This same avatar evidence is later reused by the analysis and export surfaces.

### Outcome Summaries

[outcomes.py](/home/zodrak/zod/src/v3_1/execution/outcomes.py) is the bridge from execution back into planning and memory. It derives:

- route progress and stall/blocked/noop summaries
- terminal success/failure evidence
- effect-region and changed-cell summaries
- counterfactual and exit-attempt evidence hooks
- experiment-result payloads
- mechanic-graph evidence hooks

## Analysis Pipeline

The authoritative episode analysis entry point is [episode_analysis.py](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py).

For each raw episode it performs:

1. observation normalization with [adapters_env.py](/home/zodrak/zod/src/v3_1/analysis/adapters_env.py)
2. per-frame summarization with [observation_summary.py](/home/zodrak/zod/src/v3_1/analysis/observation_summary.py)
3. area assignment with [area_assignment.py](/home/zodrak/zod/src/v3_1/analysis/area_assignment.py)
4. avatar tracking with [avatar_tracking.py](/home/zodrak/zod/src/v3_1/analysis/avatar_tracking.py)
5. fallback avatar repair if the traced path degenerates
6. motion summarization with [motion_analysis.py](/home/zodrak/zod/src/v3_1/analysis/motion_analysis.py)
7. action normalization against the environment action map
8. `step_rows` construction
9. topology reconstruction
10. consequence reconstruction
11. POI detection
12. pattern-descriptor annotation
13. trigger-zone extraction
14. POI identity attachment
15. target-effect attachment
16. structure-entity supplementation and promotion
17. POI canonicalization and collapse
18. blackboard-delta construction
19. mechanic-graph delta extraction
20. deterministic and optional LLM hypothesis orchestration

Important heuristics in the current analysis code:

- if avatar tracking degenerates to a single repeated cell, the analysis path falls back to active-region based propagation
- support-family emission now includes explicit counterfactual and exit-attempt rows when the classifier surface says that evidence was observed
- analysis mode matters; probe and directed analysis keep different evidence priorities and row selection behavior
- POI canonicalization is aggressive: parent/child central-region collapsing, same-level parent collapse, and cross-canonicalization between detector POIs and promoted structure POIs
- prior POI ids are adopted where possible to stabilize identity across rounds

Supporting files include:

- [poi_detection.py](/home/zodrak/zod/src/v3_1/analysis/poi_detection.py)
- [pattern_identity.py](/home/zodrak/zod/src/v3_1/analysis/pattern_identity.py)
- [entity_identity.py](/home/zodrak/zod/src/v3_1/analysis/entity_identity.py)
- [object_extraction.py](/home/zodrak/zod/src/v3_1/analysis/object_extraction.py)
- [observation_summary.py](/home/zodrak/zod/src/v3_1/analysis/observation_summary.py)
- [consequences.py](/home/zodrak/zod/src/v3_1/analysis/consequences.py)

## Blackboard And World Model

The blackboard implementation is in:

- [world/blackboard.py](/home/zodrak/zod/src/v3_1/world/blackboard.py)
- [world/merge.py](/home/zodrak/zod/src/v3_1/world/merge.py)
- [world/entities.py](/home/zodrak/zod/src/v3_1/world/entities.py)
- [world/areas.py](/home/zodrak/zod/src/v3_1/world/areas.py)
- [world/topology.py](/home/zodrak/zod/src/v3_1/world/topology.py)
- [world/consequences.py](/home/zodrak/zod/src/v3_1/world/consequences.py)
- [world/trigger_zones.py](/home/zodrak/zod/src/v3_1/world/trigger_zones.py)
- [world/reachability.py](/home/zodrak/zod/src/v3_1/world/reachability.py)
- [world/queries.py](/home/zodrak/zod/src/v3_1/world/queries.py)
- [world/indexes.py](/home/zodrak/zod/src/v3_1/world/indexes.py)

The blackboard stores versioned cumulative world state including:

- split `observed_*` and `hypothesized_*` stores for entities, consequences, topology, and trigger zones
- combined compatibility views rebuilt from those split stores
- area assignments
- topology nodes and edges
- consequences and support-family counters
- trigger zones
- reachability annotations
- planner-visible target queries

Merge is authoritative and serialized. `export_strict_snapshot(...)` is used when the runtime wants a stable, exact view for traceability or ledgering.

Notable world-model behaviors:

- delta application classifies rows as observed vs hypothesized before merge
- entity merge uses descriptor, signature, IoU, centroid distance, and kind-aware matching
- topology merge aggregates visit and transition counters
- consequence merge preserves support-family metadata, counters, and last-supported round/pass tracking
- exit-attempt and counterfactual transport diagnostics are preserved through merge and export

World queries drive planner target selection and prioritize:

- planner-visible canonical POIs
- local-area and reachable targets
- novelty/retry-aware targets
- frontier candidates

## Mechanic Graph And Hypotheses

The mechanic-graph pipeline spans:

- [analysis/mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py)
- [world/mechanic_graph.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph.py)
- [world/mechanic_graph_merge.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_merge.py)
- [world/mechanic_graph_queries.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_queries.py)
- [mechanics/deterministic_rules.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_rules.py)
- [mechanics/deterministic_hypothesis_generator.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_hypothesis_generator.py)
- [mechanics/hypothesis_orchestrator.py](/home/zodrak/zod/src/v3_1/mechanics/hypothesis_orchestrator.py)
- [mechanics/hypothesis_registry.py](/home/zodrak/zod/src/v3_1/mechanics/hypothesis_registry.py)
- [mechanics/llm_reasoner.py](/home/zodrak/zod/src/v3_1/mechanics/llm_reasoner.py)
- [mechanics/llm_prompt_builder.py](/home/zodrak/zod/src/v3_1/mechanics/llm_prompt_builder.py)
- [mechanics/llm_validator.py](/home/zodrak/zod/src/v3_1/mechanics/llm_validator.py)
- [mechanics/llm_schema.py](/home/zodrak/zod/src/v3_1/mechanics/llm_schema.py)

Mechanic-graph extraction converts analyzed POIs, effect regions, consequences, trigger contact, remote change evidence, and experiment outcomes into graph nodes and edges. The extractor uses heuristics to infer node kinds such as `exit`, `gate`, `panel`, `trigger`, and generic `poi`.

Mechanic-graph merge is semantic rather than append-only:

- nodes merge by semantic key
- edges merge by `(src, edge_kind, dst, condition_key)`
- support and contradiction counts accumulate across rounds
- evidence tier only upgrades to `observed` when direct support is present
- confidence is adjusted from support, contradiction, lag consistency, identity stability, and edge kind

The deterministic rule engine currently emits support for patterns such as:

- contact then remote change
- movement then remote change
- gate controls exit
- trigger required before exit
- trigger changes panel
- panel matches gate
- exit success after prerequisite
- direct exit failure without prerequisite
- repeated probe without effect demoting POIs
- missing-verification rules that promote verification paths before exit

The hypothesis registry tracks deterministic and LLM proposals, validation state, lifecycle state, and last-touched round. Mechanic-graph merge feeds observed support, contradictions, validation, and staleness back into that registry.

LLM integration is local-adapter based. [hypothesis_gating.py](/home/zodrak/zod/src/v3_1/runtime/hypothesis_gating.py) suppresses LLM calls when:

- LLM generation is disabled
- the per-round budget is exhausted
- a recent accepted LLM call already succeeded
- a strong non-LLM explanation already exists
- no trigger condition such as repeated failures, contradiction, ties, or graph ambiguity is met

## Memory And Durable Learning

Working memory is implemented by [skill_memory.py](/home/zodrak/zod/src/v3_1/memory/skill_memory.py) and updated through [reconcile.py](/home/zodrak/zod/src/v3_1/memory/reconcile.py).

The current memory model includes:

- cooldown state
- retry ledgers by candidate, target, and area
- exhaustion snapshots
- plan memory and compact decision/outcome history
- recovery and route-failure patterns
- skill library and execution stats
- durable priors loaded from persistent storage
- pending durable update batches

[plan_memory.py](/home/zodrak/zod/src/v3_1/memory/plan_memory.py) also records POI followthrough metrics, which are then reused by the planner to decide whether to escalate from route probing into mechanic verification or chain execution.

Durable writes are intentionally stricter than working-memory updates. `MemoryAgent.build_flush_request(...)` filters rows so only evidence that is:

- `durable_ready`
- `certifiable`
- cross-round stable
- contradiction-free
- supported by directed evidence
- not probe-only

is eligible for persistent flush.

## Ranker And Learning

The learning surface is small and optional:

- [learning/ranker_state.py](/home/zodrak/zod/src/v3_1/learning/ranker_state.py)
- [learning/score_service.py](/home/zodrak/zod/src/v3_1/learning/score_service.py)
- [learning/updates.py](/home/zodrak/zod/src/v3_1/learning/updates.py)
- [learning/feature_cache.py](/home/zodrak/zod/src/v3_1/learning/feature_cache.py)

The ranker currently acts as a lightweight score bonus service rather than a full online learner. `ranker_version` still participates in planning-context compatibility and the resulting ranker state can be persisted through the normal memory/storage path.

## Storage, Exports, And Visualization

Persistence is implemented in:

- [storage/artifact_store.py](/home/zodrak/zod/src/v3_1/storage/artifact_store.py)
- [storage/session_store.py](/home/zodrak/zod/src/v3_1/storage/session_store.py)
- [storage/persistent_memory.py](/home/zodrak/zod/src/v3_1/storage/persistent_memory.py)
- [storage/sqlite_index.py](/home/zodrak/zod/src/v3_1/storage/sqlite_index.py)
- [storage/paths.py](/home/zodrak/zod/src/v3_1/storage/paths.py)
- [storage/manifests.py](/home/zodrak/zod/src/v3_1/storage/manifests.py)
- [storage/checkpointing.py](/home/zodrak/zod/src/v3_1/storage/checkpointing.py)

The persistent-memory SQLite store tracks more than a flat skill table. The current schema includes sessions, memory snapshots, skill stats, candidate outcomes, failure and recovery patterns, POI/trigger/consequence patterns, entity and area signatures, mechanic hypotheses, ranker state, mechanic-graph nodes and edges, dependency paths, and deterministic/LLM proposal tables.

Round and postrun export assembly is handled by:

- [runtime/export_assembler.py](/home/zodrak/zod/src/v3_1/runtime/export_assembler.py)
- [runtime/postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py)
- [visualization/exports.py](/home/zodrak/zod/src/v3_1/visualization/exports.py)
- [visualization/heatmaps.py](/home/zodrak/zod/src/v3_1/visualization/heatmaps.py)
- [visualization/summaries.py](/home/zodrak/zod/src/v3_1/visualization/summaries.py)

The postrun layer writes more than just `summary.json`. It also exports:

- memory summaries and memory event streams
- session ledger views and chronology summaries
- mechanic graph snapshots and relation summaries
- deterministic and LLM hypothesis payloads, agreement summaries, and lifecycle views
- subgoal-chain timelines, step summaries, failures, and successes
- planning-mode timelines and switch summaries
- detector POI followthrough and probe-escalation summaries
- exit-readiness and premature-exit summaries
- graph-quality and identity-stability summaries
- avatar tracking traces and failures
- visit and POI heatmaps, plus optional PNG overlays

## Config, LLM Adapters, CLI, And Eval

Configuration is schema-driven through:

- [config/schema.py](/home/zodrak/zod/src/v3_1/config/schema.py)
- [config/loader.py](/home/zodrak/zod/src/v3_1/config/loader.py)
- [config/defaults.py](/home/zodrak/zod/src/v3_1/config/defaults.py)
- [config/validation.py](/home/zodrak/zod/src/v3_1/config/validation.py)
- [config/runtime.py](/home/zodrak/zod/src/v3_1/config/runtime.py)
- [config/resources.py](/home/zodrak/zod/src/v3_1/config/resources.py)
- [config/feature_flags.py](/home/zodrak/zod/src/v3_1/config/feature_flags.py)
- [config/hypothesis_generation.py](/home/zodrak/zod/src/v3_1/config/hypothesis_generation.py)

LLM adapter support is in:

- [llm/local_adapter_base.py](/home/zodrak/zod/src/v3_1/llm/local_adapter_base.py)
- [llm/local_adapter_stub.py](/home/zodrak/zod/src/v3_1/llm/local_adapter_stub.py)
- [llm/local_adapter_openai_compat.py](/home/zodrak/zod/src/v3_1/llm/local_adapter_openai_compat.py)

Primary CLI surfaces are:

- [cli/run_autonomous_game.py](/home/zodrak/zod/src/v3_1/cli/run_autonomous_game.py)
- [cli/export_reports.py](/home/zodrak/zod/src/v3_1/cli/export_reports.py)
- [cli/print_blackboard.py](/home/zodrak/zod/src/v3_1/cli/print_blackboard.py)
- [cli/run_hypothesis_eval.py](/home/zodrak/zod/src/v3_1/cli/run_hypothesis_eval.py)
- [cli/main.py](/home/zodrak/zod/src/v3_1/cli/main.py)

Offline evaluation code lives in [eval/](/home/zodrak/zod/src/v3_1/eval) and currently includes:

- graph quality
- entity identity stability
- hypothesis metrics and comparisons
- avatar tracking metrics
- probe escalation metrics
- subgoal-chain metrics
- planning-mode metrics
- planner chain-preference metrics
- runnable eval drivers for those reports

## Current Characterization

As implemented today, `v3_1` is a round-based symbolic agent runtime with:

- versioned authoritative state ownership
- probe and directed branching with serialized authoritative merges
- a heuristic but structured planner
- explicit mechanic-graph and hypothesis reasoning
- subgoal-chain execution for multi-step unlock behavior
- memory-driven retry/cooldown/exhaustion control
- strict durable-memory eligibility gating
- extensive export and evaluation instrumentation

It is not a learned end-to-end policy, and several surfaces remain intentionally heuristic:

- object/POI extraction
- avatar tracking
- mechanic-role inference
- candidate scoring/reranking
- LLM usage gating

Those heuristics are now deeply wired into the runtime, so the implementation should be understood as a symbolic controller with progressively stronger instrumentation, persistence, and chain-execution logic rather than a thin prototype.
