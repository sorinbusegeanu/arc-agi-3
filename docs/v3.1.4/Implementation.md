# v3.1.4 Implementation

This document describes the current implementation in `src/v3_1` as it exists in source now.

Code is treated as authoritative where older docs disagree.

## Scope

The current system is a Ray-backed multi-agent runtime with these main state owners:

- blackboard: [BlackboardAgent](/home/zodrak/zod/src/v3_1/agents/blackboard_agent.py)
- memory: [MemoryAgent](/home/zodrak/zod/src/v3_1/agents/memory_agent.py)
- mechanic graph: [MechanicGraphAgent](/home/zodrak/zod/src/v3_1/agents/mechanic_graph_agent.py)
- planner: [PlannerAgent](/home/zodrak/zod/src/v3_1/agents/planner_agent.py)
- storage: [StorageAgent](/home/zodrak/zod/src/v3_1/agents/storage_agent.py)
- execution workers: [EnvWorkerAgent](/home/zodrak/zod/src/v3_1/agents/env_worker_agent.py) running [EnvWorker](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
- analysis workers: [AnalysisWorker](/home/zodrak/zod/src/v3_1/agents/analysis_worker.py)

The coordinator path is:

- [bootstrap.py](/home/zodrak/zod/src/v3_1/runtime/bootstrap.py)
- [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)
- [round_runner.py](/home/zodrak/zod/src/v3_1/runtime/round_runner.py)

## Runtime Topology

[bootstrap.py](/home/zodrak/zod/src/v3_1/runtime/bootstrap.py) starts Ray and creates:

- one authoritative blackboard actor
- one authoritative memory actor
- one authoritative mechanic-graph actor
- one authoritative planner actor
- one authoritative storage actor
- optional ranker actor
- a pool of env workers
- analysis as a Ray task function
- helper workers as a Ray task function
- one in-process [HypothesisRegistry](/home/zodrak/zod/src/v3_1/mechanics/hypothesis_registry.py)
- one local LLM adapter

This is not only the earlier blackboard-memory-planner loop anymore. The live runtime now also carries:

- mechanic graph state
- deterministic hypothesis generation
- optional LLM hypothesis generation and validation
- hypothesis lifecycle tracking

## End-to-End Control Flow

The active session loop is driven by [Orchestrator.run](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py).

At session start it:

1. creates a [SessionLedger](/home/zodrak/zod/src/v3_1/runtime/session_ledger.py)
2. loads persistent priors through storage into memory when enabled
3. initializes blackboard
4. initializes memory
5. initializes mechanic graph

Each round is executed by [RoundRunner.run_round](/home/zodrak/zod/src/v3_1/runtime/round_runner.py).

Each round is double-pass:

1. probe plan
2. probe execution
3. probe analysis
4. probe blackboard merge
5. probe memory reconcile
6. hypothesis generation and mechanic-graph merge
7. directed plan
8. directed execution
9. directed analysis
10. directed blackboard merge
11. directed memory reconcile
12. second hypothesis generation and mechanic-graph merge
13. artifact persistence
14. round debug heatmaps

This differs from older docs that described only `observe -> analyze -> merge -> memory -> plan -> execute`.

## Blackboard and World State

The world owner is [BlackboardState](/home/zodrak/zod/src/v3_1/world/blackboard.py).

It now keeps both legacy combined stores and strict split stores:

- `observed_entities`
- `hypothesized_entities`
- `observed_consequences`
- `hypothesized_consequences`
- `observed_trigger_zones`
- `hypothesized_trigger_zones`
- `observed_topology`
- `hypothesized_topology`

It also exposes:

- `snapshot_observed()`
- `snapshot_hypothesized()`
- `snapshot_strict()`
- compatibility `snapshot()`

Strict snapshots now carry:

- split stores only
- split indexes only
- `index_contract_mode = "strict_split_native"`

Merge logic lives in [merge.py](/home/zodrak/zod/src/v3_1/world/merge.py).

Incoming rows are classified before merge with:

- `evidence_tier`
- `source_stage`
- `source_pass_id`
- `source_episode_id`
- `inference_method`
- `confidence`
- `observed_admission_reason`
- `observed_validator_name`
- `observed_validator_result`

Observed admission is no longer based on `inference_method == "direct_observation"` alone. It uses per-row-family validators for:

- areas
- entities
- consequences
- trigger zones
- topology nodes
- topology edges

The current world query layer still remains target/area/topology oriented:

- [queries.py](/home/zodrak/zod/src/v3_1/world/queries.py)

## Analysis

The main analysis entry point is [analyze_episode](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py).

It accepts:

- `analysis_mode = "probe"`
- `analysis_mode = "directed_outcome"`

The high-level flow still shares the same low-level observation utilities, but mode now changes emitted rows and priorities.

Current analysis outputs include:

- per-step summaries
- avatar tracking
- area sequence
- POIs
- consequences
- topology nodes and edges
- trigger zones
- a `BlackboardDelta`
- a `MechanicGraphDelta`
- deterministic hypothesis bundle
- optional LLM hypothesis bundle

Trigger zones are no longer left empty. They are emitted differently by mode:

- `probe`: suspicious trigger candidates from change regions
- `directed_outcome`: localized trigger attribution around acted/effect regions

All emitted blackboard rows are stamped with:

- `direct_evidence_present`
- `direct_evidence_fields`
- `contradiction_flag`
- `observation_support_span`
- `analysis_objective`

Mechanic-graph extraction now exists as a first-class analysis product in [mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py).

It builds:

- mechanic nodes such as `poi`, `trigger`, `panel`, `gate`, `exit`, `symbol_state`, `effect_region`
- mechanic edges such as `changes`, `displays`, `matches`, `requires`, `causes_remote_change`

It also produces:

- deterministic hypotheses
- optional LLM hypotheses if [hypothesis_gating.py](/home/zodrak/zod/src/v3_1/runtime/hypothesis_gating.py) permits a call

## Mechanic Graph and Hypothesis Layer

This layer is newer than the earlier v3.1 docs and is a major current-state difference.

[MechanicGraphAgent](/home/zodrak/zod/src/v3_1/agents/mechanic_graph_agent.py) owns:

- `MechanicGraphState`
- a `HypothesisRegistry`

[world/mechanic_graph.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph.py) defines:

- node kinds
- edge kinds
- mechanic graph snapshots

[world/mechanic_graph_merge.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_merge.py) merges graph deltas and updates:

- node support
- edge support
- observed vs hypothesized edge counts
- proposal support / contradiction / validation feedback

[world/mechanic_graph_queries.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_queries.py) provides graph queries such as:

- dependency paths
- exit prerequisite paths
- trigger-to-exit paths
- panel match relations
- best supported paths to exit

[HypothesisRegistry](/home/zodrak/zod/src/v3_1/mechanics/hypothesis_registry.py) tracks:

- deterministic proposals
- LLM proposals
- validation state
- lifecycle state
- support/contradiction/validation rounds
- agreement groups

This means the current source does have an explicit mechanic-graph layer, even though some older docs still describe mechanic induction as deferred.

## Planner

Planner entry point:

- [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py)

The planner now consumes:

- current blackboard snapshot
- current memory snapshot
- mechanic graph snapshot
- deterministic hypotheses
- LLM hypotheses
- hypothesis registry snapshot
- helper results

The planner still builds a large belief in [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py), but it also now creates explicit split-world planner inputs:

- `observed_world`
- `hypothesized_world`
- `uncertainty_context`
- `durable_prior_context`

Candidate pipeline:

- generation: [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py)
- filtering: [candidate_filters.py](/home/zodrak/zod/src/v3_1/planning/candidate_filters.py)
- route features: [route_features.py](/home/zodrak/zod/src/v3_1/planning/route_features.py)
- scoring: [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)
- reranking: [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)
- packaging: [decision.py](/home/zodrak/zod/src/v3_1/planning/decision.py)

The current source also includes graph-aware planner queries in:

- [planning/queries.py](/home/zodrak/zod/src/v3_1/planning/queries.py)

and uses world mechanic-graph queries directly inside [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py), including:

- best mechanic subgoal chains
- panel match dependencies
- target preconditions
- trigger-then-exit candidates
- unlock paths for exits

So the current planner is no longer only target/POI/trigger scoring. It now has a second reasoning channel based on the mechanic graph.

The planner trace now includes:

- `planner_contract_mode`
- `planning_pipeline_contract_mode`
- full candidate traces
- helper summary
- graph-related path candidates

## Helper Workers

Helper execution is coordinated by [HelperCoordinator](/home/zodrak/zod/src/v3_1/runtime/helper_coordinator.py).

Helpers are still non-authoritative. They provide:

- candidate expansion
- route analysis
- score feature computation
- hypothesis proposals
- pruning suggestions

Current helper behavior is implemented in:

- [helper_modes.py](/home/zodrak/zod/src/v3_1/planning/helper_modes.py)

Helper outputs are merged later and tracked in planner metadata; they do not directly mutate blackboard or memory.

## Execution

Environment interaction is normalized in:

- [env_factory.py](/home/zodrak/zod/src/v3_1/execution/env_factory.py)

Execution ownership is in:

- [EnvWorker](/home/zodrak/zod/src/v3_1/execution/env_worker.py)

Probe policy:

- uses available actions only
- no synthetic noop
- weighted random with anti-repeat bias

Directed policy:

- strict target-oriented routing
- bounded reroute behavior
- fail-fast on unavailable required action mapping
- does not degrade into probe mode

Executor requests are built in:

- [executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py)

The request contract now contains explicit sections like:

- objective
- navigation
- terminal action
- constraints
- stop conditions

Outcome summarization is in:

- [outcomes.py](/home/zodrak/zod/src/v3_1/execution/outcomes.py)

`outcome_evidence` now stores `{value, provenance}` and counts:

- `env_native_support_count`
- `derived_support_count`

## Memory

Working memory and durable priors are separate in:

- [skill_memory.py](/home/zodrak/zod/src/v3_1/memory/skill_memory.py)

Reconcile logic is in:

- [reconcile.py](/home/zodrak/zod/src/v3_1/memory/reconcile.py)

Current memory families include:

- cooldowns
- retries
- exhaustion
- plan memory
- skill library
- durable prior aggregates
- memory telemetry

Durable updates now carry stronger typing:

- `maturity_stage`
- `mechanic_certification_state`
- `evidence_basis`
- `observed_support_count`
- `hypothesis_support_count`
- `contradiction_count`
- `cross_round_stability`
- `directed_outcome_backed_support_count`

Flush gating is enforced by [MemoryAgent](/home/zodrak/zod/src/v3_1/agents/memory_agent.py), not by storage.

## Durable Storage

Artifact and durable persistence are separated.

Storage ownership is in:

- [StorageAgent](/home/zodrak/zod/src/v3_1/agents/storage_agent.py)

Session artifacts are written through:

- [artifact_store.py](/home/zodrak/zod/src/v3_1/storage/artifact_store.py)

Durable long-term memory is SQLite-backed in:

- [persistent_memory.py](/home/zodrak/zod/src/v3_1/storage/persistent_memory.py)

The durable DB now stores more than the original pattern families. Current code persists:

- skills
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
- mechanic graph nodes
- mechanic graph edges
- durable dependency paths
- deterministic-supported paths
- LLM-supported paths
- deterministic/LLM agreements
- repeated validated hypotheses
- contradicted LLM proposals
- deterministic hypothesis proposals
- LLM hypothesis proposals
- proposal validation state
- proposal agreement groups
- proposal outcome summaries
- ranker state

This is broader than older v3.1 docs that described only the original durable-memory families.

## Session Ledger

The runtime now maintains a typed append-only session ledger:

- [session_ledger.py](/home/zodrak/zod/src/v3_1/runtime/session_ledger.py)

It records:

- round starts
- plan selections
- episode execution
- analysis completion
- blackboard merges
- memory reconciles
- mechanic graph merges
- hypothesis generation
- LLM operations
- durable flush events
- stop decisions

Ledger payloads are validated by:

- payload type
- payload version
- payload schema name
- payload schema version

This ledger is now the preferred source of truth for chronology in post-run export.

## Post-Run Export and Visualization

Post-run export lives in:

- [postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py)

It writes:

- session summary
- memory summary
- JSON heatmap payloads
- PNG visualizations
- session ledger artifact

Visualization uses:

- [heatmaps.py](/home/zodrak/zod/src/v3_1/visualization/heatmaps.py)
- [summaries.py](/home/zodrak/zod/src/v3_1/visualization/summaries.py)

Current visualization behavior includes:

- first-screen map PNG at session level
- visit heatmap overlay and debug PNG
- POI heatmap overlay and debug PNG
- per-round debug heatmaps
- post-run heatmaps built from accumulated episodes

The export layer is ledger-first for chronology and version linkage, but still uses:

- episodes for heatmap/image payloads
- round records for some memory-summary details

## Current Source vs Older Docs

Older docs under `docs/v3.1`, `docs/v3.1.1`, `docs/v3.1.2`, and `docs/v3.1.3` are still useful, but they do not fully capture the current source.

The main differences visible in code now are:

1. mechanic graph is live, not deferred
2. hypothesis generation and LLM gating are live
3. planner uses mechanic-graph queries directly
4. durable storage schema is broader than the earlier described persistent-memory families
5. session ledger now tracks more than round sequencing
6. the active round loop is more than probe+directed blackboard/memory reconcile; it also includes mechanic-graph and hypothesis stages

Where there is disagreement, the source files in `src/v3_1` should be treated as authoritative.
