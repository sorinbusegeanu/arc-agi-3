# v3.1.5 Implementation

This document describes the current `src/v3_1` implementation as it exists in code on 2026-03-17. It is an implementation note, not a product spec. Where the system uses heuristics or fallback logic, those are described explicitly.

## Overview

`v3_1` is implemented as a round-based symbolic control loop with five persistent authoritative state surfaces:

1. environment execution state inside env workers
2. blackboard world state
3. working memory state
4. mechanic graph state
5. hypothesis registry state

The top-level controller is [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py). Per round execution is delegated to [round_runner.py](/home/zodrak/zod/src/v3_1/runtime/round_runner.py). The runtime uses Ray, but only some stages are parallelized; authoritative merges remain serialized.

## Session Algorithm

At session start, the orchestrator:

1. loads persistent priors if enabled
2. snapshots the initial blackboard
3. initializes working memory by reconciling against the initial blackboard
4. initializes the mechanic graph
5. snapshots the hypothesis registry

The main session loop then repeats for `round_id = 1..max_rounds`:

1. call `RoundRunner.run_round(...)`
2. replace the latest blackboard, memory, mechanic graph, and hypothesis snapshot handles
3. append round records and selected target ids
4. evaluate periodic durable flush policy
5. evaluate stop policy

At session end, the orchestrator calls [postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py) to write run-level outputs such as:

- `summary.json`
- `memory_summary.json`
- `mechanic_graph.json`
- deterministic and LLM hypothesis comparison files
- session ledger export

## Round Algorithm

The per-round controller is [RoundRunner](/home/zodrak/zod/src/v3_1/runtime/round_runner.py).

The live control flow is:

1. build a planning context from the current blackboard, memory, and mechanic-graph snapshots
2. run a probe planning phase
3. execute probe branches in parallel env workers
4. analyze probe episodes in parallel
5. merge all probe blackboard deltas
6. merge all probe mechanic-graph deltas
7. reconcile memory for the probe pass
8. run the directed planning phase
9. execute directed branches in parallel env workers
10. analyze directed episodes in parallel
11. choose a winning directed branch
12. merge all directed blackboard deltas
13. merge all directed mechanic-graph deltas
14. reconcile memory for the directed pass
15. export round artifacts

### Probe and Directed Branching

Branching is controlled by:

- `planning.probe_branch_count`
- `planning.directed_trial_count`

The branch count is clamped by the env worker pool size. Candidate branching is implemented by:

- selecting top candidates
- padding with repeated trial variants if distinct candidate count is smaller than the configured branch count

This means branch diversity is a mix of:

- distinct planner candidates
- repeated candidate trials with different branch seeds

### Winner Selection

The directed winner is selected after all directed branches finish. The winner is then treated as the authoritative directed outcome for:

- memory reconcile
- exported selected action
- round stop outcome

The merge stages still merge evidence from all directed analyses, but the planner-visible selected decision is derived from the winner.

## Planning Implementation

### Planning Context

Planning context is assembled in [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py) and includes:

- blackboard snapshot/version
- memory snapshot/version
- mechanic graph snapshot/version
- deterministic and LLM hypothesis handles
- policy and ranker versions

Planner inputs are therefore snapshot-based and read-only for the duration of a planning pass.

### Candidate Generation

[candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py) generates multiple families of candidates:

- frontier movement
- recovery movement
- fallback actions
- mechanic graph candidates
- deterministic and LLM hypothesis test/chain candidates

Executable family assignment is normalized to:

- `move`
- `interact`
- `click_at`

and is constrained by executable evidence and available action families.

For mechanic-oriented candidates, `v3.1.5` now applies explicit first-step gating before admitting full chains:

- graph-path candidates use first-step executability and graph-node quality
- deterministic and LLM chain candidates apply the same strong first-node gate
- weak trigger evidence is downgraded toward `unlock_trigger` / `route_probe` behavior rather than admitted as `unlock_then_exit`

This was added after live `ls20` runs showed synthetic trigger regions being selected as full prerequisite chains and then failing immediately on `go_to_trigger -> blocked`.

### Candidate Scoring

[candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py) uses additive heuristic scoring. The score combines:

- utility and novelty
- progress estimate
- route cost and uncertainty
- memory priors such as retry/cooldown/exhaustion and prior success/failure
- graph support terms
- deterministic/LLM source-aware bonuses and penalties

This is not a learned policy. It is a hand-authored weighted heuristic with structured score breakdowns.

### Reranking

[reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py) applies final preference logic after raw scoring. Reranking explicitly prefers:

- coherent prerequisite chains
- observed support over hypothesized support
- deterministic support over unvalidated LLM support

and demotes:

- contradicted chains
- long unsupported chains
- LLM-only chains when deterministic support exists

## Execution Implementation

### Executor Requests

[executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py) constructs execution requests that preserve:

- normalized action family
- target entity or coordinates
- route target or centroid
- experiment and hypothesis metadata
- origin hypothesis ids

Execution requests are semantic wrappers around low-level env actions; they are not hierarchical planners themselves.

### Directed Execution

Directed execution uses normalized action families:

- `move`: route only, no terminal interact/click
- `interact`: route, then emit terminal `ACTION5`
- `click_at`: emit `ACTION6` at coordinates

The terminal-distance rules are config-driven:

- `move_terminal_distance_cells`
- `interact_terminal_distance_cells`
- `click_terminal_distance_cells`

### Outcome Summarization

[outcomes.py](/home/zodrak/zod/src/v3_1/execution/outcomes.py) derives structured outcome evidence from:

- raw step sequence
- route telemetry
- analysis hints written back into execution info

It computes:

- route progress
- blocked/stalled/noop conditions
- terminal success/failure evidence
- effect-region and changed-cells evidence
- experiment result summaries
- mechanic-graph evidence hooks

The outcome object is the authoritative bridge from execution back to memory and later planner evidence.

## Analysis Implementation

### Episode Analysis Pipeline

[episode_analysis.py](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py) transforms a `RawEpisode` into an `AnalyzedEpisode`.

The implemented pipeline is:

1. normalize observations
2. summarize each observation
3. assign areas
4. track avatar position
5. apply fallback avatar tracking if the path degenerates
6. summarize motion and movement rows
7. normalize each executed action through the environment action map
8. build `step_rows`
9. reconstruct topology from `step_rows`
10. reconstruct flat consequences
11. detect POIs
12. annotate pattern descriptors
13. extract trigger zones
14. build a `BlackboardDelta`
15. extract mechanic-graph delta and hypothesis bundles

### Observation and Area Assignment Algorithms

The observation analysis path is deterministic and heuristic:

- observation normalization converts env-native observations into a common grid representation
- observation summary computes objects, active regions, change regions, background color, and state identity
- area assignment groups observations into area identities using stable signatures and reuse rules

This is a symbolic spatial summarization pipeline, not a learned perception stack.

### Avatar Tracking Algorithm

Avatar tracking is based on per-step object summaries. If the track degenerates to a single repeated cell, `_fallback_avatar_tracking(...)` switches to a local heuristic:

- use active regions as candidate avatar proxies
- sort by region area and distance to previous location
- propagate the chosen centroid

So avatar tracking is a best-effort heuristic with a change-region fallback.

At execution time, avatar localization is no longer taken from `info["avatar"]` as the primary source. The live execution path now uses [live_avatar_tracker.py](/home/zodrak/zod/src/v3_1/execution/live_avatar_tracker.py), which maintains runtime avatar belief from:

1. motion-consistent continuation
2. action-conditioned motion inference
3. direct env info only when present and usable
4. static scan fallback only as a low-confidence last resort

Execution telemetry exports:

- `avatar_cell`
- `avatar_confidence`
- `avatar_source`
- `avatar_ambiguous`

and downstream outcome certainty is capped when avatar localization is weak.

### Topology Reconstruction Algorithm

`_topology_from_steps(...)` in [episode_analysis.py](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py) builds:

- nodes keyed by avatar grid cell
- edges keyed by `(src_cell, dst_cell, normalized_action_key, transition_type)`

Algorithm:

1. convert each valid `avatar_cell` into `cell:x:y`
2. count visits per node
3. for consecutive cells, create an edge
4. choose `transition_type = "move"` when `action_family == "move"`, else `"action"`
5. accumulate `success_count` and evidence refs

This is a direct transition graph extracted from avatar movement, not a planning graph search.

### Consequence Reconstruction Algorithm

Flat consequence rows are reconstructed in `_consequences(...)` by zipping:

- motion movement rows
- raw episode reward/done signals
- normalized step transport

A consequence row is emitted when there is:

- local change area
- reward
- terminal state

Each consequence preserves:

- normalized action transport
- step index
- evidence refs

This consequence model is flat and local. It is separate from the mechanic graph.

### POI Detection Algorithm

[poi_detection.py](/home/zodrak/zod/src/v3_1/analysis/poi_detection.py) implements heuristic POI detection.

Algorithm:

1. summarize object persistence across step summaries
2. reject likely HUD/background/oversized/floor structures using hard thresholds
3. demote border-touching or tiny objects
4. compute distance-from-avatar score
5. compute motion score from centroid variance
6. compute effect scores by normalized action family
7. combine persistence, confidence, motion, distance, and effect into utility/confidence
8. deduplicate by color and rounded centroid

Important thresholds:

- `EFFECT_NORMALIZER = 50.0`
- `MAP_DIAGONAL = 90.0`
- `MOTION_NORMALIZER = 15.0`

This is explicitly a rule-based POI detector.

In the current code, avatar-like objects are suppressed more aggressively than in earlier `v3.1.x` iterations:

- `candidate_avatar` / `mobile_candidate` objects are not promoted into ordinary POIs unless they also have interaction support or remote-effect support
- compact object extraction thresholds are used so small world objects are not automatically treated as avatar proxies

This reduced false mechanic-role assignment from mobile/avatar-like blobs, but it also means the remaining bottleneck is upstream object-backed structure recall rather than planner scoring.

### Movement Effect Attribution

The current system computes effect by normalized action family:

- `movement_effect_score`
- `interact_effect_score`
- `click_effect_score`
- `candidate_effect_score`

In `v3.1.5`, the run-level summary and final selected-target exports are corrected using actual round analysis counters:

- `move_steps_count`
- `movement_steps_with_change`
- and the equivalent interact/click counters

This avoids relying only on earlier episode-local POI ids when the stable selected target is known.

## Blackboard Implementation

### State Structure

The blackboard state is owned by [blackboard.py](/home/zodrak/zod/src/v3_1/world/blackboard.py) and merged through [merge.py](/home/zodrak/zod/src/v3_1/world/merge.py).

Top-level combined stores are:

- `areas`
- `entities`
- `consequences`
- `trigger_zones`
- `topology_nodes`
- `topology_edges`
- `indexes`

It also maintains observed and hypothesized split stores separately.

### Merge Algorithm

`apply_delta(...)` in [merge.py](/home/zodrak/zod/src/v3_1/world/merge.py) is the authoritative merge algorithm.

For each delta:

1. classify every row as `observed` or `hypothesized`
2. validate observed-eligibility using row-kind-specific validators
3. merge areas, entities, consequences, trigger zones, and topology separately
4. prune consequences to `max_consequences`
5. combine observed and hypothesized stores into live combined views
6. recompute reachability
7. propose trigger zones from merged entities and consequences
8. rebuild indexes

Observed classification is conservative:

- rows must pass validator checks
- direct evidence must be present
- contradictions block observed admission

The blackboard also maintains strict split snapshots:

- `snapshot_observed()`
- `snapshot_hypothesized()`
- `snapshot_strict()`

The compatibility snapshot path is still present for older consumers, but the planner and ledger now record strict split-world references as the primary versioned contract.

### Entity Merge Algorithm

[entities.py](/home/zodrak/zod/src/v3_1/world/entities.py) uses heuristic entity matching.

Matching score:

- signature equality: `+0.55`
- canonical descriptor equality: `+0.2`
- bbox IoU contribution: `+0.25 * IoU`
- centroid proximity contribution: `max(0, 0.2 - distance/20)`
- kind equality: `+0.05`

If best match score `< 0.65`, a new stable entity id is allocated.

This is a greedy nearest-match merge, not a global assignment algorithm.

### Index Construction

Indexes are built from merged state and include:

- entities by area
- POIs by area and type
- reachable, blocked, and frontier targets
- consequence by normalized action key
- evidence index

The code uses normalized action family/name keys, not raw action strings.

## Memory Implementation

The memory owner is [SkillMemoryState](/home/zodrak/zod/src/v3_1/memory/skill_memory.py).

Working memory contains:

- skill library
- plan memory
- cooldowns
- retries
- exhaustion state
- telemetry

### Memory Reconcile Algorithm

The reconcile algorithm is deterministic and update-based:

1. copy previous working memory
2. optionally advance cooldowns on directed reconcile
3. update retry ledgers from selected decision/outcome
4. apply failure cooldowns
5. update exhaustion
6. rebuild or reuse skill library depending on pass
7. update skill execution stats
8. update plan memory
9. append memory telemetry events
10. build durable update batches

Probe pass and directed pass are intentionally different:

- probe pass avoids expensive rebuild paths when possible
- directed pass performs the full update path

### Retry, Cooldown, and Exhaustion Algorithms

These are rule-based tactical memory mechanisms.

- retries count repeated failed selections by scope
- cooldowns are active for configured remaining rounds
- exhaustion tracks repeatedly bad regions/targets/candidates

The planner consumes these via:

- hard blocking for some cases
- soft penalties in candidate scoring

### Skill Library Algorithm

The skill library is rebuilt from current world context and then updated by execution stats.

Skill identity continuity is currently heuristic:

- stable when upstream entity identity and target context stay stable
- less stable when trigger or area identities churn

Skill execution stats are keyed by:

- explicit `skill_id` when available
- otherwise normalized execution family plus target context fallback

## Mechanic Graph Implementation

### Ownership

The mechanic graph is separately owned by [MechanicGraphAgent](/home/zodrak/zod/src/v3_1/agents/mechanic_graph_agent.py). Planner, memory, and analysis do not directly mutate cumulative graph state.

### State and Merge

Graph state lives in:

- [mechanic_graph.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph.py)
- [mechanic_graph_merge.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_merge.py)

Nodes and edges track:

- stable ids
- evidence tier
- confidence
- support and contradiction counts
- source round/episode provenance
- first/last seen rounds

Edge identity is semantic:

- `(src_node_id, edge_kind, dst_node_id, condition_key)`

Merge rules:

- repeated support strengthens an edge
- contradiction increments contradiction count
- hypothesized edges do not auto-upgrade to observed without direct evidence

### Graph Extraction Algorithm

[mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py) converts analyzed episodes into graph deltas and hypothesis bundles.

Current extraction covers:

- trigger contact
- remote change candidates
- pattern display and match candidates
- gate and exit control candidates
- prerequisite candidates

The current extractor now uses analysis-stamped direct evidence when constructing graph nodes:

- graph node `evidence_tier` is derived from `factual_observation` / `direct_evidence_present` when the pre-merge row has not yet been classified
- trigger-zone nodes use stable region-based ids rather than per-step directed ids
- delta `entities` rows are merged into graph seeding alongside raw POIs

This was added because pre-merge graph extraction was previously reading only raw POIs and per-step trigger ids, which caused:

- directly observed trigger evidence to be treated as hypothesized-only
- unstable trigger node identities
- spurious prerequisite chains from repeated synthetic region nodes

### Current Trigger-Chain Safety Rules

`v3.1.5` now explicitly suppresses unsafe mechanic-chain inference from synthetic trigger regions.

Current safety behavior:

- heuristic `requires(trigger -> exit)` and `requires(trigger -> gate)` edges are only added for object-backed trigger nodes
- purely synthetic trigger-region nodes do not accumulate exit-link support or counterfactual support
- graph merge only feeds exit-link quality back into eligible source nodes
- deterministic and LLM chain candidates cannot bypass these graph-side safety rules

This change was driven by repeated `ls20` runs where synthetic trigger nodes such as `mg:trigger:trigger:6:0` or per-step directed trigger zones were accumulating enough heuristic support to become selected `mechanic_chain_deterministic` candidates and immediately fail on blocked `go_to_trigger` steps.

### Current Live `ls20` State

The latest validated safe run is [session_6828cd99](/home/zodrak/zod/runs_v3_1/session_6828cd99).

For that run:

- `heuristic_trigger_exit_edges = 0`
- `path_to_victory_candidates.json` contains `0` candidates
- `subgoal_chains.json` contains `0` events
- observed trigger nodes are present (`127`)
- object-backed graph nodes remain extremely sparse (`1`)
- graph edges are effectively only `displays: 1`

So the current system is in a safer but sparser state:

- bogus synthetic trigger chains are suppressed
- trigger evidence is present and stable enough for probing
- but the graph still lacks enough object-backed trigger/panel/gate structure to support safe executable chains in `ls20`

That means the dominant remaining limitation is no longer planner wiring. It is upstream recall of non-avatar, object-backed structural entities.

## Hypothesis Generation Implementation

There are two hypothesis sources:

1. deterministic generator
2. optional LLM advisory generator

Both use shared proposal records from `src/v3_1/mechanics/hypothesis_types.py`.

### Deterministic Generator Algorithm

[deterministic_hypothesis_generator.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_hypothesis_generator.py) runs:

1. event normalization
2. deterministic rule application
3. duplicate aggregation
4. support and contradiction scoring
5. discriminating-test generation
6. confidence sorting

Rule templates live in [deterministic_rules.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_rules.py).

This is a symbolic rule engine, not a learned relation predictor.

In current live behavior, deterministic proposal volume is still much larger than accepted path quality. The graph and planner therefore rely on additional downstream gating to stop symbolic overreach from weak trigger-region evidence.

### LLM Advisory Path

The LLM path is optional and fail-open.

## Current Bottleneck

After the latest `v3.1.5` safety fixes, the main limitation is:

- the system observes many trigger-like regions
- but almost no stable non-avatar object-backed entities beyond the exit

As a result:

- trigger probing still exists
- full mechanic chains are usually suppressed as unsafe
- planner behavior falls back to `unlock_trigger`, frontier, recovery, or fallback actions instead of executing reliable prerequisite chains

The dominant modules for the remaining bottleneck are:

- [object_extraction.py](/home/zodrak/zod/src/v3_1/analysis/object_extraction.py)
- [observation_summary.py](/home/zodrak/zod/src/v3_1/analysis/observation_summary.py)
- [poi_detection.py](/home/zodrak/zod/src/v3_1/analysis/poi_detection.py)
- [pattern_identity.py](/home/zodrak/zod/src/v3_1/analysis/pattern_identity.py)

So the current implementation is best described as:

- safe against the earlier synthetic-trigger chain bug
- capable of stable trigger-region evidence accumulation
- still weak at surfacing the object-backed symbolic structure needed for strong mechanic-path planning

Implemented pieces:

- local adapter contract
- OpenAI-compatible local adapter
- structured prompt builder
- strict validator
- gating logic

LLM outputs remain hypothesis-tier only until validated later by deterministic or observed evidence.

## Postrun Export Algorithms

[postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py) builds:

- run summary
- memory summary
- mechanic graph exports
- hypothesis comparison exports
- visualization payloads

### Session Summary Algorithm

Run summary is based on:

- final blackboard and memory versions
- unique selected target ids
- targeted stable entities
- per-round fallback effect reconstruction from `round_records`

The important `v3.1.5` behavior is:

- if stable target entities do not already carry effect scores, the summary reconstructs target effect from actual directed round analysis counters

This is why final run summaries are now aligned with real movement-change behavior.

### Memory Summary Algorithm

Memory summary prefers direct telemetry first:

- memory write events
- retry increments
- cooldown set/clear
- exhaustion set/clear
- recovery and route-failure writes
- skill-stat updates

It falls back to round-record inference only when direct telemetry does not exist.

## Parallelism

Current Ray usage is mixed:

- env execution: parallel
- analysis: parallel task fan-out
- helper tasks: parallel task fan-out
- blackboard merge: serialized
- memory reconcile: serialized
- mechanic graph merge: serialized

So the system is not fully data-parallel. It is a fan-out/fan-in architecture around env and analysis, with serialized authoritative commit barriers.

## Current Implementation Notes

The system currently uses several explicit heuristic corrections because stable identity and episode-local attribution do not always align cleanly:

- run summary movement effect uses authoritative round analysis fallback
- final selected target exports are stamped from round-level effect counters when winner-local targeting is sparse
- mechanic hypotheses for movement-only games are synthesized from movement-to-change evidence

These are implementation-level corrections, not abstract design goals.

## Known Weaknesses

The current `v3.1.5` implementation still has several important limitations:

1. effect attribution is partly corrected late

- run summary and final selected-target exports are now aligned with actual directed movement-change behavior
- but some earlier intermediate entity rows can still depend on later correction logic instead of clean upstream attribution

2. entity identity is heuristic, not guaranteed

- stable entity merge is greedy and threshold-based
- if signatures or centroids drift, continuity can break or merge quality can degrade

3. planner scoring is still heuristic and additive

- there is no learned calibration of score weights
- many bonuses and penalties are hand-tuned and can interact in brittle ways

4. mechanic graph remains sparse in simple movement-only games

- deterministic movement-change hypotheses now exist
- but rich prerequisite chains still require stronger trigger/panel/gate evidence than many runs produce

5. hypothesis validation is conservative

- deterministic and LLM proposals remain hypothesis-tier until supported later
- this avoids unsafe promotion, but it also means many proposals remain unvalidated for long periods

6. snapshot consistency is coordinator-based, not transactional

- planning uses versioned snapshots
- but the system is still orchestrator-coordinated rather than using a formal transactional state machine

7. Ray parallelism is partial

- env and analysis fan out
- authoritative merge and memory reconcile remain serialized barriers
- so CPU scaling is bounded by the round structure

8. some export values are reconstructed rather than purely native

- especially in postrun summaries
- when upstream fields are incomplete, the exporter falls back to authoritative round evidence instead of leaving values incorrect

9. the mechanic graph is currently safer than it is expressive

- after the latest trigger-chain safety fixes, synthetic trigger regions no longer seed bogus prerequisite chains
- but this also means many `ls20` runs end with a graph that is safe but structurally sparse
- in the latest safe run, the graph contains many observed trigger-region nodes but almost no object-backed non-avatar nodes
- as a result, there are no path-to-victory candidates and no executable subgoal chains, even though trigger evidence exists

10. trigger evidence is abundant but mostly region-backed, not object-backed

- the system can now accumulate stable trigger-region identities across rounds
- however, most of those trigger nodes are still region/change based rather than grounded in stable world-object identity
- that makes them useful for `unlock_trigger` probing, but not trustworthy enough for trigger->gate->exit chain inference

11. object-backed structural recall is the dominant upstream bottleneck

- beyond the exit, the current pipeline still surfaces very few stable non-avatar objects
- panel/gate/trigger object identity is therefore too weak for robust mechanic graph construction
- this bottleneck currently sits upstream of planner quality, in:
  - object extraction
  - observation summary
  - POI detection
  - pattern identity

12. planner contract migration is still partial

- live ledger records still show `planner_contract_mode = split_world_native_partial`
- so while generation/scoring have been moved far toward split-world inputs, the planner stack is not yet fully isolated from compatibility pathways end-to-end

13. trigger-zone generation is still high-volume and noisy

- probe mode emits many suspicious-region trigger zones
- directed mode emits many localized-attribution trigger zones
- even with safer stable ids, this can flood the mechanic graph with trigger nodes that are evidence-bearing but not semantically rich
- downstream safety gates now suppress the worst chain failures, but they do not solve the underlying trigger overproduction

14. chain execution is currently correctness-first, not throughput-first

- the chain manager now correctly refuses unsafe chain continuation
- but once safe gating removes weak prerequisite paths, the planner often falls back to `unlock_trigger`, recovery, or fallback actions
- this avoids false chain execution, but it also means that mechanic-oriented progress can disappear entirely in sparse-structure games

15. round artifacts can be misleading if read without ledger context

- compatibility snapshots and some persisted per-round JSON artifacts do not always expose the same truth surface as the strict split snapshot refs in the session ledger
- the ledger is currently the better source for authoritative per-round state counts and stage transitions
- reading raw round artifacts alone can therefore understate what was actually present in strict observed/hypothesized state during the run

## Current Issues

The most important current issues, based on the latest validated `ls20` runs, are:

1. no reliable object-backed prerequisite structure

- the latest safe run still has only one object-backed graph node
- this prevents robust trigger/panel/gate/exit chain formation

2. trigger probing works better than chain formation

- observed trigger-region evidence is present and stable
- but it mostly drives `unlock_trigger` style candidates rather than executable prerequisite chains

3. planner behavior is conservative because graph safety is now strict

- the earlier bad behavior was bogus `mechanic_chain_deterministic` selection from synthetic trigger regions
- that bug is fixed
- the replacement behavior is conservative probing and fallback, because the graph no longer manufactures unsafe paths

4. safe graph state can look underpowered in `ls20`

- current safe runs often end with:
  - `heuristic_trigger_exit_edges = 0`
  - `path_to_victory_candidates = 0`
  - `subgoal_chains = 0`
- this is intentional relative to the current evidence gates, but it also means the system still lacks the upstream structure needed to recover useful mechanic chains safely

## Files Most Relevant To v3.1.5

- [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)
- [round_runner.py](/home/zodrak/zod/src/v3_1/runtime/round_runner.py)
- [episode_analysis.py](/home/zodrak/zod/src/v3_1/analysis/episode_analysis.py)
- [poi_detection.py](/home/zodrak/zod/src/v3_1/analysis/poi_detection.py)
- [merge.py](/home/zodrak/zod/src/v3_1/world/merge.py)
- [entities.py](/home/zodrak/zod/src/v3_1/world/entities.py)
- [skill_memory.py](/home/zodrak/zod/src/v3_1/memory/skill_memory.py)
- [mechanic_graph_agent.py](/home/zodrak/zod/src/v3_1/agents/mechanic_graph_agent.py)
- [postrun_exports.py](/home/zodrak/zod/src/v3_1/runtime/postrun_exports.py)
