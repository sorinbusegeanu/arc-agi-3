# Deterministic Analysis, Mechanic Graph, Hypothesis Generation, and Planner Subgoal Chains

This document describes the current implementation in `src/v3_1` as it exists now.

It covers four connected parts of the system:

1. deterministic analysis
2. mechanic graph construction and merge
3. hypothesis generation
4. how the planner chooses a mechanic-oriented subgoal chain

If older docs disagree with the code, the code is the source of truth here.

## 1. Deterministic analysis

### Main entry point

The deterministic mechanic-analysis path is entered from:

- [mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py)

The deterministic hypothesis generator itself is:

- [deterministic_hypothesis_generator.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_hypothesis_generator.py)

### What it consumes

`generate_deterministic_hypotheses(...)` takes:

- `raw_episode`
- `analyzed_episode`
- `mechanic_graph_snapshot`
- `blackboard_snapshot`

The generator does not reason directly over raw pixels. It reasons over normalized event-like structures derived from:

- executed episode steps
- analyzed episode summaries
- current mechanic graph state
- current blackboard state

### Event normalization

The first step is:

- `normalize_events(...)` from [event_normalizer.py](/home/zodrak/zod/src/v3_1/mechanics/event_normalizer.py)

This builds the deterministic-reasoning input representation. The deterministic rules run on that normalized event layer rather than on raw episode rows directly.

### Deterministic rule families

The generator applies a fixed set of rule functions from:

- [deterministic_rules.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_rules.py)

Current edge-rule producers:

- `contact_then_remote_change(events)`
- `movement_then_remote_change(events)`
- `pattern_equality_match(events)`
- `gate_controls_exit(events)`
- `trigger_required_before_exit(events)`
- `trigger_changes_panel(events)`
- `panel_matches_gate(events)`

Current path-rule producers:

- `trigger_to_exit_dependency_path(events)`
- `movement_change_dependency_path(events)`
- `exit_success_after_prerequisite(events)`
- `direct_exit_failure_without_prerequisite(events)`

These produce deterministic proposals as typed hypothesis objects, not direct graph mutations.

### Scoring and ranking

After rule emission:

- edge proposals are deduplicated by `proposal_id`
- path proposals are deduplicated by `proposal_id`
- each proposal is scored by:
  - `score_deterministic_proposal(...)`

from:

- [deterministic_scoring.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_scoring.py)

The score is written back into proposal `confidence` and also into `proposal.metadata`.

Then proposals are sorted by:

1. higher confidence
2. more support refs
3. stable proposal id

### Deterministic tests

After scoring, the system generates deterministic test proposals through:

- `generate_deterministic_tests(...)`

from:

- [deterministic_tests.py](/home/zodrak/zod/src/v3_1/mechanics/deterministic_tests.py)

So deterministic reasoning currently produces:

- deterministic edge proposals
- deterministic path proposals
- deterministic test proposals

all wrapped in one:

- `HypothesisBundle`

with `provenance="deterministic_hypothesis"`.

### What deterministic analysis is doing in practice

The deterministic layer is not a learned model. It is a rule-based proposal generator that reconstructs candidate causal structure from:

- contacts
- delayed changes
- remote effects
- pattern matches
- exit failure/success conditions

Its output is advisory until validated by later graph evidence or execution outcomes.

## 2. Mechanic graph

### Main state owner

The authoritative mechanic graph lives in:

- [mechanic_graph_agent.py](/home/zodrak/zod/src/v3_1/agents/mechanic_graph_agent.py)

This Ray actor owns:

- `MechanicGraphState`
- a local `HypothesisRegistry`

### Core graph data model

The graph schema is defined in:

- [mechanic_graph.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph.py)

Node kinds currently include:

- `poi`
- `trigger`
- `panel`
- `gate`
- `exit`
- `symbol_state`
- `region`
- `effect_region`

Edge kinds currently include:

- `changes`
- `displays`
- `matches`
- `controls_access`
- `opens`
- `requires`
- `causes_remote_change`
- `enables_exit`
- `contradicts`

The graph state stores:

- `nodes_by_id`
- `edges_by_id`
- `adjacency_out`
- `adjacency_in`
- `object_to_node_indexes`
- `pattern_to_node_indexes`
- `observed_edge_ids`
- `hypothesized_edge_ids`

### Graph extraction from an analyzed episode

Mechanic-graph delta construction happens in:

- [mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py)

This file is doing three jobs together:

1. emitting mechanic graph nodes and edges from the analyzed episode
2. generating deterministic hypotheses
3. optionally generating LLM hypotheses

### How nodes are created

Current graph nodes are derived primarily from POIs and localized change regions.

Important constructors:

- `_node_from_poi(...)`
- `_effect_region_node(...)`

Current logic:

- POIs become graph nodes
- `pattern_id` on a POI can create a `symbol_state` node
- direct effect regions can become `effect_region` nodes

Node IDs are stable symbolic IDs such as:

- `mg:exit:...`
- `mg:gate:...`
- `mg:panel:...`
- `mg:trigger:...`
- `mg:effect_region:...`

### How edges are created

Current extraction emits edges from localized evidence heuristics, not from a single graph-learning algorithm.

Examples from [mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py):

- `changes`
  - from trigger contact evidence followed by remote/effect-region change
- `displays`
  - from a node to a derived `symbol_state`
- `matches`
  - when multiple nodes share the same `pattern_id`
- `requires`
  - when repeated contact-to-change support suggests a precondition before exit
- `causes_remote_change`
  - from delayed change evidence

The extraction layer mixes:

- observed edges
- hypothesized edges

based on evidence confidence and directness.

### Graph merge

The authoritative merge logic is:

- [mechanic_graph_merge.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_merge.py)

Important merge behavior:

- nodes merge by `semantic_key` / stable node id
- edges merge by `(src_node_id, edge_kind, dst_node_id, condition_key)`
- support counts accumulate
- contradiction counts accumulate
- confidence is updated incrementally
- evidence tier can stay or upgrade according to `_merge_tier(...)`

Important edge tracking fields after merge:

- `support_count`
- `contradiction_count`
- `observed_support_count`
- `hypothesized_support_count`
- `origin_provenance`
- `supporting_hypothesis_ids`
- `validated_from_hypothesis_id`
- `validation_round_ids`

### Hypothesis feedback from graph merge

The graph merge is not isolated from hypothesis state.

`merge_mechanic_graph_delta(...)` matches merged edges back to registered hypothesis proposals using:

- `_matching_hypothesis_ids(...)`

Then it emits feedback buckets:

- `supported_proposal_ids`
- `contradicted_proposal_ids`
- `validated_proposal_ids`

The mechanic graph agent uses those to update the `HypothesisRegistry`.

### Mechanic graph queries

Graph query helpers live in:

- [mechanic_graph_queries.py](/home/zodrak/zod/src/v3_1/world/mechanic_graph_queries.py)

Important queries:

- `find_dependency_paths(...)`
- `find_exit_prerequisite_paths(...)`
- `find_trigger_to_exit_paths(...)`
- `find_match_relations_for_panel(...)`
- `best_supported_paths_to_exit(...)`

These are the main graph-to-planner bridge. There is no separate symbolic planner graph engine beyond these queries plus candidate generation.

## 3. Hypothesis generation

### The two live hypothesis sources

The current system has two hypothesis sources:

1. deterministic hypotheses
2. LLM hypotheses

Both are proposal bundles, and both are later reconciled against graph evidence and execution outcomes.

### Shared hypothesis type system

The shared proposal schema is defined in:

- [hypothesis_types.py](/home/zodrak/zod/src/v3_1/mechanics/hypothesis_types.py)

Current proposal types:

- `HypothesisEdgeProposal`
- `HypothesisPathProposal`
- `HypothesisTestProposal`

All are grouped into:

- `HypothesisBundle`

Each proposal carries:

- stable `proposal_id`
- provenance
- source/destination nodes
- support refs
- contradiction refs
- confidence
- novelty score
- validation requirement flags
- round / episode provenance
- metadata

### Deterministic hypothesis generation

This path is described above in section 1.

It is enabled by:

- `hypothesis_generation.enable_deterministic`

in:

- [config_def.conf](/home/zodrak/zod/src/v3_1/config_def.conf)

### LLM hypothesis generation

The LLM path is assembled in:

- [mechanic_graph_extraction.py](/home/zodrak/zod/src/v3_1/analysis/mechanic_graph_extraction.py)

The current flow is:

1. compute deterministic bundle
2. compute gating inputs:
   - repeated failures
   - contradiction level
   - deterministic tie status
   - graph ambiguity
3. call:
   - `should_call_llm(...)`
4. if allowed:
   - build focused LLM prompt input
   - call the local adapter
   - validate returned proposals

### Gating

Gating logic is in:

- [hypothesis_gating.py](/home/zodrak/zod/src/v3_1/runtime/hypothesis_gating.py)

Current gate conditions include:

- LLM enabled flag
- call budget per round
- no recent accepted LLM call
- no already-strong observed / validated deterministic explanation
- one of:
  - no deterministic path proposals
  - repeated failures
  - contradictions over threshold
  - tied deterministic options
  - high graph ambiguity

There is also an explicit prompt-size skip reason:

- `prompt_too_large_after_trimming`

### Focused prompt builder

The focused prompt path is in:

- [llm_prompt_builder.py](/home/zodrak/zod/src/v3_1/mechanics/llm_prompt_builder.py)

Current behavior:

- chooses one prompt mode:
  - `hypothesis_for_exit`
  - `resolve_contradiction`
  - `suggest_experiment`
- chooses one query target
- extracts only a focused local neighborhood
- emits compact symbolic rows instead of a full graph snapshot
- hard-limits nodes, edges, paths, contradictions, exit attempts, pattern relations, and allowed node ids

### LLM reasoner

The LLM call wrapper is:

- [llm_reasoner.py](/home/zodrak/zod/src/v3_1/mechanics/llm_reasoner.py)

Current behavior:

- maps task role to prompt mode
- builds or accepts a focused payload
- serializes the final user payload to JSON
- computes:
  - prompt char count
  - approximate token count
- trims if needed
- skips if still over budget
- sends one system message and one JSON user message through the local adapter

### Local adapter

The current local adapter is:

- [local_adapter_openai_compat.py](/home/zodrak/zod/src/v3_1/llm/local_adapter_openai_compat.py)

Current behavior:

- uses OpenAI-compatible `/chat/completions`
- sends:
  - one system instruction
  - one user JSON string
- requires JSON output
- rejects empty content and `<think>` content
- injects prompt diagnostics into returned metadata

### LLM validation

LLM output is validated in:

- [llm_validator.py](/home/zodrak/zod/src/v3_1/mechanics/llm_validator.py)

This layer:

- checks schema shape
- checks output keys
- rejects wrapper/prose contamination
- caps confidence
- converts returned raw JSON into validated `HypothesisBundle` rows

### Hypothesis registry

The persistent in-session proposal registry is:

- [hypothesis_registry.py](/home/zodrak/zod/src/v3_1/mechanics/hypothesis_registry.py)

It tracks:

- deterministic proposals
- LLM proposals
- validation state
- lifecycle state
- first support / contradiction / validation rounds
- source agreement groups

Registry update methods include:

- `register_bundle(...)`
- `mark_supported_from_graph_evidence(...)`
- `mark_contradicted_from_graph_evidence(...)`
- `mark_validated_from_path_success(...)`
- `mark_stale(...)`

So the current implementation does not treat hypotheses as disposable one-shot suggestions. They are registered, updated across rounds, and fed back by graph evidence.

## 4. Planner chooses subgoal chain

### There is no standalone subgoal-chain module

There is no dedicated `subgoal_chain.py` in the current source.

The current subgoal-chain behavior is distributed across:

- [planning/queries.py](/home/zodrak/zod/src/v3_1/planning/queries.py)
- [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py)
- [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py)
- [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)
- [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)
- [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py)

### Planner-side graph queries

Planner graph query wrappers are in:

- [planning/queries.py](/home/zodrak/zod/src/v3_1/planning/queries.py)

Important planner-facing helpers:

- `query_unlock_paths_for_exit(...)`
- `query_trigger_then_exit_candidates(...)`
- `query_panel_match_dependencies(...)`
- `query_required_preconditions_for_target(...)`
- `query_best_mechanic_subgoal_chain(...)`

The main chain-selection query is:

- `query_best_mechanic_subgoal_chain(...)`

which delegates to:

- `best_supported_paths_to_exit(...)`

from the mechanic-graph query layer.

### What “best supported path” means right now

`best_supported_paths_to_exit(...)`:

- searches trigger-to-exit dependency paths
- computes path strength from edge confidence
- counts contradictions
- tracks whether the path is hypothesis-only
- ranks paths by:
  1. non-hypothesis-only first
  2. fewer contradictions
  3. higher support strength

So the planner’s current subgoal chain is not a hand-authored task tree. It is a ranked dependency path over the mechanic graph.

### Belief construction

Belief construction is in:

- [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py)

The belief builder pulls mechanic graph information into belief through:

- `find_nodes_by_kind(..., "exit")`
- `find_nodes_by_kind(..., "trigger")`
- `find_edges_by_kind(..., "matches")`
- `best_supported_paths_to_exit(...)`

It writes these into belief as:

- `mechanic_graph_view.supported_exits`
- `mechanic_graph_view.trigger_candidates`
- `mechanic_graph_view.match_relations`
- `mechanic_graph_view.paths_to_exit`

and also into top-level convenience fields used by the planner:

- `belief["mechanic_graph_paths_to_exit"]`
- `belief["mechanic_graph_trigger_candidates"]`
- `belief["mechanic_graph_match_relations"]`

### Planner service wiring

Planner orchestration is in:

- [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py)

The planner service:

1. builds belief
2. builds split planner contract:
   - `observed_world`
   - `hypothesized_world`
   - `uncertainty_context`
3. queries mechanic graph paths:
   - `query_best_mechanic_subgoal_chain(...)`
   - `query_trigger_then_exit_candidates(...)`
   - `query_panel_match_dependencies(...)`
4. injects those graph-derived rows into belief
5. runs candidate generation, filtering, scoring, reranking, and packaging

### Candidate generation for subgoal chains

Mechanic-chain candidates are generated in:

- [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py)

There are three live mechanic-graph-derived candidate families:

1. graph path candidates
2. graph trigger candidates
3. panel/match verification candidates

Current graph-path candidate generation:

- reads `belief["mechanic_graph_paths_to_exit"]`
- for each path:
  - uses `node_ids` as `prerequisite_chain`
  - uses `edge_ids` as supporting graph edges
  - sets `graph_hop_count`
  - marks `hypothesized_only_dependency` from `path.hypothesis_only`
  - generates:
    - `objective_type="unlock_then_exit"`
    - `execution_mode="move"`
    - `navigation_mode="routed"`
    - `generation_source="mechanic_graph.paths_to_exit"`

Current graph-trigger candidate generation:

- reads `belief["mechanic_graph_trigger_candidates"]`
- makes:
  - `objective_type="unlock_trigger"`
  - routed movement/interact candidate
  - `prerequisite_chain=[trigger_node_id]`

Current panel-match candidate generation:

- reads `belief["mechanic_graph_match_relations"]`
- makes:
  - `objective_type="verify_panel_state"`
  - routed candidate around a panel node

### Deterministic and LLM hypothesis candidates

The planner also turns registered hypothesis proposals directly into candidates.

Current deterministic-hypothesis candidate families:

- `mechanic_chain_deterministic`
- `mechanic_test_deterministic`

Current LLM-hypothesis candidate families:

- `mechanic_chain_llm`
- `mechanic_test_llm`

These candidates carry:

- `supporting_hypothesis_ids`
- `prerequisite_chain`
- `graph_hop_count`
- `expected_information_gain`
- `requires_validation`
- `target_node_ids`
- `discriminates_between_proposal_ids`

So the planner is not only using confirmed graph paths. It is also willing to consider chain-like deterministic or LLM hypotheses as candidate plans.

### How the planner scores chain candidates

Scoring is in:

- [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)

Mechanic/subgoal-chain related terms include:

- `graph_observed_bonus`
- `graph_path_bonus`
- `graph_pattern_bonus`
- `graph_hypothesis_penalty`
- `graph_long_chain_penalty`
- `graph_stale_penalty`
- `deterministic_priority_bonus`
- `validated_llm_bonus`
- `llm_only_penalty`
- `contradiction_hypothesis_penalty`
- `seed_requires_hypothesis_penalty`
- `zero_observed_support_penalty`

Important consequences:

- shorter, observed, supported graph paths are preferred
- fully hypothesized chains are penalized
- deterministic validated chains are preferred over unvalidated LLM chains
- long chains are penalized
- stale or contradicted chains are penalized

### What the selected subgoal chain looks like in the planner

The selected mechanic-oriented candidate usually carries:

- `candidate_class`
- `objective_type`
- `target_entity_id`
- `prerequisite_chain`
- `supporting_graph_node_ids`
- `supporting_graph_edge_ids`
- `graph_hop_count`
- `supporting_hypothesis_ids`

This is the current concrete representation of a “subgoal chain” in `v3_1`.

It is not a separate explicit chain object. It is a ranked planner candidate whose metadata contains the current dependency chain.

### Reranking and final choice

After scoring:

- [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)

does helper-aware reranking and deterministic tie-breaking.

The reranker explicitly inspects:

- `prerequisite_chain`
- `candidate_class`

and gives structured preference to some chain-like candidates such as:

- `unlock_then_exit`
- `trigger_then_target`

Finally:

- [decision.py](/home/zodrak/zod/src/v3_1/planning/decision.py)

packages the winning candidate into the final planner decision.

## Summary

Current implementation summary:

- deterministic analysis is rule-based event reconstruction over normalized episode events
- mechanic graph is a merged symbolic graph of POIs, effect regions, panels, gates, exits, triggers, and their relations
- hypothesis generation is dual-source:
  - deterministic proposals
  - optional focused LLM proposals
- planner “subgoal chains” are currently represented as ranked graph-path and hypothesis candidates carrying `prerequisite_chain` and graph support metadata

The important practical point is this:

the planner already does chain-shaped reasoning, but it does so through:

- mechanic graph queries
- graph-derived candidate generation
- chain-aware scoring penalties and bonuses

not through a standalone explicit subgoal-chain planner module.
