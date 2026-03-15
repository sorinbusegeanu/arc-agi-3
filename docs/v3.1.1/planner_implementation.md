# v3.1.1 Planner Implementation

This document describes the current native `src/v3_1/planning/` implementation as it exists today.

## Overview

The planner is a symbolic multi-stage pipeline coordinated by [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py):

1. build belief from blackboard + memory
2. generate candidate plans
3. filter invalid or weak candidates
4. compute route features
5. score candidates
6. rerank with helper input and deterministic tie-breaks
7. produce structured fallbacks
8. package one centralized final decision

The planner output is a `PlannerDecision` object. The decision metadata now also carries a full per-round planner trace.

## Main Files

- [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py)
- [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py)
- [candidate_filters.py](/home/zodrak/zod/src/v3_1/planning/candidate_filters.py)
- [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)
- [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)
- [fallbacks.py](/home/zodrak/zod/src/v3_1/planning/fallbacks.py)
- [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py)
- [decision.py](/home/zodrak/zod/src/v3_1/planning/decision.py)

## Belief Build

Belief construction happens in `build_belief(...)`.

### Inputs

- current blackboard snapshot
- current memory snapshot
- durable priors embedded in memory snapshot

### Normalized Keys

`belief_builder.py` defines shared normalization helpers used across planner stages:

- `normalized_target_key(...)`
- `normalized_route_signature(...)`
- `normalized_trigger_zone_key(...)`

These are used to keep target, route, and trigger identifiers consistent across belief rows and generated candidates.

### Belief Contents

The belief currently includes:

- `reachable_targets`
- `reachable_targets_split`
  - `directly_reachable_now`
  - `reachable_with_route`
  - `reachable_high_risk`
- `frontier_targets`
- `blocked_targets`
- `local_pois`
- `promising_pois`
- `trigger_candidates`
- `recovery_candidates`
- `local_context`
- `localized_context`
- `prior_failure_success_context`
- `durable_prior_merge`
- `cooldowns`
- `exhausted`
- `exhaustion_map`
- `exhausted_keys`
- `retries`
- `failed_candidates`
- `topology`
- `consequence_support`
- `trigger_support`
- `available_action_families`
- `versions`
- `indexes`
- `plan_memory`
- durable prior slices such as:
  - `persistent_candidate_outcomes`
  - `persistent_failure_patterns`
  - `persistent_recovery_patterns`
  - `persistent_poi_patterns`
  - `persistent_trigger_patterns`
  - `persistent_consequence_patterns`

### Reachable and Blocked Target Enrichment

Belief rows are enriched with:

- `target_key`
- `freshness`
  - `blackboard_version`
  - `memory_version`
  - `durable_prior_version`
- `contradiction_markers`
  - `stale_target`
  - `stale_trigger_support`
  - `topology_invalidation`
  - `evidence_decay`
- `provenance`
  - `source_section`
  - `supporting_refs`
  - `derived`

### Frontier Ranking

Frontier rows are enriched with:

- `frontier_type`
- `frontier_novelty`
- `expected_information_gain`
- `frontier_route_cost`

### Local and Prior Context

The planner maintains two separate local-history views:

- `local_context`
  - current area
  - area entity ids
  - recent target entity ids
  - recent decisions
- `localized_context`
  - success/failure counts by area
  - success/failure counts by local zone

It also builds `prior_failure_success_context` as an explicit object for downstream filtering and scoring.

### Durable Prior Merge Layer

The belief contains a durable advisory layer:

- `per_target`
- `per_poi_class`
- `per_trigger_type`
- `per_route_pattern`

These priors do not override current blackboard facts. They are advisory only.

## Candidate Generation

Candidate generation happens in `generate_candidates(...)`.

### Candidate Classes

The current generator supports:

- `target`
- `click_target`
- `local_probe`
- `frontier_move`
- `route_probe`
- `trigger_probe`
- `recovery_move`
- `fallback_action`

### Central Candidate Schema

Every generated candidate now uses the same schema shape. Important fields include:

- `candidate_id`
- `candidate_class`
- `target_entity_id`
- `target_area_id`
- `target_key`
- `required_action_family`
- `effect_action_family`
- `expected_progress_type`
- `route_required`
- `route_signature`
- `trigger_zone_id`
- `target_entity_class`
- `candidate_context`
  - `avatar_area`
  - `local_area`
  - `route_signature`
  - `trigger_zone_id`
  - `target_entity_class`
- `expected_outcomes`
  - `expected_state_change`
  - `expected_evidence_gain`
  - `expected_route_progress`
- `support_strength`
  - `direct_support`
  - `indirect_support`
  - `prior_support`
- `contradiction_flags`
- `stale_support_flags`
- `supporting_evidence_refs`
- `generation_source`
- `rationale`
- score-relevant fields such as:
  - `utility`
  - `novelty`
  - `confidence`
  - `distance_score`
  - `motion_score`
  - `candidate_effect_score`

### Stable Candidate IDs

Candidate ids are generated from normalized content, not incidental per-round ordering. The stable key includes:

- candidate class
- target entity id
- target area id
- route signature
- trigger zone id
- action type
- target entity class

This makes candidate IDs more stable across rounds for memory matching.

### Evidence Compression

`supporting_evidence_refs` are compressed:

- full ref set is deduplicated
- only a short sample is stored directly
- metadata is attached:
  - `supporting_evidence_ref_count`
  - `supporting_evidence_ref_sample`
  - `supporting_evidence_signature`
  - `supporting_evidence_truncated`

### Quotas and Diagnostics

Generation quotas are enforced by class so one class does not dominate all others.

Generation diagnostics currently include:

- `count_by_class`
- `dropped_during_generation`
- `unsupported_template_count`

These diagnostics are attached to generated candidates.

## Filtering

Filtering happens in `filter_candidates(...)`.

### Hard vs Soft

The filter now distinguishes:

- hard invalidation: candidate is blocked
- soft weakness: candidate survives but is downgraded

### Stable Reason Taxonomy

Reason codes are explicit and stable. Current reasons include:

Hard:

- `hard_cooldown_active`
- `hard_exhausted_scope`
- `hard_unreachable_target`
- `hard_invalid_target`
- `hard_contradiction_current_evidence`

Soft:

- `soft_repeated_failure`
- `soft_target_repeated_failure`
- `soft_local_class_repeat`
- `soft_local_target_repeat`
- `soft_route_repeat`
- `soft_trigger_repeat`
- `soft_area_repeated_failure`
- `soft_stale_support_decay`
- `soft_uncertain_contradiction`

### Filter Provenance

Every reason carries provenance metadata:

- `memory`
- `blackboard`
- `local_context`

### Scope-Aware Blocking

Cooldown and exhaustion are checked against multiple scopes:

- candidate id
- route signature
- target entity id
- target area id
- trigger zone id

### Local Repetition

The filter uses local context to penalize or block repeated failures by:

- candidate class in area
- target in area
- route signature
- trigger zone

### Audit Output

Each survivor and blocked candidate receives `filter_audit` with:

- `candidate_counts_by_class_before`
- `candidate_counts_by_class_after`
- `block_counts_by_reason`
- `downgrade_counts_by_reason`

## Scoring

Scoring happens in `score_candidates(...)`.

### Shared Score Schema

Each candidate gets:

- `score`
- `score_confidence`
- `score_uncertainty`
- `score_breakdown`

### Class-Aware Weighting

There is a per-class weight table for:

- `target`
- `click_target`
- `local_probe`
- `frontier_move`
- `route_probe`
- `trigger_probe`
- `recovery_move`
- `fallback_action`

These modify novelty, progress, and utility contributions by class.

### Current Score Terms

The scorer currently uses:

- novelty
- reachability
- progress
- retry penalty
- cooldown penalty
- exhaustion penalty
- utility
- movement/interact/click effect scores
- candidate effect score
- distance score
- motion score
- local failure risk
- neighborhood exhaustion penalty
- contradiction penalty
- support freshness
- expected progress type
- progress type score
- route risk
- route cost
- route uncertainty
- trigger bonus
- trigger uncertainty
- consequence bonus
- prior success rate
- prior failure rate
- prior route failure risk
- prior POI utility
- prior trigger bonus
- prior consequence bonus
- prior recovery usefulness
- durable prior strength

`score_breakdown` also exports the belief version set through `freshness_versions`.

### Confidence and Uncertainty

The scorer explicitly exports:

- `score_confidence`
- `score_uncertainty`

This is still heuristic, but it is now part of the candidate output.

## Reranking

Reranking happens in `rerank_candidates(...)`.

### Helper Inputs

Helper outputs can still adjust candidates through:

- `score_delta`
- `risk_delta`

### Deterministic Tie-Break Policy

After base scoring, reranking applies deterministic tie-breaks:

- reachable-now bonus
- class priority bonus
- utility bonus
- retry penalty

Class priority order is explicit:

1. `target`
2. `click_target`
3. `trigger_probe`
4. `frontier_move`
5. `route_probe`
6. `local_probe`
7. `recovery_move`
8. `fallback_action`
9. `fallback_hold`

If all other terms tie, `candidate_id` remains the final stable fallback.

### Rerank Diagnostics

Each reranked candidate carries:

- `tie_break`
- `pre_score_rank_hint`
- `pre_score_order`
- `post_score_order`
- `decisive_terms_for_winner`

## Fallbacks

Fallback selection happens in `fallback_candidates(...)`.

Important behavior:

- fallbacks use the same normalized candidate schema
- structured fallbacks can come from surviving `local_probe`, `frontier_move`, `recovery_move`, `route_probe`, or `fallback_action`
- if no valid fallback exists, a synthetic `fallback_action` hold-position row is emitted with the same schema family

This satisfies the completeness rule that fallback still uses the planner candidate schema.

## Planner Service and Trace

The full planner flow is in [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py).

It builds a `planner_trace` object containing:

- `belief`
- `generated_candidates`
- `filtered_candidates`
  - `survivors`
  - `blocked`
- `route_features`
- `score_breakdown`
- `selected_candidate`
- `summary_metrics`
- `debug_exports`
- `consistency_checks`

### Summary Metrics in Planner Trace

Current summary metrics include:

- candidates generated by class
- filtered-by-reason counts
- selected class
- score term usage
- contradiction block count
- local repeat block count

### Debug Exports in Planner Trace

Current debug exports include:

- promising POIs
- trigger candidates
- recovery candidates
- local context
- blocked targets

### Consistency Checks

The planner currently verifies:

- selected candidate is not blocked
- selected candidate is supported by current belief if it claims evidence support
- selected candidate action family is executable under current available families

## Decision Packaging

`decision.py` converts the selected candidate into the final `PlannerDecision`.

The packaged decision includes:

- selected candidate id
- selected action
- ranked candidates
- rationale
- helper proposal ids
- metadata

Decision metadata now contains:

- `selected_candidate`
- `fallback_candidates`
- `blocked_candidates`
- `planner_stats`
- `planner_trace`

This means the persisted decision artifact is now also the per-round planner trace artifact.

## Current Limitations

The planner schema is richer now, but there are still limitations:

- the quality of candidates still depends heavily on upstream analysis/world outputs
- if analysis/world produce no entities or no reachable targets, the planner degrades to fallback-heavy behavior
- some durable-prior merges are still shallow advisory mappings, not deep model-based priors
- class thresholds and weights are still heuristic constants
- route features are only as strong as the current topology/reachability layer
- movement-only regression guards are not yet a separate explicit test harness in planner code

## Practical Effect

Compared with the earlier planner version, the current implementation is now:

- more explicit in belief structure
- more consistent in candidate schema
- more diagnosable in filtering and scoring
- more stable for memory matching across rounds
- better instrumented for debug and artifact inspection

The planner is still symbolic and heuristic, but it now exposes the information needed to inspect why a candidate was generated, filtered, scored, reranked, and selected.
