# Planner Agent Flow

This document describes how the native `v3_1` planner agent currently performs the stages below in code.

Primary files:

- [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py)
- [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py)
- [candidate_filters.py](/home/zodrak/zod/src/v3_1/planning/candidate_filters.py)
- [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py)
- [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)
- [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py)

## 5. Planner agent builds belief state

Implemented in [belief_builder.py](/home/zodrak/zod/src/v3_1/planning/belief_builder.py) via `build_belief(blackboard_snapshot, memory_snapshot)`.

Inputs used:

- blackboard snapshot
- memory snapshot
- optional durable priors inside `memory_snapshot["durable_priors"]`

What it actually constructs:

- `reachable_targets`
  - from `world.queries.reachable_targets(...)`
- `frontier_targets`
  - from `world.queries.frontier_candidates(...)`
- `blocked_targets`
  - from `world.queries.unreachable_targets(...)`
- `local_pois`
  - from `world.queries.area_local_pois(...)`
- `current_area_id`
  - inferred from the last plan-memory decision if possible, otherwise first area index
- `cooldowns`
  - from working memory
- `exhausted`
  - from working memory
- `retries`
  - from working memory
- `failed_candidates`
  - built from prior unsuccessful plan-memory outcomes
- `topology`
  - nodes, edges, reachable node count
- `consequence_support`
  - grouped blackboard consequences keyed by normalized action
- `trigger_support`
  - grouped trigger zones by `entity_id`
- `indexes`
  - blackboard indexes as-is
- `plan_memory`
  - session history list
- `durable_priors`
  - full durable priors blob
- durable-prior subviews:
  - `persistent_skill_priors`
  - `persistent_candidate_outcomes`
  - `persistent_failure_patterns`
  - `persistent_recovery_patterns`
  - `persistent_poi_patterns`
  - `persistent_trigger_patterns`
  - `persistent_consequence_patterns`

How this maps to your requested items:

- reachable targets: yes
- frontier targets: yes
- blocked targets: yes
- promising POIs: approximately
  - represented as `local_pois` and also implicitly inside reachable/frontier targets if they are POI entities
- trigger candidates: indirectly
  - stored as `trigger_support`, later used during candidate generation
- recovery candidates: indirectly
  - not prebuilt as belief rows, but enough context is added for generation
- local context around avatar/current area: partially
  - current area is present; explicit avatar-neighborhood context is limited
- prior failure/success context: yes
  - retries, failed candidates, durable priors

## 6. Planner agent generates candidate actions/plans

Implemented in [candidate_generation.py](/home/zodrak/zod/src/v3_1/planning/candidate_generation.py) via `generate_candidates(...)`.

The planner does not emit just one action. It creates multiple candidate rows, then deduplicates and ranks them.

Candidate classes currently generated:

- `target`
  - target interaction
- `click_target`
  - click-at target interaction when click family is available
- `trigger_probe`
  - trigger-supported target interaction
- `frontier_move`
  - move toward frontier target
- `local_probe`
  - local area POI interaction
- `recovery_move`
  - move intended to recover route progress
- `route_probe`
  - consequence-supported probing move

Fallback action is not generated here; it is built later in [fallbacks.py](/home/zodrak/zod/src/v3_1/planning/fallbacks.py).

Each generated target candidate carries:

- target/context
  - `target_entity_id`
  - `target_area_id`
  - `centroid`
  - `required_action_family`
  - `effect_action_family`
- rationale
  - `rationale`
- expected progress/effect context
  - `movement_effect_score`
  - `interact_effect_score`
  - `click_effect_score`
  - `candidate_effect_score`
  - `distance_from_avatar`
  - `distance_score`
  - `motion_variance`
  - `motion_score`
- route need if any
  - implicitly via candidate class such as `frontier_move`, `recovery_move`, `route_probe`
  - no explicit `route_required: true/false` field exists

Notes:

- `route_probe` is a plan-like probe candidate, not a concrete route sequence.
- There is no explicit `expected_progress_type` field yet; the closest proxy is candidate class plus effect scores and route features.

## 7. Planner filters invalid or weak candidates

Implemented in [candidate_filters.py](/home/zodrak/zod/src/v3_1/planning/candidate_filters.py) via `filter_candidates(candidates, belief)`.

What it blocks:

- cooldowned candidates
  - checks cooldown on `candidate_id` and `target_entity_id`
- exhausted candidates
  - checks whether `candidate_id` is in the exhausted set
- unreachable/blocked targets
  - blocks if target is in `blocked_targets` and not `reachable_later`
- invalid current targets
  - blocks if target entity is not in blackboard reachable targets and not `reachable_later`
- repeated failed actions
  - blocks if `failed_candidates[candidate_id] >= 2`
- repeated failed targets
  - blocks if `failed_candidates[target_entity_id] >= 2`

How this maps to your requested items:

- cooldowned candidates: yes
- exhausted candidates: yes
- unreachable/blocked targets: yes
- repeated failed actions in same local context: approximately
  - repeated failure is tracked, but there is no explicit local-context key in this filter
- targets contradicted by current evidence: partially
  - handled via invalid-target / unreachable checks rather than a dedicated contradiction model

Blocked candidates are not discarded silently. They are returned separately as `blocked_candidates` with `blocked_reasons`.

## 8. Planner scores remaining candidates

Implemented in [candidate_scoring.py](/home/zodrak/zod/src/v3_1/planning/candidate_scoring.py) via `score_candidates(...)`.

Signals used directly in scoring:

- novelty
- reachability
- progress potential
  - from `route_features`
- target/POI utility
- trigger support
- consequence support
- route cost
- route risk
- retry penalty
- recovery usefulness
- durable priors if available

Additional POI-derived signals carried through the candidate:

- `movement_effect_score`
- `interact_effect_score`
- `click_effect_score`
- `candidate_effect_score`
- `distance_score`
- `motion_score`

Current scoring formula:

```text
score =
    novelty * novelty_weight
    + utility * utility_weight
    + 0.08 * candidate_effect_score
    + reachability_weight * reachability
    + progress_weight * progress
    + trigger_bonus_weight * trigger_bonus_scaled
    + consequence_bonus_weight * consequence_bonus_scaled
    + 0.12 * prior_success_rate
    + 0.06 * prior_poi_utility
    + 0.05 * prior_trigger_bonus
    + 0.04 * prior_consequence_bonus
    + 0.05 * prior_recovery_usefulness for recovery_move
    - retry_penalty_weight * retry_penalty_scaled
    - cooldown_penalty
    - exhaustion_penalty
    - route_risk_weight * route_risk
    - route_cost_weight * route_cost
    - 0.08 * prior_failure_rate
    - 0.07 * prior_route_failure_risk
```

What is included in `score_breakdown`:

- `novelty`
- `reachability`
- `progress`
- `retry_penalty`
- `cooldown_penalty`
- `exhaustion_penalty`
- `utility`
- `movement_effect_score`
- `interact_effect_score`
- `click_effect_score`
- `candidate_effect_score`
- `distance_score`
- `motion_score`
- `route_risk`
- `route_cost`
- `trigger_bonus`
- `consequence_bonus`
- `prior_success_rate`
- `prior_failure_rate`
- `prior_route_failure_risk`
- `prior_poi_utility`
- `prior_trigger_bonus`
- `prior_consequence_bonus`
- `prior_recovery_usefulness`

## Reranking and final central selection

After base scoring, the planner continues in [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py):

- applies helper boosts and penalties from helper workers
- applies tie-break preferences for:
  - `reachable_now`
  - `trigger_probe`
  - `utility`
  - retry count

Then [planner_service.py](/home/zodrak/zod/src/v3_1/planning/planner_service.py) performs the full centralized flow:

1. build belief
2. generate candidates
3. filter candidates
4. compute route features
5. score candidates
6. rerank candidates
7. build fallback set
8. select one final candidate centrally

Final selection rule:

- choose top reranked candidate if any survive
- otherwise choose top fallback candidate if available
- package the final decision in `package_decision(...)`

## Summary of gaps versus the requested ideal

Present:

- belief from blackboard + memory + durable priors
- multi-candidate generation
- explicit blocking/filtering
- central scoring
- central reranking
- central final selection

Approximate or missing:

- no explicit `promising_pois` list separate from other target lists
- no explicit `trigger_candidates` list in belief, only `trigger_support`
- no explicit `recovery_candidates` list in belief, only enough context to generate them
- repeated-failure filtering is not explicitly keyed by local spatial context
- no explicit `expected_progress_type` field on candidates
- route need is implicit rather than explicit
