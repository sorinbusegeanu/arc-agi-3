# Candidate Scoring Report

## Primary implementation

file: `src/v3_1/planning/candidate_scoring.py`  
function: `score_candidates(candidates, belief, route_features, planning_cfg)`

## Features used

- `novelty`
- `utility`
- `reachability`
- `progress`
- `retry_penalty`
- `cooldown_penalty`
- `exhaustion_penalty`
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

Additional POI-derived features currently carried into the breakdown:

- `interaction_effect_score`
- `distance_score`
- `motion_score`

These three are present in the breakdown, but they are not direct additive terms in the planner score formula. They currently influence ranking indirectly through `utility`, because POI detection folds them into POI utility upstream.

## Exact score formula

```text
score =
    novelty * planning_cfg.novelty_weight
    + utility * planning_cfg.utility_weight
    + planning_cfg.reachability_weight * reachability
    + planning_cfg.progress_weight * progress
    + planning_cfg.trigger_bonus_weight * trigger_bonus_scaled
    + planning_cfg.consequence_bonus_weight * consequence_bonus_scaled
    + 0.12 * prior_success_rate
    + 0.06 * prior_poi_utility
    + 0.05 * prior_trigger_bonus
    + 0.04 * prior_consequence_bonus
    + 0.05 * prior_recovery_usefulness_if_recovery_move
    - planning_cfg.retry_penalty_weight * retry_penalty_scaled
    - cooldown_penalty
    - exhaustion_penalty
    - planning_cfg.route_risk_weight * route_risk
    - planning_cfg.route_cost_weight * route_cost
    - 0.08 * prior_failure_rate
    - 0.07 * prior_route_failure_risk
```

Where:

- `reachability = 1.0` if reachable now, `0.45` if reachable later, else `-0.5`
- `retry_penalty = 0.18 * (candidate_retry_count + target_retry_count)`
- `trigger_bonus = 0.08 * len(trigger_support[target_entity_id])`
- `consequence_bonus = 0.05 * len(consequence_support[action])`

Scaled terms are normalized back to count-like units before multiplying by config weights.

## Score breakdown fields emitted

- `novelty`
- `reachability`
- `progress`
- `retry_penalty`
- `cooldown_penalty`
- `exhaustion_penalty`
- `utility`
- `interaction_effect_score`
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

## Other ranking stages

### Reranking

file: `src/v3_1/planning/reranking.py`  
function: `rerank_candidates(...)`

Uses:

- `candidate["score"]`
- helper `score_delta` / `score_penalty`
- tie-breaks from:
  - `confidence`
  - `reachable_now`
  - `utility`

### Final plan selection

file: `src/v3_1/planning/planner_service.py`

Flow:

- belief build
- candidate generation
- candidate filtering
- route features
- score candidates
- rerank candidates
- fallback selection
- final decision
