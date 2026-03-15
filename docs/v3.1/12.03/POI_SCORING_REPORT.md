# POI Scoring Report

## Primary implementation

file: `src/v3_1/analysis/poi_detection.py`  
function: `detect_pois(step_summaries, avatar_tracking, step_rows=None)`

This function computes POI acceptance, `utility`, `novelty`, and `confidence` inline.

### Input features used

- `stats["persistence"]`
- `stats["count"]`
- `exemplar["confidence"]`
- `exemplar["kind"]`
- `exemplar["type_hints"]`
- `exemplar["touches_border"]`
- `exemplar["area"]`
- `exemplar["bbox"]`
- `exemplar["primary_color"]`
- `signature in avatar_signatures`
- `background_colors`
- `grid_width`, `grid_height`, `grid_area`
- `bbox_width`, `bbox_height`, `bbox_area`, `bbox_area_ratio`
- `area_ratio`
- `avatar_cell` from first step row
- `motion_variance`, `motion_score` from per-signature centroid history
- `interaction_attempts`, `interaction_effect_sum`, `interaction_effect_score` from step rows

### Hard rejection rules

The candidate is rejected if any of these apply:

- `exemplar["kind"] == "hud_like"`
- `stats["persistence"] < 0.2`
- `bbox_area_ratio > 0.12`
- `bbox_width > 0.45 * grid_width`
- `bbox_height > 0.45 * grid_height`
- `primary_color in background_colors and bbox_area_ratio > 0.03`
- `area_ratio > 0.08 and persistence > 0.6`
- `area < 1 or area > 25`

### Demotion rules

Demotion reasons reduce confidence but do not reject by themselves:

- `signature in avatar_signatures` -> `avatar_like`
- `touches_border and kind != "mobile_candidate"` -> `border_touching`
- `area <= 2` -> `tiny`

### Utility formula

`utility` starts at `0.0` and is incremented by:

```text
utility += min(0.45, persistence)
utility += min(0.25, exemplar_confidence)
utility += 0.15 if "candidate_avatar" not in type_hints else 0.0
utility += 0.20 * distance_score
utility += 0.25 * motion_score
utility += 0.35 * interaction_effect_score
```

### Novelty formula

`novelty` starts at `0.0` and is incremented by:

```text
novelty += 0.2 if observation_count == 1 else 0.0
novelty += 0.15 if primary_color not in background_colors else 0.0
```

### Distance score formula

Constants:

- `MAP_DIAGONAL = 90.0`

Formula:

```text
distance_from_avatar = sqrt((cx - avatar_x)^2 + (cy - avatar_y)^2)
distance_norm = distance_from_avatar / MAP_DIAGONAL
distance_score = 1.0 - min(1.0, distance_norm)
```

### Motion score formula

Constants:

- `MOTION_NORMALIZER = 15.0`

Formula:

```text
variance_x = mean((cx_i - mean_cx)^2)
variance_y = mean((cy_i - mean_cy)^2)
motion_variance = variance_x + variance_y
motion_score = min(1.0, motion_variance / MOTION_NORMALIZER)
```

### Interaction effect formula

Constants:

- `EFFECT_NORMALIZER = 50.0`

For each `step_row`:

```text
if row.target_entity_id == poi_id
and row.action_type in {"interact", "inspect", "inspect_local"}:
    interaction_attempts += 1
    interaction_effect_sum += row.changed_cells
```

Then:

```text
interaction_effect_score = 0.0
if interaction_attempts > 0:
    interaction_effect_score = interaction_effect_sum / interaction_attempts
interaction_effect_score = min(1.0, interaction_effect_score / EFFECT_NORMALIZER)
```

### Confidence formula

```text
confidence =
    max(
        0.0,
        utility
        + novelty
        - 0.2 * len(demotion_reasons)
        - 0.4 * len(rejection_reasons)
    )
```

Note: rejected candidates are dropped before emission.

## Secondary POI-related scoring / filtering

file: `src/v3_1/visualization/heatmaps.py`  
function: `build_poi_heatmap(blackboard_state, ...)`

This is export filtering, not POI detection. It uses:

- `confidence`
- `observations`
- `utility`
- `lifecycle_state`

Stable-for-export rule:

```text
stable_flag =
    confidence >= 0.35
    and observations >= 2
    and not explicitly_rejected
    and lifecycle_state in {"active", "stale"}
```
