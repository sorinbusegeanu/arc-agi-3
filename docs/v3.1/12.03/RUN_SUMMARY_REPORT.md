# Run Summary Report

## Source files

- `src/v3_1/visualization/summaries.py`
- `src/v3_1/runtime/postrun_exports.py`

## Summary output location

Current behavior:

- session-level summary is written to `runs_v3_1/session_<id>/summary.json`

## Base summary formula

file: `src/v3_1/visualization/summaries.py`  
function: `build_run_summary(...)`

Inputs:

- `rounds_completed`
- `won`
- `latest_blackboard_version`
- `latest_memory_version`
- `unique_target_entity_ids`
- `total_number_of_entities`

Formula:

```text
percentage_targets_solved =
    unique_target_entity_ids / total_number_of_entities
    if total_number_of_entities > 0
    else 0.0
```

Fields written:

- `rounds_completed`
- `won`
- `latest_blackboard_version`
- `latest_memory_version`
- `unique_target_entity_ids`
- `total_number_of_entities`
- `percentage_targets_solved`

## Additional effect metrics

file: `src/v3_1/runtime/postrun_exports.py`  
function: `export_postrun(...)`

Inputs:

- `selected_target_entity_ids`
- `blackboard_state["entities"]`
- each targeted entity's `interaction_effect_score`

Computation:

```text
target_ids = unique(selected_target_entity_ids)
targeted_entities =
    final_blackboard_entities[target_id]
    for target_id in target_ids
    if target_id exists in final_blackboard_entities

effectful_targets =
    targeted_entities with interaction_effect_score > 0.0
```

Formulas:

```text
percentage_targets_with_effect =
    len(effectful_targets) / len(targeted_entities)
    if targeted_entities else 0.0

average_effect_strength =
    sum(entity.interaction_effect_score for entity in targeted_entities)
    / len(targeted_entities)
    if targeted_entities else 0.0
```

Fields added after base summary:

- `percentage_targets_with_effect`
- `average_effect_strength`
