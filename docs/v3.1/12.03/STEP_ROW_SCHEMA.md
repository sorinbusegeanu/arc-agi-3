# Step Row Schema

## Source

file: `src/v3_1/analysis/episode_analysis.py`  
location: `step_rows.append(...)`

## Fields actually present in each `step_row`

- `step_idx`
- `action`
- `action_type`
- `target_entity_id`
- `reward`
- `done`
- `area_id`
- `state_hash`
- `avatar_centroid`
- `avatar_cell`
- `object_ids`
- `changed_cells`
- `change_region_count`

## Field meanings

- `step_idx`
  - step index within the raw episode
- `action`
  - raw action payload from `RawStep.action`
- `action_type`
  - lowercased `action["type"]` if action is a dict, else `""`
- `target_entity_id`
  - `action["target_entity_id"]` or fallback `action["target"]`
- `reward`
  - raw step reward
- `done`
  - raw step done flag
- `area_id`
  - area assigned to the frame
- `state_hash`
  - state identity hash from observation summary
- `avatar_centroid`
  - chosen avatar track centroid for the step
- `avatar_cell`
  - integer cell version of `avatar_centroid`
- `object_ids`
  - object IDs present in the frame summary
- `changed_cells`
  - sum of `change_regions[*].area`
- `change_region_count`
  - number of change regions in the frame
