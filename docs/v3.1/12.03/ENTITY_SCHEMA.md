# Entity Schema

## Entity creation / merge sources

### Analysis-side entity creation

file: `src/v3_1/analysis/poi_detection.py`

POIs are emitted as entity-like rows before world merge with fields:

- `entity_id`
- `poi_id`
- `kind`
- `signature`
- `centroid`
- `bbox`
- `area`
- `bbox_area`
- `bbox_area_ratio`
- `primary_color`
- `type_hints`
- `poi_class`
- `persistence`
- `utility`
- `novelty`
- `interaction_attempts`
- `interaction_effect_sum`
- `interaction_effect_score`
- `distance_from_avatar`
- `distance_score`
- `motion_variance`
- `motion_score`
- `confidence`
- `observations`
- `demotion_reasons`
- `canonical_descriptor`

`canonical_descriptor` contains:

- `signature`
- `kind`
- `primary_color`
- `bbox_size`

### World-side entity representation

file: `src/v3_1/world/entities.py`

After merge, entities may additionally contain:

- `stable_entity_id`
- `history`
- `evidence_refs`
- `first_seen_round`
- `last_seen_round`
- `first_seen_episode`
- `last_seen_episode`
- `merge_matches`
- `lifecycle_state`
- `stale_steps`

The merge keeps or updates:

- `entity_id`
- `stable_entity_id`
- `bbox`
- `centroid`
- `confidence`
- `observations`
- historical metadata

## Stable entity identity basis

file: `src/v3_1/world/entities.py`  
function: `_stable_entity_id(incoming)`

Stable ID hash input:

- `signature`
- `canonical_descriptor` or `stable_descriptor`
- `kind`
- `primary_color`
