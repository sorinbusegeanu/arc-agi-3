# Missing Signals Report

## interaction_effect

present: yes  
where expected: analysis POI scoring, summary metrics, planner score breakdown  
why missing or limited:

- present in `src/v3_1/analysis/poi_detection.py`
- limited because it depends on `step_rows.target_entity_id == poi_id`
- current POI IDs are episode-local `poi:*` hashes, while executed action targets often reference planner/world entity IDs, so matches can be sparse or zero

## distance_to_avatar

present: yes  
where expected: analysis POI scoring, planner advisory context  
why missing or limited:

- present in `src/v3_1/analysis/poi_detection.py` as `distance_from_avatar` and `distance_score`
- uses first step `avatar_cell` only, not a richer route or reachability estimate

## object_motion

present: yes  
where expected: POI scoring and analysis  
why missing or limited:

- present in `src/v3_1/analysis/poi_detection.py` as `motion_variance` and `motion_score`
- computed only from centroid variance for objects sharing a signature
- no explicit motion direction, velocity model, or action-conditioned object dynamics

## pixel_change_after_action

present: partially  
where expected: step rows, effect modeling, planner/action learning  
why missing or limited:

- present as `change_regions`, `changed_cells`, `local_change_area`, and `total_changed_cells`
- not present as a richer structured transition effect model in v3.1
- no direct per-action pixel diff summary object comparable to the older `arc_agi_agent` diff/effect reporting

## entity-target alignment for interaction signals

present: no, not robustly  
where expected: POI interaction outcome attribution  
why missing:

- analysis POIs use episode-local `poi:*` IDs
- planner/execution target references come from world-merged entity IDs
- there is no native reconciliation layer that maps executed target IDs back onto analyzed POI IDs before scoring interaction effect

## direct planner use of new POI sub-signals

present: partially  
where expected: candidate scoring formula  
why missing:

- `interaction_effect_score`, `distance_score`, and `motion_score` are present in candidate rows and `score_breakdown`
- they are not independent additive terms in `candidate_scoring.py`
- they only affect planner score via upstream POI `utility`
