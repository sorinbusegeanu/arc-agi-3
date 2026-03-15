# Interaction Effect Signals

## Native v3.1 signals

### 1. Frame change regions

file: `src/v3_1/analysis/observation_summary.py`  
variable: `changed_points`, `change_regions`

What it measures:

- per-frame pixel differences between current and previous observation
- connected changed regions with bbox, area, centroid

Where it is used:

- `summary["change_regions"]`
- `summary["active_regions"]`
- `step_rows["changed_cells"]`
- motion analysis

### 2. Step-level changed cell count

file: `src/v3_1/analysis/episode_analysis.py`  
variable: `step_row["changed_cells"]`

What it measures:

- sum of all changed-region areas for the frame

Where it is used:

- POI interaction effect accumulation in `poi_detection.py`

### 3. Motion local change

file: `src/v3_1/analysis/motion_analysis.py`  
variables:

- `local_change_area`
- `changed_cell_total`
- `action_effect_near_avatar`

What they measure:

- per-step local changed area
- total changed cells across episode
- whether change occurred near avatar centroid

Where they are used:

- movement rows
- consequence extraction in `episode_analysis.py`

### 4. POI interaction effect

file: `src/v3_1/analysis/poi_detection.py`  
variables:

- `interaction_attempts`
- `interaction_effect_sum`
- `interaction_effect_score`

What they measure:

- number of targeted interact/inspect attempts for a POI
- total changed cells across those attempts
- normalized average effect size

Where they are used:

- added into POI `utility`
- exported on POI rows
- surfaced in planner candidate score breakdown

## Related but not native v3.1

The search found richer effect signals in `src/arc_agi_agent/*`, such as:

- `changed_cells`
- `changed_bbox_area`
- `effect_rate`
- `coord_action_effect_model`

These are historical/reference implementations and are not the active v3.1 runtime path.
