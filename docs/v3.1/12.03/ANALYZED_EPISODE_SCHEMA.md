# AnalyzedEpisode Schema

## Source

file: `src/v3_1/analysis/episode_analysis.py`  
function: `analyze_episode(raw_episode)`

## `AnalyzedEpisode` top-level fields

From `src/v3_1/contracts/messages.py`:

- `session_id`
- `run_id`
- `game_id`
- `round_id`
- `pass_id`
- `episode_id`
- `raw_episode_id`
- `summary`
- `objects`
- `avatar_tracks`
- `points_of_interest`
- `areas`
- `motion`
- `blackboard_deltas`
- `metadata`

## `AnalyzedEpisode.summary` keys actually present

Created in `src/v3_1/analysis/episode_analysis.py`.

- `step_count`
- `won`
- `total_reward`
- `main_track_id`
- `avatar_visits`
- `area_sequence`
- `step_rows`
- `background_colors`
- `state_hashes`

## `AnalyzedEpisode.metadata` keys actually present

- `step_summaries`
- `avatar_tracking`
- `motion`

## Other populated collections

- `objects`
  - object rows derived from per-frame object extraction, each enriched with `step_idx` and `area_id`
- `avatar_tracks`
  - exported avatar track rows from `avatar_tracking.py`
- `points_of_interest`
  - emitted POI rows from `poi_detection.py`
- `areas`
  - canonical area rows from `area_assignment.py`
- `motion`
  - one episode-level motion row plus per-step movement rows
- `blackboard_deltas`
  - one `BlackboardDelta` currently emitted per analyzed episode
