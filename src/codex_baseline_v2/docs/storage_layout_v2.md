# V2 Storage Layout

Root directory: `storage.root_dir` (default `runs_v2`)

```
{root}/
  game_{game_id}/
    round_000/
      raw_trajectories/
      normalized_trajectories/
      analyst_outputs/
      blackboard_snapshots/
      controller_decisions/
      executor_outcomes/
      metrics/
      round_reports/
      exports/
      logs/
      crash_markers/
    round_001/
      ...
    blackboard_latest.json
    summary.json
```

Required directories:
- `sessions`: implicitly game-scoped directory `game_{game_id}`
- `rounds`: `round_###`
- `raw_trajectories`: raw env output or raw episode records
- `normalized_trajectories`: `TrajectoryEpisodeV2` jsonl
- `analyst_outputs`: serialized analyst summaries per round
- `blackboard_snapshots`: `blackboard.json` per round
- `controller_decisions`: controller instruction records
- `executor_outcomes`: executor outputs and step logs
- `metrics`: round metrics and V2 metrics reports
- `round_reports`: high-level summaries per round
- `exports`: reports exported for external use
- `crash_markers`: recovery/resume markers

Crash recovery markers:
- `round_###/crash_markers/resume_marker.json`
- `game_{game_id}/blackboard_latest.json`

Naming rules:
- Always include `game_id` and `round_id` in path.
- Store one primary artifact per category per round.
- All autonomous runs must only write under the V2 root directory.
