## v4.5 Benchmark Data Model

### Game catalog entry

Represents one known game in the predefined benchmark catalog.

Required fields:

- `game_id`
- `title`
- `description`
- `family`
- `in_benchmark`
- `notes`
- `created_at`
- `updated_at`

### Benchmark run

Represents one benchmark session across one or more games.

Required fields:

- `run_id`
- `run_label`
- `started_at`
- `finished_at`
- `solver_version`
- `runtime_mode`
- `notes`

### Benchmark game result

Represents the aggregate result for one game inside one benchmark run.

Required fields:

- `run_id`
- `game_id`
- `attempted`
- `levels_seen`
- `levels_solved`
- `total_steps_executed`
- `solved_levels_total_steps`
- `unsolved_levels_total_steps`
- `terminal_success`
- `terminal_failure`
- `status`
- `failure_reason`
- `created_at`

### Benchmark level result

Represents the normalized result for one level in one benchmark run.

Required fields:

- `run_id`
- `game_id`
- `level_index`
- `attempted`
- `solved`
- `steps_executed`
- `terminal_status`
- `failure_reason`
- `solution_action_count`
- `created_at`

### Derived best results

Derived summaries are stored separately for fast lookup.

`game_best_results` fields:

- `game_id`
- `best_levels_solved`
- `best_solved_levels_total_steps`
- `best_total_steps_for_best_solved`
- `best_run_id`
- `updated_at`

`level_best_results` fields:

- `game_id`
- `level_index`
- `best_steps_executed`
- `best_run_id`
- `updated_at`

## Best-result update rules

### Game best summary

For a game, the better run is chosen using persisted normalized game results only.

Rule order:

1. higher `levels_solved` is better
2. if tied, lower `solved_levels_total_steps` is better
3. if still tied, lower `total_steps_executed` is better
4. if still tied, keep the earlier best run

### Level best summary

For a level, the better result is chosen using persisted normalized level results only.

Rule order:

1. a solved level result beats an unsolved result
2. among solved results, lower `steps_executed` is better
3. if tied, keep the earlier best run
