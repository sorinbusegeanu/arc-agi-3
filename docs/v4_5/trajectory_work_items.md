# v4.5 Trajectory Work Items

## Purpose

Manage bounded trajectory and action attempts as explicit work items.

## Owners

- Planner creates and ranks
- Outcome updates

## Entities

- `TrajectoryWorkItem`
- `TrajectoryQueue`
- `TrajectoryOutcome`

## Statuses

- `pending`
- `running`
- `succeeded`
- `failed`
- `blocked`
- `needs_replan`
- `superseded`
- `closed`

## Required Fields

- `work_item_id`
- `game_id`
- `level_index`
- `poi_id`
- `subgoal_id`
- `plan_prefix`
- `expected_contact_or_effect`
- `priority`
- `created_round`
- `attempt_count`
- `status`
- `rationale_codes`

## Rules

- Planner executes only the top ranked bounded work item, not all candidates
- failed items can spawn a replacement item instead of mutating the old item in place
