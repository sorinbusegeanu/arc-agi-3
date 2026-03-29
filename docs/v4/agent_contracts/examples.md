Status: implemented and verified
Scope: agent contracts doc: examples
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Contract Examples

All examples below are grounded in local engine and wrapper behavior observed from:

- `FrameDataRaw` returned by local environments
- `env.info`
- `env.action_space`
- local ARCEngine test game classes used in the repository

## Reset Observation

Raw source shape summary:

- local offline `ez01` reset returns `FrameDataRaw`
- `state = NOT_FINISHED`
- `full_reset = true`
- `available_actions = [1, 2, 3, 4]`
- `frame` contains one plane with shape `64 x 64`

Normalized v4 representation:

- `raw_object_name = FrameDataRaw`
- `game_id = ez01-63be02fb`
- `state = NOT_FINISHED`
- `levels_completed = 0`
- `win_levels = 5`
- `full_reset = true`
- `available_actions = (1, 2, 3, 4)`

Authoritative fields:

- `game_id`
- `frame`
- `state`
- `levels_completed`
- `win_levels`
- `guid`
- `full_reset`
- `available_actions`

Absent fields that must stay absent:

- reward
- truncation
- POIs
- planner beliefs

## Legal Primitive Action Step

Raw source shape summary:

- local offline `ez01` allows `ACTION1`
- one real step with `ACTION1` returns `FrameDataRaw`
- post-step `state` remains `NOT_FINISHED`
- post-step `full_reset = false`

Normalized v4 representation:

- action: `action_id = 1`, `action_name = ACTION1`, `payload = null`
- legality: true because `1` is present in pre-step `available_actions`
- post observation: `state = NOT_FINISHED`

Authoritative fields:

- pre-step `available_actions`
- executed action id and name
- post-step observation

Absent fields that must stay absent:

- semantic action label such as “up”
- planner rationale

## Illegal Action Example

Raw source shape summary:

- local offline `ez01` reset exposes `available_actions = [1, 2, 3, 4]`
- `ACTION7` is not exposed as legal

Normalized v4 representation:

- action: `action_id = 7`, `action_name = ACTION7`
- legality: false because `7` is absent from pre-step `available_actions`
- execution status in transition form: rejected

Authoritative fields:

- pre-step `available_actions`
- requested action id and name

Absent fields that must stay absent:

- guessed fallback action
- inferred “closest legal action”

## Coordinate Action Example

Raw source shape summary:

- local offline `lo01` exposes `env.action_space = [ACTION1, ACTION2, ACTION3, ACTION4, ACTION6]`
- reset exposes `available_actions = [1, 2, 3, 4, 6]`
- one real step with `ACTION6` and payload `x = 1, y = 1` returns `FrameDataRaw`
- `levels_completed` increases from `0` to `1`

Normalized v4 representation:

- action: `action_id = 6`, `action_name = ACTION6`
- payload: `x = 1`, `y = 1`
- legality: true because `6` is present in pre-step `available_actions`
- post observation: `levels_completed = 1`

Authoritative fields:

- `ACTION6`
- payload `x`, `y`
- post-step `levels_completed`

Absent fields that must stay absent:

- guessed grid-cell interpretation beyond the raw coordinate payload

## Terminal Win Step

Raw source shape summary:

- local ARCEngine test class `TestGameWithWinScore`
- real `ActionInput(id=ACTION5)` step returns `FrameDataRaw`
- post-step `state = WIN`

Normalized v4 representation:

- terminal signal: success
- `is_terminal = true`
- `reset_required = true`

Authoritative fields:

- post-step raw `state = WIN`

Absent fields that must stay absent:

- reward-derived win inference

## Terminal Failure Step

Raw source shape summary:

- local ARCEngine test class `TestGame`
- real `ActionInput(id=ACTION7)` step returns `FrameDataRaw`
- post-step `state = GAME_OVER`

Normalized v4 representation:

- terminal signal: failure
- `is_terminal = true`
- `reset_required = true`

Authoritative fields:

- post-step raw `state = GAME_OVER`

Absent fields that must stay absent:

- heuristic failure labels

## Metadata Example

Raw source shape summary:

- observed `env.info` on local offline `ez01`
- `game_id = ez01-63be02fb`
- `title = Go Up`
- `description = null`
- `local_dir` present
- `date_downloaded` present
- `env.action_space` exposes `ACTION1` through `ACTION4`

Normalized v4 representation:

- `game_id = ez01-63be02fb`
- `title = Go Up`
- `description = null`
- `action_ids = (1, 2, 3, 4)`
- `action_names = (ACTION1, ACTION2, ACTION3, ACTION4)`
- `coordinate_action_id = null`

Authoritative fields:

- direct wrapper metadata fields
- direct action-space fields

Absent fields that must stay absent:

- static frame width and height
- level id

## Transition Record Example

Raw source shape summary:

- pre observation from local `ez01` reset
- executed legal action `ACTION1`
- post observation from the next real step

Normalized v4 representation:

- pre observation: raw-authoritative snapshot
- action: `ACTION1`
- post observation: raw-authoritative snapshot
- `action_legal = true`
- `execution_status = executed`
- terminal signal: non-terminal

Authoritative fields:

- pre observation
- executed action
- post observation
- terminal derivation from post-step `state`

Absent fields that must stay absent:

- planner belief state
- mechanic labels

## Step Result Example

Raw source shape summary:

- derived from the transition above

Normalized v4 representation:

- executed action: `ACTION1`
- `action_legal = true`
- `raw_state_before = NOT_FINISHED`
- `raw_state_after = NOT_FINISHED`
- `levels_completed_delta = 0`
- `win_levels_delta = 0`
- `reset_required = false`

Authoritative fields:

- raw states before and after
- executed action
- legality

Absent fields that must stay absent:

- POI hits
- candidate ranking
- planner rationale

