# v5.0 movement and level solving design

## Scope in code

In the current source, movement and level solving are implemented as a **reset-and-replay trajectory system**, not as a full world-model planner.

Main files:

- `v5_0/solve/service.py`
- `v5_0/solve/loop_runner.py`
- `v5_0/solve/policy_builder.py`
- `v5_0/solve/target_selector.py`
- `v5_0/contact/service.py`
- `v5_0/route/trajectory_enumerator.py`
- `v5_0/replay/player.py`
- `v5_0/runtime/run_avatar_bootstrap.py`
- `v5_0/runtime/campaign_state.py`
- `v5_0/memory/trace_store.py`

## Core idea

The system does not first infer a full traversable/blocking map and then solve the level symbolically.

Instead it does this:

1. bootstrap the level
2. identify avatar
3. detect and rank POIs
4. optionally run contact experiments on POIs
5. pick a target POI
6. generate action routes from avatar to target from geometry
7. execute actions on a live session
8. classify the outcome
9. keep target, retarget, or finish
10. if a level is solved, save the trace and replay-verify it
11. for the next level, reset and replay solved prefixes

## How movement is represented

Movement is represented as **short action sequences** over primitive actions:

- `LEFT`
- `RIGHT`
- `UP`
- `DOWN`

The movement code is not based on tile graph search.
It is based on:

- avatar bbox / center
- target bbox / center
- approximate action step scale from bbox sizes
- generated routes whose net displacement matches target displacement

## Route generation

Route generation is in `v5_0/route/trajectory_enumerator.py`.

### Action-space delta

The code estimates step scale from avatar and target bbox size, then converts pixel-space delta into action-space delta:

- `compute_action_space_delta(...)`
- `estimate_action_step_scale(...)`

This gives approximate `(dx, dy)` in action steps.

### Route enumeration

`enumerate_routes_between_points(...)` generates candidate routes between start and target.

Properties:

- shortest interleavings first
- deterministic ordering
- bounded route count
- bounded max length
- route metadata is stored per candidate

Each `RouteCandidate` contains:

- `route_id`
- `actions`
- `length`
- `net_dx`, `net_dy`
- `first_action`
- `turn_count`
- `axis_order`
- `waypoints`
- `score_components`

### Route validation

`validate_route_actions_for_action_delta(...)` filters routes.

It rejects routes with things like:

- excessive length
- early cancel patterns like `LEFT, RIGHT`
- impossible displacement
- anti-target first move

So movement is generated as a small plausible set of candidate trajectories, not as arbitrary exploration.

## Contact-driven movement testing

`v5_0/contact/service.py` is the first place where generated movement routes are actually tested against POIs.

For each POI and episode, it does roughly this:

1. build policies toward the POI
2. enumerate multiple route candidates
3. try routes in order
4. run them on a fresh or replay-prepared session
5. classify the result
6. stop early if a useful effect is found

Useful effects include things like:

- level transition
- terminal success/failure
- reward change
- object removed
- new object appeared
- `door_opens`

This is the first full implementation of “movement toward object + evaluate what happened”.

## Solve policy at step level

`v5_0/solve/loop_runner.py` implements online movement during solving.

### `run_solve_episode(...)`

This is a step-by-step loop.

At each frontier step it:

1. reads current observation
2. tracks avatar bbox in current frame
3. tracks target bbox in current frame
4. builds a solve policy for the current target
5. takes the first action from that policy
6. executes the action
7. compares pre/post frame
8. classifies outcome
9. decides whether to continue or retarget

The loop tracks:

- blocked actions
- invalid actions
- contact with target
- screen changes
- HUD-only changes
- reward changes
- level completion count before/after
- terminal state

### Retarget logic

It retargets when progress is poor, for example:

- repeated `no_effect`
- blocked streak
- target disappeared without useful world change
- target becomes invalid

Target selection is delegated to `v5_0/solve/target_selector.py`.

## Target selection

The system does not blindly chase the highest-confidence POI forever.

`target_selector.py` combines:

- POI ranking confidence
- HUD target selection if available
- contact experiment results
- mechanic memory labels
- no-effect history
- blocked history
- retirement of decoys/hazards

Main behavior:

- choose initial target from HUD/mechanics/contact/POI ranking
- keep target if recent evidence is still useful
- switch target after repeated failure/no effect
- avoid border-locked, decoy, or low-value targets

So the movement loop is **target-driven** and **outcome-adaptive**.

## Closed-loop solve vs adaptive solve

`v5_0/solve/service.py` contains two main solving modes.

### Closed-loop solve

`run_closed_loop_solve_multi_reset(...)`

Behavior:

- requires stable avatar and at least one POI
- builds mechanic report
- selects initial target
- runs `run_solve_episode(...)` for each episode
- aggregates diagnostics

This is the simpler target-following loop.

### Adaptive solve

`run_adaptive_solve_multi_reset(...)`

Behavior:

- also requires stable avatar and POIs
- runs `run_adaptive_solve_episode(...)`
- tracks target switches, retargets, useful changes, no-progress counts
- is used as the main per-level solving path in campaign flow

This mode is closer to the actual implemented level-solving logic.

## How a level is solved in campaign mode

Campaign flow is implemented in `v5_0/runtime/run_avatar_bootstrap.py`.

### Per-level structure

For a frontier level, the system does this:

1. prepare the frontier state
2. bootstrap/analyze the current level from that frontier
3. discover avatar, POIs, HUD, contact evidence
4. run adaptive or closed-loop solve
5. convert solve result into a level solution trace
6. replay-verify the trace
7. store verified best trace in the trace DB
8. move to next level

### Frontier preparation

The important part is that new attempts are usually **not started from the raw game beginning only**.

The system uses prefix replay:

- solved earlier levels are stored as traces
- before solving a later level, it resets the game
- replays verified solved prefixes to reach the frontier level
- then performs frontier analysis and solve from there

That logic is in:

- `replay_prefix_traces_to_frontier(...)`
- `replay_prefix_to_frontier(...)`
- `replay_trace_at_frontier(...)`
- `get_verified_prefix_traces(...)`
- `get_current_run_prefix_traces(...)`

This matches the design of “reset before each trajectory and replay solved prefix to reach the correct level/state”.

## Solved trace extraction and verification

After solving, the code extracts a replayable level trace.

Important functions in `solve/service.py`:

- `build_level_solution_from_adaptive_report(...)`
- `extract_replayable_level_trace(...)`
- `verify_level_trace_replay(...)`
- `finalize_solved_level_trace(...)`

### What is saved

A saved solved trace includes:

- `game_id`
- `level_id`
- `action_trace`
- `step_count`
- `solved`
- `replay_verified`
- `action_sources`
- `trace_id`

### Verification rule

A level trace is not trusted only because the solve loop thought it solved the level.

The code resets, replays the prefix plus the level trace, and checks whether the intended level is actually solved.

Only verified traces are used as reliable prefixes later.

## Campaign state

`v5_0/runtime/campaign_state.py` keeps per-level campaign state.

For each level it stores fields like:

- `status`
- `solved`
- `best_step_count`
- `attempt_count`
- `solution_trace_path`

The frontier level is the first level not yet solved in the current run or from DB, depending on mode.

## Database-backed reuse

The movement/solving system is persistent.

`trace_store.py` is used to:

- save solved traces
- keep best/shortest verified trace
- load solved levels for a game
- rebuild trace indexes
- mark replay verification results

That means later runs can skip already solved levels by replaying verified traces from storage.

## Important implementation characteristic

The system is **trajectory- and evidence-based**, not model-based.

It does not currently solve movement games by building a full explicit board semantics model with:

- traversable map
- blocking map
- object interaction graph
- exact deterministic transition simulator

Instead, it relies on:

- geometric route guesses
- repeated reset/replay
- direct environment execution
- frame comparison
- contact outcome classification
- retargeting when evidence is weak

## What “level solved” means in current code

In `build_level_solution_from_adaptive_report(...)`, a level is treated as complete if either:

- the adaptive solve report says `solved=True`, or
- a level transition happened

So level transition is accepted as proof that the level was completed.

## Short summary

Movement and level solving in current `v5_0` work like this:

- generate short movement routes from avatar-to-target geometry
- test them on the real environment
- classify the effect of contact / movement
- adapt target selection based on outcomes
- once a level transitions or is solved, save the action trace
- replay-verify the trace
- use verified traces as prefixes for later levels

This is a **reset + replay + route enumeration + outcome classification** solver.
It is not a general symbolic path planner yet.
