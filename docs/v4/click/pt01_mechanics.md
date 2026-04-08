Status: implemented with explicit transition-frame handling
Scope: `pt01` live rebuild and policy behavior in `src/v4/click/*`
Source of truth: `/home/zodrak/zod/src/v4/click/familyAdapters.py`, `/home/zodrak/zod/src/v4/click/stateBuilder.py`, `/home/zodrak/zod/src/v4/click/solverPolicy.py`
Last verified against: current repo state on 2026-03-30; targeted manual live regression run `python -m tests.v4.live_regression.runner --game pt01 --jobs 1 --max-steps 500`

# `pt01` Mechanics

## Stable Runtime Phases

`pt01` uses a family-local three-phase split:

- `pt01_active_board`
- `pt01_transition_frame`
- `pt01_new_level_board`

The phase detector uses only the current observation, previous observation, derived `levels_completed_delta`, and the authoritative surfaced level index.

## Transition Rule

- `pt01_active_board`: normal tile/arrow matching and deterministic click-plan rebuild are allowed.
- `pt01_transition_frame`: the normal stable-board matcher is not used, and stale cached click state from the old level is not reused.
- `pt01_new_level_board`: cached plan/state is rebound to the new authoritative level before normal click decisions resume.

## Cache Boundary

`pt01` cached click state is level-bound.

When the authoritative level index changes, the solver clears old-board derived cache state, including:

- cached click plan
- cached last typed state
- old level-bound action assumptions

The new board is rebuilt from the new authoritative level before the next normal click plan is emitted.

## Transition Handling

If a true transition frame is surfaced, the solver does not route it through the old stable-board matcher. It uses only deterministic family-local progression behavior from surfaced action availability, and if the transition frame does not resolve after a small fixed number of replans it fail-closes with a specific `pt01_transition_stall` tag.

## Stable-Board Planning

On stable `pt01` boards, the solver uses a deterministic family-local rotation plan derived from the typed state:

- each rotatable tile is matched against the target rotation for its type
- the required number of quarter-turn clicks is emitted deterministically
- stale old-level click plans are never carried into the new level

## Live Regression Diagnostics

The separate manual live regression runner surfaces these `pt01` diagnostics:

- current authoritative level index
- cached level index
- transition-frame detected flag
- cache-invalidated flag
