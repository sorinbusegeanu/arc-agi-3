Status: implemented and verified
Scope: movement doc: `pb02` mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/pb02/63be02fb/pb02.py`, `/home/zodrak/zod/tests/v4/movement/test_transition_model_pb02.py`, `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_pb02.py`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`

# `pb02` Mechanics

- Family: exact push-block movement family inside `src/v4/movement`
- Local env source: `other_repos/arc-interactive/environment_files/pb02/63be02fb/pb02.py`
- Level-0 static layout: player `(1, 1)`, crates `(2, 3)` and `(4, 2)`, goals `(5, 5)` and `(6, 4)`

## Implemented Mechanics

- Two crates and two real goals are represented explicitly.
- Success requires both crates to occupy the goal set.
- Movement into open floor is deterministic.
- A push is legal only when the adjacent cell contains exactly one crate and the cell beyond it is in-bounds and free of walls and crates.
- Multi-crate pushes are rejected.
- Search uses canonicalized crate ordering so equivalent crate permutations hash identically.

## Verified Coverage

- Typed-state build from env metadata plus observation.
- Exact two-crate transition tests.
- Exact planner search to a certifying solution.
- Real-env replay of the certifying plan to a level-0 win.
- Stage 2 live replanning through hidden-on-goal occupancy without widening the runtime surface.

## Allowed Reconstruction Rule

- `pb02` may rebuild a two-crate state from one currently visible crate only when a prior `pb02` carry state already exists.
- The reconstruction uses only the current parsed observation, the previous parsed observation when present, the prior per-episode carry state, and the static local env goal definition.
- The carry state is advanced by the previous legal movement action under the exact `pb02` transition rules.
- The reconstructed two-crate set becomes authoritative for search and policy only when the current visible crate evidence is a subset of that predicted carry-state set, every hidden predicted crate remains on a configured real goal, and the predicted set still has size two.
- If any of those checks fail, the builder raises a `pb02 reconstruction inconsistent` error and the path stays fail-closed.
