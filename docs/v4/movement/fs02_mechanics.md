Status: implemented; gate-covered, live regression still mixed
Scope: movement doc: `fs02` mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/fs02/63be02fb/fs02.py`, `/home/zodrak/zod/tests/v4/movement/test_transition_model_fs02.py`, `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_fs02.py`
Last verified against: current repo state on 2026-03-30; targeted movement tests for `fs02` and targeted manual live regression for `fs02`

# `fs02` Mechanics

- Family: exact floor-switch movement family inside `src/v4/movement`
- Local env source: `other_repos/arc-interactive/environment_files/fs02/63be02fb/fs02.py`
- Local repo rule: any one switch removes the door permanently

## Implemented Mechanics

- Switch positions, door positions, target cells, and switch logic mode are explicit.
- Entering any switch opens the door.
- Once opened, the door remains removed in this local env variant.
- Closed-door cells block motion exactly.
- Search reasons over switch and door state explicitly.
- The current movement solver keeps a short family-local certifying plan prefix so live replans do not discard an already found door-opening route.

## Verified Coverage

- Env-backed OR-switch extraction.
- Transition tests for opening and blocked-door behavior.
- Exact planner success on level 0.
- Stage 2 live controller solves level 0 before the next-level controller boundary.
- The manual live regression runner currently alternates between a win and a certifying-prefix / repeat-cycle failure on the seeded `fs02` slice.
