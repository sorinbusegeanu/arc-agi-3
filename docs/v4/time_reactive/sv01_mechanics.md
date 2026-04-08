Status: implemented; gate-covered, live regression still mixed
Scope: `sv01` mechanics
Source of truth: `/home/zodrak/zod/src/v4/time_reactive/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/sv01/63be02fb/sv01.py`
Last verified against: current repo state on 2026-03-30; targeted manual live regression for `sv01`

# `sv01` Mechanics

Current implementation treats `sv01` as a movement + wait survival family with explicit hunger, warmth, and timer decay.

Implemented now:

- hunger and warmth are read from the rendered bars
- food and warm-zone cells are extracted from the grid
- wait is legal when action `5` is exposed
- the transition model decrements resources exactly according to the current package rules
- policy-side bounded-safe fallback ranks currently feasible non-failure survival actions and now distinguishes `no_legal_safe_move`, `all_safe_moves_pruned_by_repeat_guard`, and `bounded_horizon_infeasible` from typed-state build failure

Current boundary:

- the manual live regression row is still a real bounded-safe search failure, now tagged specifically as `bounded_horizon_infeasible`
