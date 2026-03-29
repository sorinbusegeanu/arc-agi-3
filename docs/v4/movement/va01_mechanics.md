Status: implemented and verified
Scope: movement doc: va01 mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `va01` Mechanics

The real local `va01` implementation is a visit-all-walkable-cells game.

- Coverage applies to every open cell that is not occupied by a wall.
- Covered state is represented in the live environment by green trail sprites on visited cells plus the current player position.
- The start cell counts immediately. The environment marks it visited in `on_set_level`.
- Revisits are allowed.
- Revisits do not remove coverage and do not create a second state change beyond moving the avatar.
- Completion is exactly `visited == open_cells`.
- There is no separate exit or goal tile beyond full coverage.
- Walls are the only blocking cells in the real local `va01` implementation.

| mechanic field | observed source | state representation | update rule | notes |
| --- | --- | --- | --- | --- |
| coverage-eligible cells | local level layout after excluding wall sprites | exact open-cell set | static for a level | full-coverage target surface |
| covered state | trail sprites plus current player cell | visited-cell set | add current cell on first entry; keep on revisit | start cell is inserted immediately |
| revisit rule | local `step()` logic | same visited-cell set | no-op for coverage if already visited | revisits remain legal |
| completion condition | local `step()` logic | equality of visited and open-cell sets | level clears when sets match | no separate exit tile |
| blockers | wall sprites | blocked positions | movement into wall is rejected | no hazards in `va01` |
