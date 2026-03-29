Status: implemented and verified
Scope: movement doc: ic01 mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ic01` Mechanics

The real local `ic01` implementation is a frictionless straight-line slide game.

- There is no separate ice sprite on the board. The slide surface is the set of cells that are not walls and not red hazards.
- A movement action starts sliding immediately.
- Sliding continues in the chosen cardinal direction until the next cell would leave the grid, hit a gray wall, or hit a red hazard.
- The avatar stops before entering that blocking cell.
- The yellow goal is not solid. It does not stop a slide by itself.
- Level completion is checked after the slide resolves, from the final landing cell.

| mechanic field | observed source | state representation | update rule | notes |
| --- | --- | --- | --- | --- |
| slide surface | local board layout after excluding gray walls and red hazards | traversable cell set | unchanged during a level | no separate ice tile sprite is authored |
| wall cells | gray wall sprites in level layout and frame | explicit blocked positions | never entered; stop before them | collidable |
| red hazard cells | red hazard sprites in level layout and frame | explicit hazard positions | never entered; stop before them | collidable and blocking |
| goal cell | yellow target sprite | explicit goal position | completion only if final landing cell equals goal | goal does not block motion |
| slide direction | player action `ACTION1`-`ACTION4` | one cardinal delta | fixed for the full action | no mid-slide retargeting |
| stop rule | local game step logic | deterministic landing cell | stop before OOB, wall, or red hazard | exact `ic01` rule surface |
