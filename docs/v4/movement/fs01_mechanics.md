Status: implemented and verified
Scope: movement doc: fs01 mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `fs01` Mechanics

This document is grounded only in the local implementation at `/home/zodrak/zod/other_repos/arc-interactive/environment_files/fs01/63be02fb/fs01.py`.

## Exact Semantics

- switch type actually present: yellow floor switch sprite with tag `switch`
- switch collision: non-collidable
- directly observed activated-switch signal: activated switches are recolored from `11` to `10`
- switch activation rule in the real implementation: first entry adds that switch position to `_activated`
- switch persistence: once activated, it stays activated for the rest of the level
- door behavior: gray collidable door sprite with tag `door`
- directly represented door state in v4 typed state: closed-door positions plus one explicit open/closed bit
- door update rule mirrored from the local implementation: `_open_door_if_ready()` removes the door list when `len(_activated) >= len(_switch_positions)`
- target rule: reaching the target advances the level only when the local door list is empty

The v4 movement layer must not replace those observed rules with a generic switch-door abstraction or a hidden per-level lookup table.

## Mechanic Table

| mechanic field | observed source | state representation | update rule | notes |
| --- | --- | --- | --- | --- |
| switch positions | level sprite positions with tag `switch` | tuple of grid cells | static per level | directly derivable from the local level definition |
| activated switch set | game field `_activated` plus switch recolor `11 -> 10` | exact set or bitmask over switch positions | add current player position when it lands on an inactive switch | latch, not toggle or pressure-hold |
| door positions while closed | level sprites with tag `door` | tuple of closed-door cells | removed when all switches are activated | closed door is collidable |
| door open state | `_door` list empty or non-empty | one open/closed bit | open when `len(_activated) >= len(_switch_positions)` | no partial opening state |
| switch-to-door linkage | `_open_door_if_ready()` | explicit open/closed door bit plus closed-door positions | local implementation compares `_activated` against `_switch_positions` | no authored per-switch linkage is exposed in metadata |
| target completion | target sprite test in `step()` | success when player overlaps target after door removal | call `next_level()` only if `len(_door) == 0` | target itself is non-collidable |
