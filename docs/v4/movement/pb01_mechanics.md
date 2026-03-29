Status: implemented and verified
Scope: movement doc: pb01 mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `pb01` Mechanics

The real local `pb01` implementation is a one-block push puzzle.

## Exact Observed Semantics

- the player is one collidable `player` sprite
- the movable object is one collidable `block` sprite
- the destination is one non-collidable `target` sprite
- `wall` sprites are collidable blockers
- only primitive movement actions `1` to `4` are exposed
- pulling does not exist
- a push is attempted only when the adjacent cell in the chosen direction contains the block
- the push succeeds only when the cell beyond the block is in bounds and contains neither a wall nor a block
- a blocked push leaves the positions unchanged
- a successful push moves the block one cell, then moves the player into the block's previous cell
- when the block reaches the target, the block recolors and the environment advances to the next level
- the local level config also exposes a positive `step_limit`; reaching that limit without solving causes failure

## Table

| mechanic field | observed source | state representation | update rule | notes |
| --- | --- | --- | --- | --- |
| pushable block | local `pb01.py` block sprite and live frame | exact block coordinate | moves one cell on a legal push | exactly one block in the local levels |
| target | local `pb01.py` target sprite and level config | exact target coordinate | static | non-collidable |
| wall blockers | local `pb01.py` wall sprites and level config | exact blocker coordinate set | static | block avatar movement and pushes |
| legal push | local `step()` logic | adjacent block plus free cell beyond | block and avatar both advance one cell | no pulling or diagonal push |
| blocked push | local `step()` logic | unchanged avatar and block coordinates | no movement on OOB or occupied destination | completion is not inferred from a blocked push |
| completion | local `_block_on_target()` and `next_level()` | block position equals target coordinate | level advances immediately after a solving push | target remains non-collidable |
| failure limit | local `step_limit` level data | positive integer | `lose()` when reached without solving | exposed by level config |
