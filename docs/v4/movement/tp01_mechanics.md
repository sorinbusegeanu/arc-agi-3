Status: implemented and verified
Scope: movement doc: tp01 mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `tp01` Mechanics

`tp01` is a fixed symmetric-teleporter movement game. The local implementation exposes teleporter pair identity through level data and renders teleporter cells directly in the frame as magenta cells.

## Exact Observed Behavior

- teleporter cells are rendered as color `7`
- pair identity is stored in local level data as `portal_pairs`
- either endpoint of a pair warps to the other endpoint
- warp happens when the avatar enters a teleporter cell
- warp is mandatory
- the move ends on the paired endpoint; there is no extra continuation after warp
- walls block the tentative move before teleport logic runs
- the goal check runs after any warp resolution

## Table

| mechanic field | observed source | state representation | update rule | notes |
| --- | --- | --- | --- | --- |
| teleporter cell | direct frame content | grid cell with color `7` | fixed across the level | HUD also uses `7`, so pair identity must come from config rather than raw color count alone |
| teleporter pair identity | level `data["portal_pairs"]` | ordered endpoint pairs | fixed across the level | symmetric map is built from both directions of each pair |
| warp trigger | local `step()` implementation | enter teleporter endpoint | after legal movement lands on endpoint, avatar position is replaced with paired endpoint | not a stand-still trigger |
| warp directionality | local `step()` implementation plus `portal_pairs` map | bidirectional mapping | both `a -> b` and `b -> a` are valid | `tp01` is not one-way |
| post-warp motion | local `step()` implementation | none | action ends at warped destination | no chained movement after warp |
| wall interaction | direct frame content and local `step()` implementation | blocked cell set | wall blocks before warp logic | warp never bypasses wall entry checks |
| goal interaction | direct frame content and local `step()` implementation | yellow target cell | success check happens from the post-warp avatar position | landing on target through warp clears the level |
