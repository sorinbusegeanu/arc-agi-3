### Established mechanics
- Level 0: Blue square is the player. Moving leaves colored trails. Up/left from top-left start are boundary no-ops. Staircase (alternating down/right) completed the level; progress indicator fills green as player advances.
- Level 1: Trail alternates yellow/green. Backtracking onto trail cells causes GAME_OVER. Snake-like path visiting all yellow cells without revisiting completed the level.
- Level 2: Yellow cells are targets; gray cells are obstacles. Progress fills green as targets collected. Route around gray obstacle completed the level.
- Level 3: Yellow targets around gray center obstacle. Route around obstacle completed the level.
- Level 4: Four yellow squares at grid corners; player starts on top-left one. Cyan bar near bottom-right depletes each move (move budget); small green square at its end tracks remaining budget. Up/left from start are boundary no-ops. Down from start works, leaving green trail. No obstacles observed on the grid. Trail does NOT cause GAME_OVER on Level 4 (player moved along trail-adjacent cells without death).

### Failed hypotheses
- Level 1: Staircase from start → GAME_OVER (backtracking onto trail). Snake path is correct.
- Level 4: Staircase route reached bottom-right corner without completing; budget exhausted before all corners visited.
- Level 4: Perimeter tour (right along top, down along right) reached two corners without completing; budget insufficient for remaining corners.
- Level 4: Straight-down route reached bottom-left corner without completing after visiting only two corners (start + bottom-left).
- Level 4: Down-left-column then right-bottom-row then up-right-column route reached top-right area and triggered GAME_OVER (~19 total moves). Cause unclear: budget exhaustion or trail interaction.
- Level 4: All-corners minimum path (21 moves) appears to equal or exceed the budget, making visit-all-corners completion impossible with a minimum path.

### Unresolved hypotheses
- Level 4: Completion condition is unknown. Visit-all-corners is contradicted by budget constraints. Possible alternatives: visiting a subset of corners, reaching a specific target, or a different mechanic entirely.
- Level 4: The green trail may or may not be hazardous on Level 4; no direct backtracking-onto-trail test has been performed.

### Next experiment
- Level 4 (player one cell right of bottom-left corner, trail on left column): Continue right along bottom edge to bottom-right corner, then stop and observe whether visiting 3 corners (top-left, bottom-left, bottom-right) triggers completion. If not, continue up along right edge toward top-right to test whether 4 corners ever completes. This distinguishes "3 corners suffice" from "all 4 required but budget insufficient" from "completion condition is entirely different."
