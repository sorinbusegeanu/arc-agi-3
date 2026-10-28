### Established mechanics
- Level 0: Blue square is the player. Moving leaves colored trails. Up/left from top-left start are boundary no-ops. Staircase (alternating down/right) completed the level; progress indicator fills green as player advances.
- Level 1: Trail alternates yellow/green. Backtracking onto trail cells causes GAME_OVER. Snake-like path visiting all yellow cells without revisiting completed the level.
- Level 2: Yellow cells are targets; gray cells are obstacles. Progress fills green as targets collected. Route around gray obstacle completed the level.
- Level 3: Yellow targets around gray center obstacle. Route around obstacle completed the level.
- Level 4: Four yellow squares at grid corners; player starts on top-left one. Cyan bar near bottom-right depletes each move (move budget). Up/left from start are boundary no-ops. Down from start works, leaving green trail. No obstacles observed on the grid.

### Failed hypotheses
- Level 1: Staircase from start → GAME_OVER (backtracking onto trail). Snake path is correct.
- Level 4: Staircase route reached bottom-right corner without completing; budget exhausted before all corners visited.
- Level 4: Perimeter tour (right along top, down along right) reached two corners without completing; budget insufficient for remaining corners.
- Level 4: Straight-down route reached bottom-left corner without completing after visiting only two corners (start + bottom-left).
- Level 4: Down-left-column then right-bottom-row then up-right-column route reached top-right area and triggered GAME_OVER (~19 total moves). Cause unclear: budget exhaustion or trail interaction.

### Unresolved hypotheses
- Level 4: Completion likely requires visiting all four yellow corners. Minimum path is 18 moves (three edges of 6). Whether the budget permits this is unconfirmed; the GAME_OVER at ~19 moves in one attempt suggests budget may be tight or trail is hazardous.
- Level 4: The cyan bar is likely a move budget; the small green square at its end may track progress but its exact role is unclear.
- Level 4: The green trail may be a hazard (backtracking causes GAME_OVER) as in Level 1, but this has not been directly confirmed on Level 4.

### Next experiment
- Level 4 (player one cell right of bottom-left corner, trail on left column, ~8 moves used): Continue right along bottom edge to bottom-right corner, then up along right edge to top-right corner. Tests whether visiting all four corners completes the level and whether the budget (~26 estimated from bar depletion rate) suffices for an 18-move minimum path. If GAME_OVER occurs before reaching top-right, distinguishes budget exhaustion from trail hazard.
