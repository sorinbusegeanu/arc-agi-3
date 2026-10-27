## Established mechanics (Levels 0–4)
- The yellow square is the stationary goal; the blue square is the movable agent. Overlapping the blue agent with the yellow goal triggers level completion (extra color transitions plus an intermediate frame where the yellow disappears).
- The agent moves one cell per action in the commanded direction. Ordinary movement produces two color transitions; completion produces additional transitions (observed as 4–6 transitions across Levels 2–4).
- In Level 1, rightward actions from a state where the agent sat at the right edge produced no visible change; ineffective from that specific state.

## Verified strategy (Levels 0–4)
- Move the blue agent directly toward the yellow goal and overlap its cell. In Levels 2, 3, and 4 the goal was to the left on the same row, so repeated leftward movement completed the level with no obstacles encountered. A direct path avoids wandering and avoids edge-of-screen no-op moves.
