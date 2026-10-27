### Established mechanics (Level 4)
- The blue agent consumes the yellow object by moving into it; the yellow disappears and the bottom-right indicator changes from yellow to green.
- After consuming the yellow, the blue agent briefly disappears and reappears on the next movement action.
- The blue agent can pass through the gap in the gray column and continue rightward past the column.
- Moving right into the green object (after consuming the yellow) triggers level completion.

### Established mechanics (Level 5)
- Consuming a yellow by moving into it: the yellow becomes a stationary light-blue square at its original cell; the bottom-right indicator widens after each consumption.
- After consuming ALL yellow objects, the bottom-right indicator changes from yellow to green.
- The agent can pass through the gray barrier gap in both directions (upward and downward confirmed).
- After all yellows are consumed, moving down into the green object triggers level completion.
- Light-blue split artifacts are stationary and do not move with the agent.

### Failed / ineffective approaches (Level 5)
- Approaching the green from the left (rightward) or from above (downward) did not trigger completion when not all yellows had been consumed; the precondition of consuming all yellows must be met first.
- Several state-specific no-ops: moving up from certain gap-row positions and moving right from a position directly below the gap produced zero visible change; do not repeat those exact state/action combinations.
