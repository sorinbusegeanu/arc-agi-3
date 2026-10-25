## Mechanics
- Blue square is the player; moves one grid cell per directional input.
- Gray square is the goal; reaching it completes the level (verified on level 0).
- Yellow square is displaced when overlapped; not the goal.

## Strategy
- On level 0 matching initial layout, navigate blue to gray; use verified route when starting state matches.
- On level 1: Approaching gray from the left (right input) and from below (up input) are both blocked—confirmed by zero-change transitions. Prior note that down-from-above also fails remains unchallenged this cycle.
- Player oscillates between left-of-gray and below-gray positions without progress. Untried approaches: entering from the right side of gray, or interacting with the yellow square (possibly as a key or pushable obstacle).
- Yellow square present on level 1 but not yet interacted with; its behavior is unconfirmed.
