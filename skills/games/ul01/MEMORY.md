## Established mechanics (Levels 0–2)
- Blue is the movable agent; moves one cell per action.
- Yellow is pushable: blue moving into yellow's cell displaces yellow one cell in the same direction (confirmed Levels 0, 1, 2).
- Gray bracket in Level 2 is a wall corner, not a standalone goal.

## Level 0 strategy (verified)
- Push yellow away from the goal area, then navigate blue to the gray square and enter it.

## Level 1 strategy (verified)
- Navigate blue to the cell directly below the gray square, then move up to enter it. Entering gray from the left (moving right) is blocked in the observed state (no pixel change, state-specific).

## Level 2 strategy (verified)
- Push yellow into the bottom-right corner (where the gray bracket is), then navigate blue to enter that corner cell. Blue entering the bracket area alone (without yellow co-located) does not complete the level; the combination of yellow trapped in the corner and blue entering that same cell triggers completion.
