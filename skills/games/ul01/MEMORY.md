## Established mechanics (Levels 0–1)
- Blue is the movable agent; moves one cell per action.
- Yellow is pushable: blue moving into yellow's cell displaces yellow one cell in the same direction (confirmed Levels 0 and 1).
- Gray is a stationary goal in Levels 0 and 1; blue entering it completes the level.

## Level 0 strategy (verified)
- Push yellow away from the goal area, then navigate blue to the gray square and enter it.

## Level 1 strategy (verified)
- Navigate blue to the cell directly below the gray square, then move up to enter it. Entering gray from the left (moving right) is blocked in the observed state (no pixel change, state-specific).

## Level 2 findings (confirmed across multiple runs)
- Yellow push confirmed in all four directions from starting layout.
- The gray element in the bottom-right area is a wall corner/bracket, NOT a goal.
- Yellow pushed right, down, left, OR up from starting layout always traps in the bottom-right corner; no further push is possible once there.
- Yellow overlapping the bracket cell changes the bracket's appearance but does not trigger completion.
- Blue-alone traversal of all accessible cells produced no completion.
- Blue entering the bracket from any direction produced no completion.
- Pushing yellow up into the bracket area produced a more complex pixel change (3 color transitions, no clear displacement) than simple overlap, suggesting a different interaction type with the bracket.

## Failed approaches (Level 2)
- All four push directions from starting layout: yellow traps in corner, no completion.
- Blue entering bracket from any direction: no completion.
- Yellow overlapping bracket: no completion.
- Blue-alone full traversal: no completion.

## Unresolved hypotheses (Level 2)
- Completion condition unknown. Possible: (a) unmarked goal cell elsewhere; (b) specific blue-yellow spatial arrangement relative to gray/bracket; (c) bracket has a state-dependent interaction triggered by specific push direction or sequence.

## Next experiment (Level 2)
- Yellow is trapped in the bottom-right corner after an up-push. The bracket's appearance changed during this push (complex pixel transition). Test whether blue entering the bracket cell now (with the changed bracket state) produces a different result than previously-tested bracket entries. If no change, the level likely requires an approach not yet discovered (e.g., an unmarked goal cell or a multi-step interaction sequence).
