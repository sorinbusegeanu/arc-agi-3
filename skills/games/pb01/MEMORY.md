## Established mechanics (Level 0)
- Blue square is the movable agent; purple and yellow squares are stationary. Agent moves one cell per action in four directions.
- Double-step displacement (dx=16 or dy=16) confirms pass-through of a stationary object's cell.
- Purple pass-through confirmed in all four directions. No completion in any direction.
- Yellow causes agent to vanish when entered from any direction; agent reappears on the reverse action. No completion.
- Small purple square at bottom-right corner persists across all screens; reaching it does not trigger completion.

## Failed hypotheses (Level 0)
- Overlapping purple or yellow in any of the four tested directions does not trigger completion.
- Reaching corner, full-grid traversal, ordered visits (purple→yellow→corner, yellow→purple→corner) do not trigger completion.
- Adjacency to yellow without entering its cell does not trigger completion.
- Pushing purple in any direction (up, down, left, right) does not trigger completion.
- Extended free-roaming without object interaction does not trigger completion.
- All four directions of both objects tested; no single overlap or push completes the level.
- Pushing purple rightward (confirmed via double-step) without it reaching the corner does not trigger completion.

## Unresolved hypotheses (Level 0)
- Completion condition unknown. All tested interactions, orderings, traversals, and pushes fail.
- Hidden mechanic (specific final agent position, pushing purple to a target location, specific multi-object arrangement, or a state trigger) may govern completion.
- The small purple corner square's role beyond a persistent indicator is unclear; approach from untested directions may matter.

## Next experiment (Level 0)
- Push purple toward the bottom-right corner (rightward then downward) and check whether purple overlapping the corner triggers completion. Purpose: distinguish whether completion requires a specific object-to-object spatial relationship (purple at corner) versus a specific agent final position or hidden trigger.
