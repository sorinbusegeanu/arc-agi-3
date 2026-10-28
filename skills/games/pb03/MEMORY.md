## Established mechanics
- Levels 0-2: Blue moves one grid step per action. Purple, yellow, large orange, small orange, dark red are stationary. Entering large orange or yellow makes blue vanish; opposite direction restores it. Entering purple merges blue into a pair (double displacement along merge axis; perpendicular move breaks merge). Small orange is passable. Dark red is not a goal. Pressing toward screen edge triggers GAME_OVER. Purple can be pushed but may become trapped at edges.
- Level 0: Completion when merged pair pushed down into orange+yellow cluster.
- Level 1: Completion when merged pair pushed right into orange+yellow cluster.
- Level 2: Merge works horizontally and vertically. Gray block blocks pair movement. Completion when vertically merged pair pushed up into orange+yellow cluster.
- Level 3: Vertical merge + down = double displacement. Horizontal merge + right = double displacement. Horizontal merge at gap row traverses barrier as unit. Vertical merge + right at barrier = no-op. Vertical merge + right in open space breaks merge. Unmerged blue passes through gap.

## Failed hypotheses
- Levels 0-2: No single-object interaction alone completes a level. Oscillating in/out of yellow or large orange does not complete. Pushing pair into gray block is a no-op. Oscillating along merge axis without reaching cluster produces no progress.
- Level 3: Right while vertically merged in open space breaks merge. Entering cluster with unmerged blue vanishes blue but does not complete. Oscillating merged/unmerged without crossing plan = no progress. Up while vertically merged at top of barrier = no-op. Horizontal merge at barrier not aligned with gap row + right = no-op (confirmed repeatedly).

## Successful strategies
- Level 0: Merge blue+purple, navigate right, push pair down into cluster.
- Level 1: Merge blue+purple, navigate upper-right, push pair right into cluster.
- Level 2: Merge vertically, align below cluster, push pair up into cluster. Avoid gray block and axis oscillation.

## Level 3 observations
- Gray vertical barrier with ~1-cell gap; blue alone can cross.
- Orange+yellow cluster is right of barrier, below the gap.
- Horizontal merged pair aligned with gap row crosses barrier successfully.
- After crossing, blue separated and entered cluster, vanishing; level did not complete.
- Pushing pair down one step toward cluster did not complete.
- Pushing pair right (double displacement) near cluster did not complete.

## Next experiment (Level 3)
- Align horizontally merged pair with gap row (move down), cross right, then push pair down into diagonal cluster so both cells land on orange and yellow simultaneously. Purpose: test whether pair must enter cluster as unit (matching levels 0-2) or a different condition applies. If fails, try vertical merge after crossing and push down.
