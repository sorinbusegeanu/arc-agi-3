## Established mechanics
- Levels 0-2: Blue moves one grid step per action. Purple, yellow, large orange, small orange, dark red are stationary. Entering large orange or yellow makes blue vanish; opposite direction restores it. Entering purple merges blue into a pair (double displacement along merge axis; perpendicular move breaks merge). Small orange is passable. Dark red is not a goal. Pressing toward screen edge triggers GAME_OVER. Purple can be pushed but may become trapped at edges.
- Level 0: Completion when merged pair pushed down into orange+yellow cluster.
- Level 1: Completion when merged pair pushed right into orange+yellow cluster.
- Level 2: Merge works horizontally and vertically. Gray block blocks pair movement. Completion when vertically merged pair pushed up into orange+yellow cluster.
- Level 3: Vertical merge + down = double displacement. Horizontal merge + right = double displacement. Horizontal merge at gap row CAN traverse the barrier (pair crosses as unit). Vertical merge + right at barrier = no-op (gap too narrow for 2-cell-tall pair). Vertical merge + right in open space = merge breaks. Unmerged blue passes through the gap.

## Failed hypotheses
- Levels 0-2: No single-object interaction alone completes a level. Oscillating in/out of yellow or large orange does not complete. Pushing pair into gray block is a no-op. Oscillating along merge axis without reaching cluster produces no progress.
- Level 3: Right while vertically merged in open space breaks the merge. Entering orange+yellow cluster with unmerged blue makes blue vanish but does not complete. Oscillating between merged/unmerged states without a crossing plan produces no progress. Up while vertically merged at top of barrier = no-op. Horizontal merge at barrier + right = no-op when not aligned with gap row.

## Successful strategies
- Level 0: Merge blue+purple, navigate right, push pair down into cluster.
- Level 1: Merge blue+purple, navigate upper-right, push pair right into cluster.
- Level 2: Merge vertically, align below cluster, push pair up into cluster. Avoid gray block and axis oscillation.

## Level 3 observations
- Gray vertical barrier with ~1-cell gap; blue alone can cross.
- Orange+yellow cluster is right of barrier, below the gap.
- Horizontal merged pair aligned with gap row successfully crosses the barrier.
- After crossing, blue separated and entered cluster, vanishing; level did not complete.

## Next experiment (Level 3)
- Press left to test whether the established restore mechanic (opposite direction recovers vanished blue) applies in level 3. If blue is restored, re-merge horizontally, cross the gap, then attempt to push the merged pair into the diagonal cluster from a position where both pair cells enter cluster cells simultaneously rather than blue entering alone. Purpose: distinguish whether level 3 requires the pair to enter the cluster as a unit (matching levels 0-2) or whether a different completion condition applies.
