## Established mechanics (level 0)
- Player moves in four directions and leaves a gray trail. The trail is lethal in level 0 (re-entering causes GAME_OVER).
- Green region in the bottom-right is the completion target; reaching it triggers a level transition.
- Verified strategy: navigate from top-left start to bottom-right green region by moving down along the left edge, then right along the bottom edge, avoiding the trail.

## Established mechanics (level 1)
- Trail is NOT lethal in level 1: re-entering the trail does not cause GAME_OVER.
- The far-right green block (color 11) at the top-right corner is the completion target; reaching it triggers a level transition.
- The red cell that appears at the trail endpoint during rightward movement is a non-hazardous visual marker; passing through it is safe.
- GAME_OVER is triggered by a right move into a specific upper-middle region; right moves from lower rows proceed normally.
- Verified strategy: from top-left start, move down along the left edge, right along the bottom edge (passing through the red cell safely), then up along the right edge to reach the green block at top-right.

## Failed hypotheses (level 1)
- "Trail is lethal in level 1 like level 0": contradicted by safe re-entry into the trail.
- "Rightward movement universally causes GAME_OVER": contradicted by safe right moves from lower positions.
- "Red cell at trail endpoint marks a lethal boundary": contradicted by safe passage through it.

## Established mechanics (level 2)
- Player starts at top-left; green target (color 14) at bottom-right corner. Up and left at start are no-ops. Down and right each move one cell and paint a gray trail cell.
- Reaching the green target completes the level (confirmed by level transition).
- Verified strategy: move down along the left edge, then right along the bottom edge to reach the green target. No obstacles encountered on this monotone path.
- Trail lethality in level 2 remains unconfirmed: the player never re-entered the trail during the successful route.
