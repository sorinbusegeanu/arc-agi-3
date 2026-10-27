## Established mechanics (levels 0–4)
- The blue square is the movable agent; the yellow square is the stationary goal. Directional actions move the agent one cell in that direction.
- Moving the agent into the goal (overlap) collects it and completes the level. The intermediate frame during collection shows the goal disappearing before the next level's start screen appears.
- All four directions produce consistent one-cell displacement when the agent is not at a screen boundary. At the top edge, "up" produces no visible change (boundary constraint, not a general wall). Confirmed on levels 0 and 4.

## Verified strategy (level 0)
- Move the agent toward the goal until adjacent, then one more step in the same direction to overlap and collect.

## Verified strategy (level 1)
- The agent started above the goal. Consecutive downward moves brought it into overlap. No obstacles or detours were needed.

## Verified strategy (level 2)
- The agent started above and offset horizontally from the goal. Move horizontally to align with the goal, then move vertically toward it to overlap and collect. No obstacles encountered.

## Verified strategy (levels 3–4)
- The agent started directly above the goal (vertically aligned). Consecutive downward moves brought it into overlap. No horizontal alignment or obstacles were needed. The route was fully direct with no wasted actions.

## Ineffective approaches
- "up" from the top edge of the screen produced no pixel change; do not repeat this action from that state.
