## Established mechanics
- Levels 0–4: The blue square is the movable agent (responds to directional inputs); the yellow square is the stationary goal. Completing the level requires the agent to overlap the goal.

## Verified strategy
- Levels 0–4 (agent left of goal, same row): Move the agent directly rightward toward the goal until overlap. Each level used only rightward moves with no loops or wasted actions.

## Ineffective approaches
- Levels 2–4 (agent at or near left boundary, goal to the right): A leftward action produced zero pixel change from that state. Do not repeat leftward moves when the agent is already at or near the left boundary and the goal is to the right.
