## Established mechanics

- **Level 0 (initial state):** Rightward movement merged the agent with intermediate objects along its path and reached the goal, triggering completion.
- **Level 1:** The gray vertical bar blocks rightward movement at the agent's starting height but can be traversed at its bottom end.
- **Level 1:** Pink squares are passable/consumable; moving into one causes it to disappear.
- **Level 1:** The large yellow square in the upper-right is the goal; reaching it completes the level.
- **Level 2:** The gray vertical bar has a gap in the middle; the agent passes through it to reach the right side.
- **Level 2:** The yellow goal in the upper-right completes the level on direct overlap. Pink squares are NOT required for completion; the agent reached the goal without consuming any.
- **Level 3:** Rightward movement from the left side reached the yellow goal in the upper-right, completing the level. Pink squares were passable/consumable along the path. No obstacles blocked the route.
- **Level 4:** The gray vertical bar has a gap in the middle; the agent passed through it to reach the right side. Pink squares are passable/consumable. The yellow goal in the bottom-right completes the level on direct overlap.

## Failed / ineffective approaches

- **Level 1, starting height:** Consecutive right actions produced zero pixel change due to the gray bar at that height. State-specific blockage; does not generalize to other heights.
- **Level 1, initial state:** A single left action produced no visible change. State-specific no-op.
- **Level 3, starting position:** Up and left actions produced zero pixel change. State-specific no-ops at that position; do not generalize to other positions in the level.

## Unresolved hypotheses

- **Levels 1, 3, and 4:** A small gray/white square appeared in the level, turned yellow when the agent was adjacent, then disappeared. Its role (switch, collectible, side-effect) is unconfirmed; it was not required for completion in any of these levels.
