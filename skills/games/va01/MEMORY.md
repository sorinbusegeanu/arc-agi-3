## Established mechanics (Levels 1–2)
- Agent leaves a persistent orange trail in every vacated cell; trail remains after the agent moves on.
- Agent can move through its own trail.
- Black cells are impassable obstacles; gray cells are impassable obstacles.
- Movement toward a screen edge from the edge cell produces zero pixel change (state-specific no-op, not a general wall).
- Yellow cell at bottom-right is a goal indicator; turns green on level completion.
- Level completion is triggered by covering every open (non-obstacle, non-goal) cell with trail. Entering the yellow alone does not complete the level; full coverage does.

## Verified strategy (Levels 1–2)
- Systematically sweep all open cells with the agent's trail, navigating around black and gray obstacles and screen edges, until every non-obstacle cell is covered. The level completes automatically upon full coverage.

## Level 3 (index 2) – observations and unresolved question
- A gray horizontal barrier spans the middle with a narrow gap (one column wide) near the center; the gap is passable vertically only at the gap column. Agent has confirmed crossing the gap in both directions.
- Multiple state-specific no-ops observed at barrier edges and screen edges. These are barrier-edge or edge blocks, not general walls.
- Agent has extensively swept both lower and upper regions across many actions without level completion. Upper region swept to top edge; lower region swept to bottom and left edges. Full coverage has not been confirmed.
- The yellow goal cell remains yellow throughout all sweeping. The agent has been adjacent to the yellow cell but has not entered it.
- Agent oscillated left-right-left-right while adjacent to the yellow cell without entering it; this oscillation produced no completion and is an ineffective pattern for that state.
- **Unresolved question:** Is coverage still incomplete (some open cells remain uncovered), or is entering the yellow goal cell required as a final step after full coverage?
- **Next experiment:** Navigate to the yellow goal cell and enter it (move right toward it, then down into it). If the level completes, goal entry is the final completion trigger after coverage. If it does not complete, coverage is still incomplete and remaining uncovered cells must be identified and swept.
