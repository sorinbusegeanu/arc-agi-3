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
- State-specific no-ops at barrier edges and screen edges are barrier-edge or edge blocks, not general walls. Do not repeat the exact no-op actions from those states.
- The yellow goal cell remains yellow throughout all sweeping. The agent has entered/overlapped the yellow goal cell at the bottom-right and then moved away; the level has NOT completed. Repeated left-right oscillation adjacent to or overlapping the yellow goal without completion is an ineffective approach for that state.
- Agent has swept both the upper and lower regions extensively. The orange trail appears to cover all visible open cells in both regions.
- **Unresolved question:** Does the level complete upon full coverage of all open cells (goal irrelevant), or must the yellow goal cell be entered as the final action after full coverage? The agent entered the goal and then moved away without completion, suggesting either (a) some open cells remain uncovered, or (b) goal-entry must be the last step after coverage is complete.
- **Next experiment:** After sweeping any remaining uncovered open cells (check corners and cells adjacent to obstacles), enter the yellow goal cell as the final action and do not move afterward. If the level completes, explanation (b) is confirmed. If it does not, explanation (a) is more likely and coverage must be re-examined.
