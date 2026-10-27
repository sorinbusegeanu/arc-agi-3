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
- A black obstacle exists in the upper-right area; black obstacles also exist in the lower-left area.
- State-specific no-ops at barrier edges and screen edges are barrier-edge or edge blocks, not general walls.
- The yellow goal cell remains yellow throughout all sweeping. Entering/overlapping the yellow goal and then moving away did not complete the level.
- **Failed hypothesis:** Entering the yellow goal cell as the final action did not complete the level.
- **Visual ambiguity:** Background color equals trail color, so uncovered cells are visually indistinguishable from covered cells.
- **Unresolved question:** After extensive lower-region sweeping and multiple gap crossings, the level has not completed. The upper region (above the barrier) may not be fully covered, especially cells near the upper-right black obstacle and along the top edge.
- **Next experiment:** From the gap position, move up into the upper region and systematically sweep every open cell above the barrier, paying special attention to cells adjacent to the upper-right black obstacle and along the top edge. If the level still does not complete after confirmed upper-region coverage, the completion condition for this level differs from levels 1–2.
