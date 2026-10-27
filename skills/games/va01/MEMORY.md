## Established mechanics (Levels 1–2)
- Agent leaves a persistent trail in every vacated cell; the trail remains after the agent moves on.
- Agent can move through its own trail.
- Black cells and gray cells are impassable obstacles.
- Movement from an edge cell toward that edge produces no visible change; this is a state-specific no-op, not a general wall.
- The yellow goal cell turns green on completion. Entering the yellow goal alone does not complete Levels 1–2.
- Levels 1–2 complete when every open non-obstacle, non-goal cell is covered by trail.

## Verified strategy (Levels 1–2)
- Systematically sweep all open cells with the agent's trail, navigating around obstacles and edges, until every non-obstacle cell is covered. Completion occurs automatically upon full coverage.

## Level 3 (index 2) – observations and unresolved question
- A gray barrier separates the level into upper and lower regions with a narrow passable gap; the gap has been crossed in both directions.
- Black obstacles occur in both regions.
- Edge and barrier-edge no-ops are state-specific, not general walls.
- Entering or overlapping the yellow goal without confirmed full coverage did not complete the level.
- Background and trail are visually indistinguishable, so coverage cannot be confirmed visually.
- Unresolved question: Level 3 has not completed after partial sweeps of separated regions. Competing explanations: (1) some open cells in one or both regions remain uncovered, especially near obstacles or edges; (2) the completion rule differs from Levels 1–2, such as a required final action, goal entry after coverage, or another trigger.
- Next experiment: From the upper region, perform a deliberate serpentine sweep of every open cell above the barrier, then cross the gap and repeat in the lower region, using path shape to track visited cells. If completion still does not occur after both regions are path-tracked as fully covered, test whether the yellow goal must be entered last or whether a different completion trigger exists.
