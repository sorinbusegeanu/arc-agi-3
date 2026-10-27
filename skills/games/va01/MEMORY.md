## Established mechanics (Levels 1–2)
- Agent leaves a persistent trail in vacated cells and can move through it.
- In Levels 1–2, black and gray cells were impassable obstacles.
- Movement from an edge cell toward that edge produced no visible change; this is a state-specific no-op, not a general wall.
- The yellow goal cell turned green on completion. Entering the yellow goal alone did not complete Levels 1–2.
- Levels 1–2 completed when every open non-obstacle, non-goal cell was covered by trail.

## Verified strategy (Levels 1–2)
- Systematically sweep all open cells with the agent's trail, navigating around obstacles and edges, until every non-obstacle cell is covered. Completion occurred automatically upon full coverage.

## Level 3 (index 2) – observations and unresolved question
- On Level 3, the agent also leaves a persistent trail and can move through it; moves through already-trailed cells did not add new coverage, while moves into untrailed open cells extended the trail.
- On Level 3, gray barrier cells are impassable except for a narrow passable gap; black cells are passable open cells.
- On Level 3, the trail is visually distinguishable from the background, so coverage can be tracked visually.
- A gray barrier separates the level into upper and lower regions; the gap has been crossed in both directions.
- Edge and barrier-edge no-ops are state-specific, not general walls.
- Entering or overlapping the yellow goal without confirmed full coverage did not complete Level 3; this does not prove goal entry is irrelevant.
- Level 3 has not completed after upper-region sweeps and partial lower-region traversal; full lower-region coverage has not been confirmed.
- Unresolved question: Why has Level 3 not completed after upper-region sweeps and partial lower-region traversal? Competing explanations: (1) some open cells remain uncovered, especially in the lower region or near obstacles/edges; (2) the completion rule differs from Levels 1–2, such as a required final action, goal entry after coverage, or another trigger.
- Next experiment: On Level 3, perform a deliberate serpentine sweep of the lower region until all lower open cells are visibly trail-covered; if completion still does not occur, test whether the yellow goal must be entered last or whether a different completion trigger exists.
