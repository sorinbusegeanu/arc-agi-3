## Level 0 (first level)

- **Player and goal:** The blue square is the player (moves with directional actions); the yellow square is a static goal. Both are on the same horizontal row, separated by a gap.
- **Completion mechanic:** Moving the player rightward to close the gap (adjacent to the goal) does not complete the level. A second rightward step after contact causes the goal to vanish and the level to complete. Verified route: two consecutive right moves from the starting state.
- **Failed hypotheses:** A single right move from the initial state only steps the player; it does not complete the level. Simply reaching adjacency (overlap) is insufficient—completion requires one additional step past the goal in the same direction.
