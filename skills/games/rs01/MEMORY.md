### Established mechanics (Level 0)
- Agent (blue square) moves 8 px per cardinal step; confirmed across multiple runs with displacement score 1.0.
- Moving into screen edges triggers GAME_OVER; confirmed for left and bottom edges. Treat all edges as lethal.
- Exact overlap followed by a subsequent move away consumes the field object; adjacency alone does not. Re-confirmed: red consumed after moving down through it, green consumed after moving right through it.

### Failed hypotheses (Level 0)
- Consuming 3 of 4 field squares (2 red, 1 green) did not complete the level (prior run).
- Consuming 2 red squares did not complete the level (earlier run).
- Top-left indicators (blue, red, green) did NOT change when field objects were consumed in the current run, contradicting the prior hypothesis that they track consumption. Their role remains unknown.

### Unresolved hypotheses (Level 0)
- Whether consuming all 4 field squares triggers completion remains unconfirmed.
- The yellow bar fixed in the top-right corner has not been overlapped. Whether it is a goal or decoration is unknown.
- The top-left indicators' function is unknown; they may be decorative or track a different state.

### Next experiment (Level 0)
- Consume the remaining 2 field squares (1 green, 1 red) in a continuous path from the current adjacent state. If the level completes after all 4 are consumed, the completion condition is "consume all field squares." If it does not complete, navigate to the yellow bar in the top-right corner and test direct overlap with it. Avoid moving into screen edges to prevent GAME_OVER.
