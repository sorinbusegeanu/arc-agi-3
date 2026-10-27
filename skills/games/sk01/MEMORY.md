### Established mechanics (Level 4)
- The player moves freely through the gap in the gray wall separating left and right regions.
- Walking into a colored square pushes it one step in the same direction; colored squares can be pushed through the gap.
- Pushing a colored square into another colored square causes the target square to vanish (observed: blue pushed left into purple on the yellow shape; purple disappeared).
- A purple pushed onto the edge of the yellow L-shape remains visible without being absorbed (confirmed across multiple runs).
- Walking through the yellow shape as the player produces no interaction.
- The green square at the bottom-right corner is present from the level's initial state.

### Failed hypotheses (Level 4)
- Absorbing one purple into the yellow shape: no completion (confirmed in prior run and this run).
- Visiting the green square at bottom-right: no transition (confirmed in prior run).
- Extensive free movement in the left region without pushing colored squares toward the yellow shape: no progress (ineffective approach).
- Pushing the blue square onto the yellow shape and then moving it off: no completion.
- Causing a purple to vanish by pushing the blue into it: no completion.
- Up or left from certain positions: state-specific no-op; does not prove a general obstacle.

### Unresolved hypotheses (Level 4)
- Whether leaving a colored square on the yellow shape (without moving it off) triggers completion.
- Whether the remaining purple must be pushed through the gap into the yellow shape and left there.
- Whether a different interaction (e.g., pushing a colored square onto the green square, or a specific spatial arrangement) is required.
- Whether the yellow shape needs to be covered or filled by a colored square in a specific way.

### Next experiment (Level 4)
- Question: Does pushing the remaining purple through the gap and leaving it on the yellow shape trigger completion, or is a different condition needed?
- Test: Navigate to the remaining purple in the upper-left, push it right through the gap and down onto the yellow shape, then leave it there without moving it off. If no completion, try pushing the blue square onto the yellow shape and leaving it there. If still no completion, test pushing a colored square onto the green square.
