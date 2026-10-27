### Established mechanics (Level 4)
- A vertical gray wall with a gap separates purples (left) from yellows (right). The player can walk through the gap.
- A purple aligned with the gap was pushed through it from the left.
- Pushing one purple into a yellow removes it without completing the level. Whether every purple must be removed to finish this level remains unconfirmed.
- Pressing into the wall (not at the gap) produces no pixel change; this is a state-specific no-op, not proof the gap is impassable.

### Tested approach (Level 4)
- Aligning a purple with the gap and pushing horizontally can move it through the wall. This is a demonstrated mechanic, not a verified full-level solution.

### Next experiment (Level 4)
- Question: Does removing the remaining purple-yellow pairings complete the level?
- Test: Use the demonstrated gap push to remove another purple, then check the environment level state. Treat the completion rule as unconfirmed until a level transition occurs.
