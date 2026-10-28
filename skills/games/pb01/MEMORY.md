### Established mechanics (Levels 0–4)
- Blue square is the player; directional actions move it one cell per step.
- Purple square is pushable: player moving into it shifts purple one cell in the same direction.
- Yellow square is a target: pushing purple onto yellow removes purple and completes the level.
- Small purple square in the bottom-right corner is a persistent UI element, not a game object.
- Gray squares are obstacles; player and purple cannot occupy them. Obstacles may have gaps allowing passage.

### Level 4 strategy (verified)
- Gray obstacles form a vertical barrier in the center column with gaps at certain rows; yellow target is to the right of the barrier, purple starts to the left.
- The barrier gap is passable for the purple when pushed rightward through it.
- Approach: navigate around the barrier to get above the purple, push purple rightward through the gap, then reposition above purple and push downward onto the yellow target.
- A right action when the player was directly adjacent to a gray obstacle produced no pixel change (state-specific no-op, not a general block on rightward pushes).
