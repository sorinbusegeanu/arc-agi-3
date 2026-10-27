## Established mechanics (Level 1)
- Blue agent moves one cell per cardinal action; confirmed across all four directions.
- Yellow squares are paired portals: agent vanishes on entry, reappears at the paired location after the next directional input. Confirmed across multiple runs and entry directions.
- Purple squares are passable and non-consumable; agent passed through multiple purple squares and they persisted unchanged.
- Moving through a passable object (e.g., purple square) in one action produces a two-cell displacement, confirming the object is neither blocking nor consumable.
- Moving right at the right edge produced zero pixel change; state-specific no-op, not a general wall.
- Moving down from a specific bottom-edge position caused GAME_OVER; state-specific, not a general hazard.

## Failed / ineffective approaches (Level 1)
- Extensive free-roaming (100+ actions across multiple runs, all directions, multiple portal traversals, edge sweeps, passing through all visible purple squares) produced no completion.
- Portal traversal followed by moving away from the exit portal did not complete the level.
- Entering the bottom portal from below (moving up into it) did not complete the level.
- Oscillation at the bottom edge (left-right alternation) produced no progress.
- Coverage sweep, portal traversal in both directions, portal-then-position at bottom-right purple, and exact corner landing all failed to complete the level.
- Entering the partially visible bottom-right purple square produced only a small pixel overlap and no completion; that square is not a goal.

## Unresolved hypotheses (Level 1)
- Completion condition remains unknown. Purple squares are confirmed non-consumable and non-blocking; they may be decorative or require a specific spatial relationship with portals or a specific final agent position.
- Completion may require a specific portal sequence (e.g., consecutive portal entries), a specific arrival position after portal traversal, a specific entry direction into a portal, or a spatial relationship between portal and purple-square positions.
- Whether a double portal traversal (enter one portal, reappear at the paired portal, then immediately enter that portal again) triggers completion has not been tested.

## Next experiment (Level 1)
- After the agent reappears from a portal entry, immediately move toward the other portal to test consecutive portal traversal (enter portal A → reappear at portal B → enter portal B). This distinguishes a double-portal-sequence hypothesis from a single-portal-plus-position hypothesis.
