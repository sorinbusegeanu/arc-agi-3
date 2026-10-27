## Established mechanics
- On level 0, the blue square is the controllable agent; it moves one cell per action in four directions. Purple, yellow, and green squares are static.
- The agent can pass through colored squares; during overlap colors blend, but the square reappears after the agent moves past it. A single action can produce a 2-cell displacement when passing through a colored square (confirmed for purple in both horizontal and vertical directions, and for yellow on level 0).

## Failed hypotheses
- Overlapping purple, yellow, or green individually (in various orders) does not complete level 0.
- Perimeter traversal (both directions), column/row sweeps, exhaustive agent-alone traversal, staircase trajectories, zigzag/diagonal approaches, and continuous paths visiting all three objects in various orders do not complete level 0.
- Reaching the extreme bottom-right corner adjacent to or partially overlapping the green square does not complete level 0.
- Returning to the starting position after visiting all three colored objects does not complete level 0.
- Repeated oscillation near colored squares produces no progress; ineffective.
- Moving right at the right edge or down at the bottom edge produces no visible change from that state; state-specific no-op, not a proven wall.

## Unresolved hypotheses
- Whether the green square's center is fully reachable or partially off-screen, making exact overlap impossible. Unconfirmed.
- Whether completion requires a specific movement sequence, timing, or non-spatial condition beyond cell overlap. Unconfirmed.
- Whether the 2-cell pass-through displacement causes the agent to skip specific cells that are never visited, and whether one of those skipped cells is the actual trigger. Unconfirmed.

## Next experiment
- On level 0, approach a colored square from one direction to trigger the 2-cell pass-through displacement, note which cell is skipped (the cell immediately beyond the colored square in the direction of travel), then approach that skipped cell from a perpendicular direction to land on it without triggering pass-through. If completion triggers, a skipped cell is the trigger. If not, eliminate that cell and test other cells skipped by pass-through in different directions.
