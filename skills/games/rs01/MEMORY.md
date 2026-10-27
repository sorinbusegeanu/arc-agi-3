## Established mechanics (Level 0)
- The large blue square is the player; the small blue square in the top-left corner is a HUD indicator, not the player.
- Moving the player into a colored square (red or green) causes that square to disappear (collection event). The player occupies the square's former cell.
- Moving up from the top row produced no visible change from that state. This is a boundary-specific no-op; it does not by itself establish a general wall mechanic.
- The yellow bar in the top HUD area grows as collectibles are gathered; it is above the player's reachable area and is not a navigable goal.

## Failed / ineffective approaches (Level 0)
- Repeated attempts to descend near the far-right edge ended in GAME_OVER. Avoid that route until its cause is isolated; this does not show that downward movement is generally unsafe.
- After the visible colored squares were collected, up, right, and left produced no visible change from that state. Down was not tested there, so whether the player was trapped remains unresolved.

## Unresolved hypotheses (Level 0)
- Whether collecting all four colored squares (2 red, 2 green) completes the level, or whether an additional condition (e.g., reaching a specific location, a timer, or a collection order) is required. Unconfirmed.
- The yellow bar's role (progress indicator, timer, or something else) is unconfirmed beyond its growth correlating with collections.

## Next experiment (Level 0)
- On a fresh attempt, test one safely reachable colored square and compare the before/after screen and level state. Avoid the previously hazardous far-right descent. If collection is confirmed but the level does not change, test the yellow bar as a separate possible completion condition.
