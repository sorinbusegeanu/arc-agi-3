# Board Perception Builder

The deterministic board builder consumes the last 3 observations only.

The last-3-observation window exists to support:

- movement detection
- object persistence checks
- future learned-model compatibility

## Optional v1 Object Classes

These classes are optional and never mandatory:

- avatar/player
- POIs
- HUD
- progress bar
- lives
- traversable background
- blocking background / maze
- hazard
- clickable items
- moving enemies / NPCs

Any missing object class must be emitted as absent or empty, not as a failure.

## Background Representation

The contract exposes traversable and blocking background as pixel masks so downstream movement logic can read:

- plan movement only there
- do not plan movement there

This keeps inclusion and exclusion simple without requiring logical cell inference in v1.

