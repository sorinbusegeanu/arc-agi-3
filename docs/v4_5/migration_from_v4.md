# Migration From v4

v4 remains the source of truth for solver logic.

v4.5 wraps and orchestrates existing v4 logic. It does not replace it.

## Stable Points

- family packages stay in `src/v4/*`
- deterministic runtime semantics stay in `src/v4/*`
- planner-family logic is reused through thin plugin adapters

## New in v4.5

- control-plane clarity
- explicit contracts
- one execution authority
- offline optimization outputs
- optional advisory-only interface

## Non-Goals

- moving family packages into v4.5
- collapsing v4 family logic into a single rewritten planner
- allowing advisory paths to bypass deterministic verification
