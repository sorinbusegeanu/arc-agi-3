# Board Perception Overview

The board perception layer is a shared pluggable perception module.

It is not a live agent, not an execution authority, and not a replacement for the existing deterministic runtime logic. In v1 it produces advisory-only outputs that other v4.5 components may consume.

## Scope

- board geometry handling
- board construction from observations
- object extraction
- optional learned supplement later
- advisory-only output in v1

## Placement

- `src/v4_5/perception/board_geometry/*`
- `src/v4_5/perception/board_builder/*`
- `src/v4_5/perception/board_fusion/*`

## Boundaries

- the module does not call the environment directly
- the module does not mutate authoritative runtime state
- the module does not own control flow
- the Orchestrator remains the authority

