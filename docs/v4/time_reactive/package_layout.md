Status: implemented and verified
Scope: `src/v4/time_reactive` package layout
Source of truth: `/home/zodrak/zod/src/v4/time_reactive/*`
Last verified against: unknown

# Package Layout

- `typedState.py`: exact `sv01` typed-state dataclasses
- `familyAdapters.py`: parsed-state to typed-state extraction
- `stateBuilder.py`: package entry point
- `transitionModel.py`: hunger, warmth, timer, food, warm-zone, and wait consequences
- `search.py`: bounded exact survival search
- `solverPolicy.py`: Stage 2-compatible policy output
