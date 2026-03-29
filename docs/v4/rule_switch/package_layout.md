Status: implemented and verified
Scope: `src/v4/rule_switch` package layout
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/*`
Last verified against: unknown

# Package Layout

- `typedState.py`: exact `rs01` typed-state dataclasses
- `familyAdapters.py`: parsed-state to typed-state extraction
- `stateBuilder.py`: package entry point
- `transitionModel.py`: deterministic safe-color collection and wrong-color loss
- `heuristics.py`: local distance helper
- `search.py`: bounded exact search over movement actions
- `solverPolicy.py`: Stage 2-compatible policy surface
