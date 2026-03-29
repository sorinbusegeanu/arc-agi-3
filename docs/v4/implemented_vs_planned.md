Status: implemented and verified
Scope: Implemented vs planned split for v4 tracks
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 311 tests

# Implemented vs Planned

## Contracts

Implemented now:

- `src/v4/agentContract` types, validators, adapters, metadata extraction, and error surface.
- Contract tests under `tests/v4/agentContract`.

Only documented or planned:

- None beyond the implemented Phase 1 contract surface in the current `src/v4` tree.

Intentionally deferred:

- Planner logic, POIs, mechanic inference, reward shaping, symbolic promotions, and other advisory layers outside the authoritative contract path.

## Closed Loop

Implemented now:

- `src/v4/runtime`, `src/v4/state`, `src/v4/memory`, and `src/v4/policy`.
- Stage 2 live-loop validation in `tests/v4/closed_loop` and `tests/v4/easy_games`.

Only documented or planned:

- Historical next-gate and checklist docs that have not yet been rewritten to match the current implemented state.

Intentionally deferred:

- Blackboard, branch/merge logic, durable memory, and legacy `v3_1` runtime machinery.

## Movement Track

Implemented now:

- Exact movement typed state, adapters, transition model, search, heuristics, and solver policy in `src/v4/movement`.
- Implemented and verified families: `ul01`, `fs01`, `tp01`, `ic01`, `va01`, `pb01`.

Only documented or planned:

- Movement families outside those six remain outside the implemented `src/v4/movement` tree.

Intentionally deferred:

- Broader movement families and stronger family-specific heuristics not present in the current source tree.

## Click/Perception Track

Implemented now:

- Exact click typed state, adapters, transition model, search, heuristics, and solver policy in `src/v4/click`.
- Implemented and verified families: `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, `mm01`.

Only documented or planned:

- Additional click/perception families beyond the current six remain outside `src/v4/click`.

Intentionally deferred:

- Additional click/perception families beyond the six implemented under `src/v4/click`.

## Later Tracks

Implemented now:

- `src/v4/memory_hidden` and `tests/v4/memory_hidden` implement and verify `ms01`.
- `src/v4/rule_switch` and `tests/v4/rule_switch` implement and verify `rs01`.
- `src/v4/time_reactive` and `tests/v4/time_reactive` implement and verify `sv01`.
- `src/v4/hybrid_construction` and `tests/v4/hybrid_construction` implement and verify `tb01`.

Only documented or planned:

- Families beyond `ms01`, `rs01`, `sv01`, and `tb01` appear only in planning or reference docs.

Intentionally deferred:

- Any later-track family that does not yet have an implemented `src/v4` package and matching tests.
