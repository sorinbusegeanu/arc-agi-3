Status: implemented and verified
Scope: Implemented vs planned split for v4 tracks
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-30; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`, plus targeted manual live regression for `pt01`, `wm01`, `fs01`, `fs02`, `fs03`, `ic01`, `ms01`, `rs01`, `sv01`, `pb02`, and `tb01`

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
- Implemented and verified families: `ul01`, `fs01`, `fs02`, `fs03`, `tp01`, `ic01`, `va01`, `pb01`, `pb02`, `pb03`.
- Current live-regression status is mixed: `fs02` now alternates between a win and a certifying-prefix / repeat-cycle failure in the manual live runner, while `fs01`, `fs03`, `ic01`, and `pb02` still have unresolved live-only issues.

Only documented or planned:

- Movement families outside `ul01`, `fs01`, `fs02`, `fs03`, `tp01`, `ic01`, `va01`, `pb01`, `pb02`, and `pb03` remain outside the implemented `src/v4/movement` tree.

Intentionally deferred:

- Broader movement families and stronger family-specific heuristics not present in the current source tree.

## Click/Perception Track

Implemented now:

- Exact click typed state, adapters, transition model, search, heuristics, and solver policy in `src/v4/click`.
- Implemented and gate-covered families: `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, `mm01`.
- `pt01` includes explicit family-local handling for transition frames, level-bound cache invalidation, deterministic transition progression, and stable new-level rebuild before normal click planning resumes.
- `wm01` no longer uses the vague live builder tag; its current manual live row is a classified repeated-cycle / step-budget failure instead of an unclassified builder abort.

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
- Current manual live regression is still mixed for these later tracks: `ms01`, `rs01`, `sv01`, and `tb01` remain implemented and gate-covered, but not yet live-verified end-to-end. `tb01` is now explicitly classified as a zero-step `worker_timeout` startup row instead of a blank timeout row.

Only documented or planned:

- Families beyond `ms01`, `rs01`, `sv01`, and `tb01` appear only in planning or reference docs.

Intentionally deferred:

- Any later-track family that does not yet have an implemented `src/v4` package and matching tests.
