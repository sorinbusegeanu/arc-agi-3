Status: implemented and verified
Scope: Precise current-state summary for v4
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-30; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`, plus targeted manual live regression for `pt01`, `wm01`, `fs01`, `fs02`, `fs03`, `ic01`, `ms01`, `rs01`, `sv01`, `pb02`, and `tb01`

# Current v4 Status

## Phase 1 Status

Status: implemented and verified

Evidence:

- `src/v4/agentContract` exists with types, validators, adapters, metadata extraction, and error types.
- `tests/v4/agentContract` exists with observation, action, terminal, transition, and real-environment adapter tests.

## Stage 2 Status

Status: implemented and verified

Evidence:

- `src/v4/runtime`, `src/v4/state`, `src/v4/memory`, and `src/v4/policy` are implemented.
- `tests/v4/closed_loop` covers parser, env session, ledger, stop conditions, policy, and local memory.
- `tests/v4/easy_games` covers live Stage 2 sessions, easy-game gates, rejection paths, and longer-horizon behavior.

## Phase 3 Movement Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `ul01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `fs01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests; current manual live regression now surfaces an explicit repeated-non-progress cycle trace |
| `tp01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `ic01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests; current manual live regression still ends in no-plan search failure |
| `va01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `pb01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `pb02` | implemented; gate-covered, live regression not yet verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests, including hidden-goal live replanning reconstruction; current live regression classifies the zero-step startup as `worker_timeout` with `pb02_timeout_before_first_action` and a staged pre-typed-state trace |
| `pb03` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `fs02` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests; current manual live regression alternates between a win and a certifying-prefix / repeat-cycle failure |
| `fs03` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests; current manual live regression still shows repeated non-progress cycles |
| later movement families outside `src/v4/movement` | documented only | no additional movement family implementation exists under `src/v4/movement` beyond `fs03` |

## Phase 4 Click/Perception Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `pt01` | implemented and verified | implemented in `src/v4/click`; covered by gate, transition, search, and policy tests; manual live regression now uses explicit transition-frame handling, level-bound cache invalidation, and stable new-level rebuild before normal click planning resumes |
| `sy01` | implemented and verified | implemented in `src/v4/click`; covered by gate, transition, search, and policy tests |
| `ff01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| `sq01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| `wm01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests; current manual live regression no longer uses a vague builder tag and now lands in a classified repeated non-progress cycle / step-budget path |
| `mm01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| later click/perception families outside `src/v4/click` | documented only | no additional click family implementation exists under `src/v4/click` |

## Phase 5 Memory-Hidden Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `ms01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/memory_hidden`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests; current manual live regression still reports a low-expansion no-plan frontier/search failure |

## Phase 6 Rule-Switch Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `rs01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/rule_switch`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests; current manual live regression stays in the search-failure bucket and no longer collapses into a successor-validation abort |

## Phase 7 Time/Reactive Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `sv01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/time_reactive`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests; current manual live regression now surfaces `bounded_horizon_infeasible` instead of a generic safe-action-set abort |

## Phase 8 Hybrid Construction Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `tb01` | implemented; gate-covered, live regression still mixed | implemented in `src/v4/hybrid_construction`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests; current live regression now classifies the zero-step startup as `worker_timeout` with `tb01_timeout_before_first_action` and a staged pre-typed-state trace |

## Implemented Source Packages

- `src/v4/agentContract`
- `src/v4/runtime`
- `src/v4/state`
- `src/v4/memory`
- `src/v4/policy`
- `src/v4/movement`
- `src/v4/click`
- `src/v4/memory_hidden`
- `src/v4/rule_switch`
- `src/v4/time_reactive`
- `src/v4/hybrid_construction`

## Implemented Tests

- `tests/v4/agentContract`
- `tests/v4/closed_loop`
- `tests/v4/easy_games`
- `tests/v4/movement`
- `tests/v4/click`
- `tests/v4/memory_hidden`
- `tests/v4/rule_switch`
- `tests/v4/time_reactive`
- `tests/v4/hybrid_construction`
- `tests/v4/live_regression`
- `tests/v4/test_isolation.py`
- `tests/v4/run_suite.py`

## Passed Gates

- Phase 1 contract surface is implemented and verified by `tests/v4/agentContract/*`.
- Stage 2 easy-game coverage is implemented and verified by `tests/v4/easy_games/*`.
- Phase 3 gate tests are present for `ul01`, `fs01`, `fs02`, `fs03`, `tp01`, `ic01`, `va01`, `pb01`, `pb02`, and `pb03`.
- Phase 4 gate tests are present for `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, and `mm01`.
- Phase 5 gate test is present for `ms01`.
- Phase 6 gate test is present for `rs01`.
- Phase 7 gate test is present for `sv01`.
- Phase 8 gate test is present for `tb01`.

## Remaining Gaps

- Older Phase 1 and Stage 2 handoff/checklist docs still need cleanup.
- Families beyond the currently implemented movement, click, and later-track slices remain outside the `src/v4` package set.

## Separate Live Regression Surface

- `tests/v4/live_regression` adds a manual implemented-family live regression runner plus a small smoke test.
- The full live regression run is separate from the default unit and gate suite.
- The runner prints one summary table and writes one CSV plus one JSON export.
- `pt01` live rows now also surface level-bound transition/cache diagnostics for authoritative level index, cached level index, transition-frame detection, and cache invalidation.

The Step 8 trace runner is a diagnostic harness that can run in two policy modes: `certified` and `family_exact`. `certified` exercises `CertifiedPlannerPolicyV4`, while `family_exact` uses the implemented family solver head for the game under test.

## Known Boundaries / Exclusions

- No `v3_1` runtime path is allowed in `src/v4`.
- No blackboard, POIs, hypotheses, mechanic graph, or durable memory is part of the v4 loop.
- No LLM, VLM, or RL logic is part of the Stage 2 action loop or the implemented exact solver tracks.
- This status document covers only what exists under `src/v4`, `tests/v4`, and the reviewed docs tree.
