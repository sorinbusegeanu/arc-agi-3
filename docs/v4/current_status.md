Status: implemented and verified
Scope: Precise current-state summary for v4
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 311 tests

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
| `fs01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `tp01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `ic01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `va01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| `pb01` | implemented and verified | implemented in `src/v4/movement`; covered by gate, transition, search, and policy tests |
| later movement families outside `src/v4/movement` | documented only | no additional movement family implementation exists under `src/v4/movement` |

## Phase 4 Click/Perception Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `pt01` | implemented and verified | implemented in `src/v4/click`; covered by gate, transition, search, and policy tests |
| `sy01` | implemented and verified | implemented in `src/v4/click`; covered by gate, transition, search, and policy tests |
| `ff01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| `sq01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| `wm01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| `mm01` | implemented and verified | implemented in `src/v4/click`; covered by dedicated gate, transition, family-adapter, family-smoke, and regression tests |
| later click/perception families outside `src/v4/click` | documented only | no additional click family implementation exists under `src/v4/click` |

## Phase 5 Memory-Hidden Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `ms01` | implemented and verified | implemented in `src/v4/memory_hidden`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests |

## Phase 6 Rule-Switch Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `rs01` | implemented and verified | implemented in `src/v4/rule_switch`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests |

## Phase 7 Time/Reactive Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `sv01` | implemented and verified | implemented in `src/v4/time_reactive`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests |

## Phase 8 Hybrid Construction Status By Family

| family or group | status | evidence |
| --- | --- | --- |
| `tb01` | implemented and verified | implemented in `src/v4/hybrid_construction`; covered by typed-state, family-adapter, transition-model, search, family-smoke, and gate tests |

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
- `tests/v4/test_isolation.py`
- `tests/v4/run_suite.py`

## Passed Gates

- Phase 1 contract surface is implemented and verified by `tests/v4/agentContract/*`.
- Stage 2 easy-game coverage is implemented and verified by `tests/v4/easy_games/*`.
- Phase 3 gate tests are present for `ul01`, `fs01`, `tp01`, `ic01`, `va01`, and `pb01`.
- Phase 4 gate tests are present for `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, and `mm01`.
- Phase 5 gate test is present for `ms01`.
- Phase 6 gate test is present for `rs01`.
- Phase 7 gate test is present for `sv01`.
- Phase 8 gate test is present for `tb01`.

## Remaining Gaps

- Older Phase 1 and Stage 2 handoff/checklist docs still need cleanup.
- Families beyond `ms01`, `rs01`, `sv01`, and `tb01` remain outside the currently implemented later-track package set.

## Known Boundaries / Exclusions

- No `v3_1` runtime path is allowed in `src/v4`.
- No blackboard, POIs, hypotheses, mechanic graph, or durable memory is part of the v4 loop.
- No LLM, VLM, or RL logic is part of the Stage 2 action loop or the implemented exact solver tracks.
- This status document covers only what exists under `src/v4`, `tests/v4`, and the reviewed docs tree.
