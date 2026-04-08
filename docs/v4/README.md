Status: implemented and verified
Scope: Top-level v4 overview
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`

# v4 Overview

## Purpose Of v4

`v4` is the current environment-facing stack in this repo. It keeps the authoritative contract strict, keeps the Stage 2 runtime single-session and deterministic, and layers exact family-explicit solver tracks on top of that runtime.

## Design Principles

- Direct environment output is the only authoritative per-step truth.
- Static environment metadata is used only when the repo already exposes it.
- The Stage 2 loop remains single-session, append-only, and failure-localized.
- Solver tracks are package-explicit, deterministic, bounded, and fail-closed.
- No `v3_1` runtime, blackboard, POIs, hypotheses, mechanic graph, durable memory, LLM, VLM, or RL logic is allowed in the action loop.

## Implemented Phases And Current Status

| phase | scope | current status | evidence |
| --- | --- | --- | --- |
| Phase 1 | `src/v4/agentContract` | implemented and verified | `tests/v4/agentContract/*` exists and is part of the passing v4 suite |
| Stage 2 | `src/v4/runtime`, `src/v4/state`, `src/v4/memory`, `src/v4/policy` | implemented and verified | `tests/v4/closed_loop/*` and `tests/v4/easy_games/*` exist and are part of the passing v4 suite |
| Phase 3 | `src/v4/movement` | implemented and verified for `ul01`, `fs01`, `fs02`, `fs03`, `tp01`, `ic01`, `va01`, `pb01`, and `pb03`; `pb02` is partially implemented | `tests/v4/movement/*` includes dedicated gate and regression coverage for the current ten-family movement slice |
| Phase 4 | `src/v4/click` | implemented and verified for `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, `mm01` | `tests/v4/click/*` exists and is part of the passing v4 suite |
| Phase 5 | `src/v4/memory_hidden` | implemented and verified for `ms01` | `tests/v4/memory_hidden/*` exists and its dedicated package suite passes |
| Phase 6 | `src/v4/rule_switch` | implemented and verified for `rs01` | `tests/v4/rule_switch/*` exists and its dedicated package suite passes |
| Phase 7 | `src/v4/time_reactive` | implemented and verified for `sv01` | `tests/v4/time_reactive/*` exists and its dedicated package suite passes |
| Phase 8 | `src/v4/hybrid_construction` | implemented and verified for `tb01` | `tests/v4/hybrid_construction/*` exists and its dedicated package suite passes |

## Package Map

| source package | role |
| --- | --- |
| `src/v4/agentContract` | authoritative observation, action, terminal, transition, step-result, metadata, adapters, and validators |
| `src/v4/runtime` | environment session, loop controller, ledger, and stop conditions |
| `src/v4/state` | parsed-state build, authoritative-state extraction, state hashing, and changed-region summaries |
| `src/v4/memory` | session-local bounded memory state and updates |
| `src/v4/policy` | shared policy interface and action helpers |
| `src/v4/movement` | exact movement typed-state, adapters, transition model, search, heuristics, and solver policy |
| `src/v4/click` | exact click/perception typed-state, adapters, transition model, search, heuristics, and solver policy |
| `src/v4/memory_hidden` | exact hidden-information movement typed-state, adapters, transition model, search, heuristics, and solver policy |
| `src/v4/rule_switch` | exact rule-switch movement typed-state, adapters, transition model, search, heuristics, and solver policy |
| `src/v4/time_reactive` | exact bounded time/reactive typed-state, adapters, transition model, search, heuristics, and solver policy |
| `src/v4/hybrid_construction` | exact hybrid movement-plus-build typed-state, adapters, transition model, search, heuristics, and solver policy |

## Test Map

| test package | coverage |
| --- | --- |
| `tests/v4/agentContract` | Phase 1 contract models, extraction, validation, and real-env adapter checks |
| `tests/v4/closed_loop` | Stage 2 parser, env session, policy, ledger, stop conditions, and local memory |
| `tests/v4/easy_games` | Stage 2 live-loop validation on `ez01`-`ez04`, `ul01`, and `tt01`, plus rejection and longer-horizon checks |
| `tests/v4/movement` | Phase 3 typed state, adapters, transition models, search, policy, gates, regression, and consolidation |
| `tests/v4/click` | Phase 4 typed state, adapters, transition models, search, policy, gates, regression, and consolidation |
| `tests/v4/memory_hidden` | Phase 5 typed state, adapters, transition model, search, policy smoke, and gate for `ms01` |
| `tests/v4/rule_switch` | Phase 6 typed state, adapters, transition model, search, policy smoke, and gate for `rs01` |
| `tests/v4/time_reactive` | Phase 7 typed state, adapters, transition model, search, policy smoke, and gate for `sv01` |
| `tests/v4/hybrid_construction` | Phase 8 typed state, adapters, transition model, search, policy smoke, and gate for `tb01` |
| `tests/v4/test_isolation.py` | isolation from legacy `v3_1` references |

## What Is Already Proven

- Phase 1 contract extraction and validation are implemented and exercised against real local environments.
- Stage 2 runs real local sessions and records valid observations, transitions, step results, memory updates, and ledger entries.
- Phase 3 exact movement solving is implemented and verified for `ul01`, `fs01`, `fs02`, `fs03`, `tp01`, `ic01`, `va01`, `pb01`, and `pb03`.
- `pb02` has env-backed typed-state extraction, exact transition/search coverage, and a real-env certifying plan replay; its current Stage 2 live replanning path still fails closed on hidden goal occupancy.
- Phase 4 exact click/perception solving is implemented and verified for `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, and `mm01`.
- Phase 5 exact memory-hidden solving is implemented and verified for `ms01`.
- Phase 6 exact rule-switch solving is implemented and verified for `rs01`.
- Phase 7 exact time/reactive solving is implemented and verified for `sv01`.
- Phase 8 exact hybrid-construction solving is implemented and verified for `tb01`.
- The `v4` tree is tested for isolation from legacy `v3_1` references.

## What Is Not Yet Implemented

- No movement family beyond `ul01`, `fs01`, `fs02`, `fs03`, `tp01`, `ic01`, `va01`, `pb01`, `pb02`, and `pb03` is implemented under `src/v4/movement`.
- No click/perception family beyond `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, and `mm01` is implemented under `src/v4/click`.
- No memory-hidden family beyond `ms01` is implemented under `src/v4/memory_hidden`.
- No rule-switch family beyond `rs01` is implemented under `src/v4/rule_switch`.
- No time/reactive family beyond `sv01` is implemented under `src/v4/time_reactive`.
- No hybrid-construction family beyond `tb01` is implemented under `src/v4/hybrid_construction`.

## Current Next-Step Options

- Keep the current implemented stack stable and maintain the docs against the verified repo state.
- Expand an existing implemented package to its next unimplemented family.
- Clean up the remaining outdated historical checklist docs called out in [documentation_gap_checklist.md](documentation_gap_checklist.md).

## Major Docs

- [current_status.md](current_status.md): precise implementation and verification status.
- [implemented_vs_planned.md](implemented_vs_planned.md): implemented vs documented-only split by track.
- [navigation_index.md](navigation_index.md): compact document navigation.
- [source_to_doc_map.md](source_to_doc_map.md): source-package documentation coverage.
- [test_to_doc_map.md](test_to_doc_map.md): test-package documentation coverage.
- [documentation_gap_checklist.md](documentation_gap_checklist.md): missing and outdated documentation audit.
- [agent_contracts/README.md](agent_contracts/README.md): Phase 1 contract overview.
- [closed_loop/stage2_overview.md](closed_loop/stage2_overview.md): Stage 2 closed-loop overview.
- [movement/phase3_overview.md](movement/phase3_overview.md): Phase 3 movement overview.
- [movement/fs02_mechanics.md](movement/fs02_mechanics.md): `fs02` mechanics.
- [movement/fs03_mechanics.md](movement/fs03_mechanics.md): `fs03` mechanics.
- [movement/pb02_mechanics.md](movement/pb02_mechanics.md): `pb02` mechanics.
- [movement/pb03_mechanics.md](movement/pb03_mechanics.md): `pb03` mechanics.
- [click_perception/phase4_overview.md](click_perception/phase4_overview.md): Phase 4 click/perception overview.
- [memory_hidden/phase5_overview.md](memory_hidden/phase5_overview.md): Phase 5 memory-hidden overview.
- [rule_switch/phase6_overview.md](rule_switch/phase6_overview.md): Phase 6 rule-switch overview.
- [time_reactive/phase7_overview.md](time_reactive/phase7_overview.md): Phase 7 time/reactive overview.
- [hybrid_construction/phase8_overview.md](hybrid_construction/phase8_overview.md): Phase 8 hybrid-construction overview.
