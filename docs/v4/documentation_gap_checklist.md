Status: implemented and verified
Scope: Documentation audit against implemented v4 source and tests
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`

# Documentation Gap Checklist

## Existing Docs

- `agent_contracts/*` covers the implemented Phase 1 contract surface.
- `closed_loop/*` covers the implemented Stage 2 runtime, parsed state, local memory, and gates.
- `movement/*` covers the implemented Phase 3 movement track, including dedicated docs for `pb02`, `pb03`, `fs02`, and `fs03`.
- `click_perception/*` covers the implemented Phase 4 click/perception track and all six implemented click families.
- `memory_hidden/*` covers the implemented Phase 5 memory-hidden package and `ms01`.
- `rule_switch/*` covers the implemented Phase 6 rule-switch package and `rs01`.
- `time_reactive/*` covers the implemented Phase 7 time/reactive package and `sv01`.
- `hybrid_construction/*` covers the implemented Phase 8 hybrid-construction package and `tb01`.

## Missing Docs

- No additional missing major doc was found for the currently implemented `src/v4` package set.

## Outdated Docs

- `agent_contracts/phase1_checklist.md`
- `closed_loop/stage2_checklist.md`
- `tests/README_stage2_next_gate.md`

## Docs That Mention Planned Features Not Yet Implemented

- `reference/arc_interactive_games.md` lists many local game families that do not have corresponding implementations under the current `src/v4` package set.
- `movement/family_coverage.md` mentions deferred movement work beyond the currently implemented ten-family movement slice.
- `click_perception/family_coverage.md` mentions deferred click work beyond the current six-family slice.
- `memory_hidden/family_coverage.md`, `rule_switch/family_coverage.md`, `time_reactive/family_coverage.md`, and `hybrid_construction/family_coverage.md` remain single-family docs because only one family is implemented in each package.

## Docs That Are Missing Links From The Top-Level Overview

- None after this update. The top-level `README.md` links the major current-status, audit, and track overview docs.

## Docs That Need Status Tags

- None after this update. All existing markdown files under `docs/v4` have a status block.

## Checklist

- [ ] Update `agent_contracts/phase1_checklist.md` to match the implemented Phase 1 state.
- [ ] Update `closed_loop/stage2_checklist.md` to match the implemented Stage 2 state.
- [ ] Update `tests/README_stage2_next_gate.md` to reflect the current implemented post-Stage-2 tracks.
