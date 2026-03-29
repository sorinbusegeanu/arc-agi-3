Status: implemented and verified
Scope: Phase 5 hidden-information movement track overview for `src/v4/memory_hidden` and `tests/v4/memory_hidden`
Source of truth: `src/v4/memory_hidden`, `tests/v4/memory_hidden`
Last verified against: repo state on 2026-03-29

# Phase 5 overview

`memory_hidden` is the v4 exact hidden-information movement track.

Implemented family:
- `ms01`

Current package scope:
- family-explicit typed state for `ms01`
- local count-consistency facts from revealed clue cells only
- exact search over the currently known safe region
- deterministic Stage 2 policy output through `PolicyDecisionV4`
