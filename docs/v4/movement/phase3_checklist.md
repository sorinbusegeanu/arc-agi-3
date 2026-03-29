Status: implemented and verified
Scope: movement doc: phase3 checklist
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Phase 3 Checklist

- [x] docs created
- [x] movement package created
- [x] typed state implemented
- [x] family adapters implemented
- [x] no `_LEVEL_DATA` or equivalent per-level spec table in movement adapters
- [x] family layout and mechanic extraction is environment-backed
- [x] exact transition model implemented
- [x] search implemented
- [x] solver policy implemented
- [x] Stage 2 integration implemented
- [x] `ul01` gate passes
- [x] `fs01` gate passes
- [x] `fs01` typed-state fields implemented
- [x] `fs01` adapter implemented
- [x] `fs01` transition semantics tested against real env
- [x] `fs01` search uses augmented state, not position only
- [x] `fs01` solver policy solves at least one verified real level
- [x] `fs01` door semantics are represented explicitly in typed state, not hidden in transition-model assumptions
- [x] `fs01` path uses no forbidden runtime dependencies
- [x] `tp01` gate passes
- [x] `tp01` teleporter typed-state fields implemented
- [x] `tp01` adapter implemented
- [x] `tp01` transition semantics tested against real env
- [x] `tp01` search handles symmetric warp transitions correctly
- [x] `tp01` solver policy solves at least one verified real level
- [x] `tp01` path uses no forbidden runtime dependencies
- [x] `ic01` gate passes
- [x] `ic01` ice and hazard typed-state fields implemented
- [x] `ic01` adapter implemented
- [x] `ic01` transition semantics tested against real env
- [x] `ic01` search handles forced slide successors correctly
- [x] `ic01` solver policy solves at least one verified real level
- [x] `ic01` path uses no forbidden runtime dependencies
- [x] `va01` gate passes
- [x] `va01` coverage typed-state fields implemented
- [x] `va01` adapter implemented
- [x] `va01` transition semantics tested against real env
- [x] `va01` search handles coverage-state progression correctly
- [x] `va01` solver policy solves at least one verified real level
- [x] `va01` path uses no forbidden runtime dependencies
- [x] `pb01` gate passes
- [x] `pb01` pushable-object typed-state fields implemented
- [x] `pb01` adapter implemented
- [x] `pb01` transition semantics tested against real env
- [x] `pb01` search handles push-state progression correctly
- [x] `pb01` solver policy solves at least one verified real level
- [x] `pb01` path uses no forbidden runtime dependencies
- [x] all family gates pass together
- [x] full movement suite passes
- [x] full isolated v4 suite passes
- [x] no Stage 2 regressions
- [x] no forbidden runtime dependencies across any movement family
- [x] summary and consolidation docs written
- [x] no forbidden runtime dependencies
- [x] failures localizable to builder, adapter, transition, search, or policy boundaries
- [x] Phase 3 gate runs against `PYTHONPATH=src python tests/v4/run_suite.py` only unless legacy coverage is explicitly requested
