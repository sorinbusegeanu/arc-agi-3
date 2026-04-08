Status: implemented and verified
Scope: movement doc: next family order
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`

# Next Family Order

The Phase 3 rollout order implemented in the current repo is:

1. `ul01`
2. `fs01`
3. `tp01`
4. `ic01`
5. `va01`
6. `pb01`
7. `pb02`
8. `pb03`
9. `fs02`
10. `fs03`

All ten families above now exist under `src/v4/movement` and are covered by dedicated gate tests plus Phase 3 consolidation/regression coverage.

`sk01` remains the next listed movement family after the currently implemented ten-family slice.
