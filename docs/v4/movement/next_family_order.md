Status: implemented and verified
Scope: movement doc: next family order
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Next Family Order

The Phase 3 rollout order implemented in the current repo is:

1. `ul01`
2. `fs01`
3. `tp01`
4. `ic01`
5. `va01`
6. `pb01`

All six families above now exist under `src/v4/movement` and are covered by dedicated gate tests plus Phase 3 consolidation/regression coverage.

No later movement family is implemented under `src/v4/movement` after `pb01`.
