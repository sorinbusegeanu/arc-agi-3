Status: implemented and verified
Scope: click perception doc: next family order
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

The Phase 4 rollout order implemented in the current repo is:

1. `pt01`
2. `sy01`
3. `ff01`
4. `sq01`
5. `wm01`
6. `mm01`

All six families above now exist under `src/v4/click` and have dedicated gate coverage in `tests/v4/click`.

No later click/perception family is implemented under `src/v4/click` after `mm01`.
