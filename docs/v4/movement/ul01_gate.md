Status: implemented and verified
Scope: movement doc: ul01 gate
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ul01` Gate

The first Phase 3 gate passes only when all of the following are true:

- `ul01` typed-state build passes
- `ul01` transition-model tests pass
- `ul01` search tests pass
- `ul01` solver policy solves at least one verified real level end to end in the Stage 2 loop
- no forbidden runtime dependencies are present

`fs01` work begins after this gate passes.
