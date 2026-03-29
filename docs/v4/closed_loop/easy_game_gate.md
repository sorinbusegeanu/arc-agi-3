Status: implemented and verified
Scope: closed loop doc: easy game gate
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 Easy-Game Gate

The Stage 2 easy-game gate for `ez01` through `ez04` passes only when all of the following are true:

- all `ez01` through `ez04` tests pass
- at least one deterministic win-path test passes for each of the four games
- no legacy v3.1 runtime path is active
- every executed step yields valid observation, action, transition, and step-result records
- failures are localizable to a single Stage 2 bucket

Phase 3 must not begin until this gate passes.
