Status: implemented and verified
Scope: tests doc: test longer horizon loop stability spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Longer-Horizon Loop-Stability Test Spec

These tests must:

- run a bounded but longer Stage 2 session than the `ez01` through `ez04` tutorial runs
- use `ul01`, `tt01`, or both, depending on real local availability
- verify monotonic step index
- verify one ledger record per executed action
- verify memory remains bounded
- verify no branch, merge, or probe logic appears after many steps
- verify repeated non-terminal stepping does not corrupt contract records

## Acceptable Stop Reasons

- terminal win
- terminal fail
- hard step budget
- explicit invalid-state abort
