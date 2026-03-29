Status: implemented and verified
Scope: `sq01` Phase 4 gate
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_sq01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `sq01` Gate

`sq01` is an implemented Phase 4 family.

The current gate proves:

- `sq01` typed state builds from live observation
- sequence order and progress fields are extracted from the live family state
- center-of-block click payloads advance the real family sequence
- the Stage 2 loop can carry `sq01` through the live pending-advance state without aborting
- the current family policy clears at least one real level in a bounded live run
