Status: implemented and verified
Scope: `ff01` Phase 4 gate
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_ff01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ff01` Gate

`ff01` is an implemented Phase 4 family.

The current gate proves:

- `ff01` typed state builds from live observation
- configured fill regions are present in typed state
- the transition model marks a clicked region as filled
- the Stage 2 loop runs `ff01` without legacy runtime dependencies
- the current family policy clears at least one real level in a bounded live run
