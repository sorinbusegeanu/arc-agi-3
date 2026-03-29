Status: implemented and verified
Scope: `wm01` Phase 4 gate
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_wm01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `wm01` Gate

`wm01` is an implemented Phase 4 family.

The current gate proves:

- `wm01` typed state builds from live observation
- visible mole state and click radius are present in typed state
- the transition model clears a visible mole on a typed hit
- the Stage 2 loop runs `wm01` without legacy runtime dependencies
- the current family policy completes at least one real level in a bounded live run
