Status: implemented and verified
Scope: movement doc: `fs03` gate
Source of truth: `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_fs03.py`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `fs03`

# `fs03` Gate

The `fs03` gate verifies:

- exact env-backed threshold typed-state extraction
- exact search to a legal level-0 plan
- Stage 2 live-policy execution reaches `levels_completed >= 1`
- no legacy `v3_1` runtime usage on the movement path

Current boundary:

- after solving level 0, the live controller may still stop with `invalid_state_abort` while parsing the next level
