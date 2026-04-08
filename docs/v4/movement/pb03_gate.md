Status: implemented and verified
Scope: movement doc: `pb03` gate
Source of truth: `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_pb03.py`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb03`

# `pb03` Gate

The `pb03` gate verifies:

- exact env-backed typed-state extraction for the crate, true goal, and decoy pad
- exact transition behavior for ordinary moves and decoy-loss pushes
- exact search to a certifying plan
- successful Stage 2 live-policy execution on a real local level
- no legacy `v3_1` runtime usage on the movement path
