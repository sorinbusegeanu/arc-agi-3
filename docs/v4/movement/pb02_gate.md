Status: implemented and verified
Scope: movement doc: `pb02` gate
Source of truth: `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_pb02.py`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`

# `pb02` Gate

The `pb02` gate currently proves:

- the movement builder extracts the exact two-crate level-0 layout
- the transition model preserves legal movement and single-crate push semantics
- exact search finds a certifying level-0 plan
- replaying that plan against the real env completes the level
- the policy replans through the hidden-on-goal boundary and uses the existing per-episode carry state without changing the Stage 2 runtime shape
- the movement policy path remains isolated from legacy `v3_1` surfaces
