Status: implemented and verified
Scope: `rs01` Phase 6 gate
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/*`, `/home/zodrak/zod/tests/v4/rule_switch/test_phase6_gate_rs01.py`
Last verified against: unknown

# `rs01` Gate

The current gate proves:

- `rs01` typed state builds from the `v4` parsed-state surface
- the package extracts the active safe color and visible targets
- the solver returns only Stage 2-compatible movement actions
- the live loop runs without patched legacy `v3_1` surfaces
