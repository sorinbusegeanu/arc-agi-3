Status: implemented and verified
Scope: gate coverage for `tests/v4/memory_hidden/test_phase5_gate_ms01.py`
Source of truth: `tests/v4/memory_hidden/test_phase5_gate_ms01.py`
Last verified against: repo state on 2026-03-29

# ms01 gate

Gate checks:
- uses only v4 parsed state and Stage 2-compatible `PolicyDecisionV4`
- patches legacy `v3_1` runtime, merge, hypothesis, and chain-manager surfaces
- validates typed-state family identity and bounded parsed-memory surface
- validates v4 transition and step records on live controller execution
