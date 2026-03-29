Status: implemented and verified
Scope: `mm01` family mechanics and typed-state surface
Source of truth: `/home/zodrak/zod/src/v4/click/familyAdapters.py`, `/home/zodrak/zod/src/v4/click/solverPolicy.py`, `/home/zodrak/zod/src/v4/click/transitionModel.py`, `/home/zodrak/zod/tests/v4/click/test_transition_model_mm01.py`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_mm01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `mm01` Mechanics

`mm01` is represented in `v4` as a reveal-and-match memory family.

## Implemented Typed State

- exact slot colors from local level config
- current hidden, revealed, and matched slot sets from direct observation
- slot geometry for stable slot-to-payload conversion

## Implemented Transition And Selection Semantics

- clicking a hidden tile reveals it
- matching revealed colors promotes those slots into the matched set
- policy selection uses slot geometry directly so slot index stays aligned with payload targeting

## Current Solver Boundary

`ClickSolverPolicyV4` uses deterministic reveal-then-match selection for `mm01`. It does not use bounded search for this family.
