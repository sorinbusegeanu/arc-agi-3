Status: implemented and verified
Scope: `wm01` family mechanics and typed-state surface
Source of truth: `/home/zodrak/zod/src/v4/click/familyAdapters.py`, `/home/zodrak/zod/src/v4/click/transitionModel.py`, `/home/zodrak/zod/src/v4/click/solverPolicy.py`, `/home/zodrak/zod/tests/v4/click/test_transition_model_wm01.py`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_wm01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `wm01` Mechanics

`wm01` is represented in `v4` as a visible-only timing click family.

## Implemented Typed State

- visible hole positions from local level config
- currently active mole cells from direct observation only
- fixed click radius used by the current local solver slice

## Implemented Transition Semantics

- a visible hit clears the active mole set in the typed transition
- the model does not infer hidden timing or spawn state

## Current Solver Boundary

`ClickSolverPolicyV4` uses deterministic visible-step selection for `wm01`. It does not use bounded search for this family.
