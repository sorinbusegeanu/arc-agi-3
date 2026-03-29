Status: implemented and verified
Scope: `ff01` family mechanics and typed-state surface
Source of truth: `/home/zodrak/zod/src/v4/click/familyAdapters.py`, `/home/zodrak/zod/src/v4/click/transitionModel.py`, `/home/zodrak/zod/tests/v4/click/test_transition_model_ff01.py`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_ff01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ff01` Mechanics

`ff01` is represented in `v4` as a closed-region fill family.

## Implemented Typed State

- exact fill-region interiors from local level config
- current filled-region index set from direct observation
- clickable payloads at region centers

## Implemented Transition Semantics

- clicking inside a configured region marks that region filled
- filled-region state is carried in typed state only
- no invented topology or planner metadata is added

## Current Solver Boundary

`ClickSolverPolicyV4` uses deterministic region selection for `ff01`. It does not use bounded search for this family.
