Status: implemented and verified
Scope: `sq01` family mechanics and typed-state surface
Source of truth: `/home/zodrak/zod/src/v4/click/familyAdapters.py`, `/home/zodrak/zod/src/v4/click/solverPolicy.py`, `/home/zodrak/zod/src/v4/click/transitionModel.py`, `/home/zodrak/zod/tests/v4/click/test_transition_model_sq01.py`, `/home/zodrak/zod/tests/v4/click/test_phase4_gate_sq01.py`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `sq01` Mechanics

`sq01` is represented in `v4` as an ordered click-sequence family.

## Implemented Typed State

- authored sequence order from local level config
- current sequence progress derived from which blocks remain visible
- color-to-click payload mapping using the center of each live 2x2 block

## Implemented Transition And Integration Semantics

- correct next click advances sequence progress
- wrong click resets progress in the typed transition model
- after the last correct click, the live env spends several pending-advance frames with no remaining blocks
- `v4` keeps `sq01` runnable during those pending-advance frames with a placeholder legal click candidate so the Stage 2 loop can advance cleanly

## Current Solver Boundary

`ClickSolverPolicyV4` uses deterministic next-click selection for `sq01`. It does not use bounded search for this family.
