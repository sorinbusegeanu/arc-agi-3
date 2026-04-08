Status: implemented and verified
Scope: movement doc: `pb03` mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/pb03/63be02fb/pb03.py`, `/home/zodrak/zod/tests/v4/movement/test_transition_model_pb03.py`, `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_pb03.py`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb03`

# `pb03` Mechanics

- Family: exact push-block movement family inside `src/v4/movement`
- Local env source: `other_repos/arc-interactive/environment_files/pb03/63be02fb/pb03.py`
- Local repo mechanics are one crate, one real goal, and one decoy lose pad

## Implemented Mechanics

- One crate and one true goal are represented explicitly.
- Decoy lose-pad cells are represented explicitly.
- Pushing the crate onto a decoy pad is immediate terminal loss.
- Pushing the crate onto the true goal is success.
- Search treats decoy placements as losing states and avoids them.

## Verified Coverage

- Typed-state build from env metadata plus observation.
- Decoy-loss transition tests.
- Exact planner success.
- Stage 2 live controller solves a real `pb03` level under the movement policy.
