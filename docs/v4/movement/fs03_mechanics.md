Status: implemented; gate-covered, live regression still mixed
Scope: movement doc: `fs03` mechanics
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/fs03/63be02fb/fs03.py`, `/home/zodrak/zod/tests/v4/movement/test_transition_model_fs03.py`, `/home/zodrak/zod/tests/v4/movement/test_phase3_gate_fs03.py`
Last verified against: current repo state on 2026-03-30; targeted movement tests for `fs03` and targeted manual live regression for `fs03`

# `fs03` Mechanics

- Family: exact floor-switch movement family inside `src/v4/movement`
- Local env source: `other_repos/arc-interactive/environment_files/fs03/63be02fb/fs03.py`
- Local repo rule: distinct switch activations latch, and the door is removed permanently once `activated >= required_plates`

## Implemented Mechanics

- Switch positions, threshold `k`, activated-switch bits, door state, and target cells are explicit.
- Distinct switch activations latch persistently.
- Door opening uses the env-backed threshold from metadata.
- Closed-door cells block motion exactly until the threshold is met.
- Search reasons over activated-switch and door state explicitly.
- family-local fallback ranking now treats repeated activation-order dead loops as a distinct no-progress case in the live runner.

## Verified Coverage

- Env-backed threshold extraction.
- Transition tests for below-threshold and threshold-crossing behavior.
- Exact planner success on level 0.
- Stage 2 live controller solves level 0 before the next-level controller boundary.
- Manual live regression still shows repeated non-progress cycles, so the family is not yet live-verified end-to-end.
