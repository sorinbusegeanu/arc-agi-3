Status: implemented and verified
Scope: Stage 2 policy surface
Source of truth: `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/test_policy.py`, `/home/zodrak/zod/tests/v4/easy_games/_helpers.py`, `/home/zodrak/zod/tests/v4/movement/test_solver_policy_*`, `/home/zodrak/zod/tests/v4/click/test_solver_policy_*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 Policy Surface

`src/v4/policy` defines the shared decision interface used by the Stage 2 loop and by the movement and click solver heads.

## Implemented Modules

- `policyBase.py`: `PolicyBaseV4`, `PolicyDecisionV4`, and `legal_action_from_id`.
- `primitivePolicy.py`: primitive-action policy helper.
- `shortPlanPolicy.py`: short-plan policy helper.

## Current Status

Status: implemented and verified

Evidence:

- The package exists under `src/v4/policy`.
- `tests/v4/closed_loop/test_policy.py` exercises the Stage 2 policy surface.
- Movement and click solver policy tests depend on the same shared policy interface.

## Boundary

- Policies consume `ParsedStateV4`.
- Policies emit exactly one legal primitive action or one short plan through `PolicyDecisionV4`.
- Short plans remain bounded and interruptible.
- Action legality is still enforced against the current observation through the contract validator.

## Exclusions

- No blackboard or branch-selection state.
- No durable planner memory inside the policy surface.
- No direct environment stepping from the policy package.
