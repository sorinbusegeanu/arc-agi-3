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
- `certifiedPlannerPolicy.py`: Step 1 grounded candidate-plan policy.

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

## Certified Planner Step 1

- `CertifiedPlannerPolicyV4` generates a small deterministic set of grounded candidate plans from currently legal actions.
- The scorer ranks them deterministically.
- The verifier certifies one legal prefix without simulating branches or hypotheses.
- Runtime still executes only the first action from the certified short plan.
- Step 1 adds no hypotheses, no subgoals, and no branching runtime behavior.

## Certified Planner Step 2

- The planner now selects one explicit subgoal before generating candidate plans.
- Subgoals are deterministic and extracted from typed state plus known family classification.
- There is still no belief state.
- There are still no hypotheses.
- There is still no branching.
- Step 2 still executes only the first action of the certified short plan.

## Certified Planner Step 4

- When hidden information remains, the planner may select `reveal_information` as the active subgoal.
- Step 4 uses bounded, deterministic probe templates.
- Only low-risk exploration candidates are allowed.
- Information gain is scored explicitly.
- There are still no hypotheses.
- There is still no branching.
- Runtime still executes only the first action of the certified short plan.

## Step 5

- Policies may observe `hypothesis_reference`.
- Step 5 does not require policies to use hypotheses directly.
- Action execution flow remains unchanged.
- Disambiguation planning is deferred to Step 6.

## Step 6

- When competing hypotheses exist, the planner may select `disambiguate_hypothesis` as the active subgoal.
- Step 6 uses bounded deterministic experiment templates.
- Expected evidence is attached to disambiguation candidates.
- Post-step reconciliation may collapse competing hypotheses.
- There is still no branching.
- Runtime still executes only the first action of the certified short plan.

## Step 7

- When temporal state is present for `sv01`, the planner may select `preserve_safety_margin`.
- Temporal candidates are filtered by temporal safety verification.
- Contingent notes may be attached to expected effects.
- Runtime still executes only the first action of the certified short plan.

## Step 8

- When composition state is present for `tb01`, the planner may select a hybrid subgoal.
- Step 8 attaches present domains and cross-domain effect codes to hybrid candidates.
- Runtime still executes only the first action of the certified short plan.
- No branching is introduced in Step 8.

## Tracing Note

- Policy annotations now expose per-step generated, accepted, and selected counts for Step 6, Step 7, and Step 8 candidate families.
- Policy annotations now expose extracted subgoal kinds and per-subgoal progress rows.
- These tracing fields are diagnostic only.
- They do not change action selection semantics.

## Exclusions

- No blackboard or branch-selection state.
- No durable planner memory inside the policy surface.
- No direct environment stepping from the policy package.
