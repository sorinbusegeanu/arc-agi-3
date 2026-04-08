Step 8 Trace vs Family-Exact Comparison

## Purpose

This comparison exists to separate candidate-generation activation in the certified planner from actual live win-path behavior in the exact family solver.

## Policy modes

- `certified` -> `CertifiedPlannerPolicyV4`
- `family_exact` -> exact implemented solver for the game, currently `TimeReactiveSolverPolicyV4` for `sv01` and `HybridConstructionSolverPolicyV4` for `tb01`

## Required comparison outputs

- success
- stop_reason
- step_count
- selected_policy_name
- trace_row_count
- last_selected_goal_kind
- last_selected_subgoal_kind

## Interpretation

- if `certified` shows correct Step 7 or Step 8 activation but `family_exact` still fails, the blocker is in the exact live solver path
- if both fail before meaningful trace rows, the blocker is likely startup/runtime/worker-side
- if `family_exact` wins and `certified` does not, that is expected and does not indicate a Step 8 regression
