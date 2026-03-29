# Executable Subgoal Chain Design

Chain-shaped reasoning must become chain-shaped execution.

Current design requirements:
- chain manager owns runtime progress
- planner selects a chain or updates the active chain context
- executor executes one chain step per cycle, never the whole chain at once
- outcome summaries decide whether the chain advances, retries, aborts, or completes

Runtime contract:
- `SubgoalChain` is the first-class chain object
- `SubgoalStep` is the executable unit
- `SubgoalChainManager` owns at most one active chain per run
- planner emits `selected_subgoal_chain`, `selected_subgoal_step_id`, and `selected_subgoal_step_kind`
- executor request carries the selected chain and current step
- outcome summaries emit step-level chain results:
  - `step_success`
  - `step_failure_reason`
  - `chain_should_advance`
  - `chain_should_retry`
  - `chain_should_abort`

Ownership boundary:
- planner does not mutate retry counts or completion state
- executor does not advance the chain itself
- round runner updates the chain manager from the outcome summary
- ledger records chain start, step success/failure, advance, abort, and completion

Safety requirements:
- retry budgets are enforced by the chain manager
- replanning is requested on contradiction, repeated failure, or missing expected evidence
- avatar-confidence-aware outcomes weaken spatial chain-step success when localization is weak

Planning mode telemetry:
- `previous_planning_mode` is a committed cross-round telemetry field
- it must be sourced from the prior round committed mode only
- it must not be synthesized from defaults or tentative local mode calculations
