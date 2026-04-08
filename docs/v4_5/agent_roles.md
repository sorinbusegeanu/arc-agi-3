# v4.5 Agent Roles

## Orchestrator Agent

- Purpose: enforce stage ordering, own execution authority, and manage stop conditions
- Inputs: current observation, memory snapshot, prior reports, stage context, stop conditions
- Outputs: next stage selection, committed action or prefix, stage transition record
- Authority boundaries: only execution authority in v4.5
- Reused v4 modules: `src/v4/runtime/*`, `src/v4/policy/*`
- Failure behavior: fail closed, emit stop or retry-stage decision

## Discovery Agent

- Purpose: build initial scene summary and bounded bootstrap probe summary
- Inputs: parsed observation, prior memory, unseen-level flags, optional advisory handle
- Outputs: `DiscoveryReport`
- Shared support: may call the board perception module during bootstrap and on demand, but the module is not a live agent
- Authority boundaries: cannot execute live actions directly
- Reused v4 modules: `src/v4/state/*`, `src/v4/exploration/*`, `src/v4/affordance/*`, `src/v4/experiments/disambiguationPlanner.py`
- Failure behavior: emit minimal deterministic report with explicit gaps

## Hypothesis Agent

- Purpose: maintain ranked hypotheses and current mode labels
- Inputs: `AgentInput`, `DiscoveryReport`, prior hypotheses, evidence summary
- Outputs: `HypothesisReport`
- Authority boundaries: cannot commit live actions
- Reused v4 modules: `src/v4/belief/*`, `src/v4/hypothesis/*`, `src/v4/experiments/expectedEvidence.py`
- Failure behavior: emit conservative ranked hypotheses and fail-closed mode labels

## Planner Agent

- Purpose: gather planner plugin outputs, verify candidates, and return one deterministic decision
- Inputs: `PlannerContext`, plugin registry, optional advisory response
- Outputs: `PlanDecision`
- Authority boundaries: may verify internally but may not execute
- Reused v4 modules: `src/v4/planning/*`, `src/v4/subgoals/*`
- Failure behavior: emit no-op or no-plan decision with rationale codes

## Outcome Agent

- Purpose: compare expected vs observed effect and classify outcome
- Inputs: committed decision, pre/post observations, current memory, hypothesis state
- Outputs: `OutcomeReport`
- Authority boundaries: may update memory-facing outputs but may not execute
- Reused v4 modules: `src/v4/belief/*`, `src/v4/hypothesis/*`, `src/v4/temporal/*`, `src/v4/composition/*`, `src/v4/memory/*`, `src/v4/analysis/*`
- Failure behavior: classify as contradiction or non-progress conservatively

## Post-Level Optimizer Agent

- Purpose: inspect completed level traces and identify wasted or redundant prefixes
- Inputs: completed level trace, ledger data, prior level hints
- Outputs: `LevelOptimizationReport`
- Authority boundaries: offline only, no live execution role
- Reused v4 modules: trace/analysis surfaces through v4.5 adapters
- Failure behavior: emit empty hint set

## Post-Game Optimizer Agent

- Purpose: consolidate reusable game priors and mechanic notes across levels
- Inputs: cross-level summaries, optimization reports, ledger summaries
- Outputs: `GameOptimizationReport`
- Authority boundaries: offline only, no live execution role
- Reused v4 modules: trace/analysis surfaces through v4.5 adapters
- Failure behavior: emit empty priors and notes
