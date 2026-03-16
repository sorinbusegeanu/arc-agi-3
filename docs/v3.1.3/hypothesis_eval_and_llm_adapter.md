# Hypothesis Eval And LLM Adapter

## Local Adapter Contract

- `LocalLLMAdapter` is provider-agnostic and receives `LocalLLMRequest`
- adapters return `LocalLLMResponse`
- generation is structured JSON only
- adapters never mutate runtime state

## Safe Fail-Open Behavior

- when LLM is disabled, the runtime uses the stub adapter
- when provider config is invalid, the runtime falls back to the stub adapter
- failed adapter calls do not raise into the round loop
- failed calls produce empty LLM hypothesis output with error metadata

## Gating Path

- deterministic hypotheses are always generated first
- LLM calls are gated by:
  - no supported path
  - repeated failures
  - contradiction pressure
  - tied deterministic hypotheses
  - graph ambiguity
- strong observed or validated deterministic paths suppress LLM calls

## Validator Path

- schema validation
- node id validation
- edge/path kind validation
- semantic duplicate suppression against deterministic proposals
- exact duplicate suppression against prior LLM proposals
- confidence cap enforcement

## Registry Lifecycle

- states: `new`, `supported`, `contradicted`, `stale`, `validated`, `rejected`
- graph evidence can move proposals to supported or contradicted
- strong later evidence can mark proposals validated
- untouched speculative proposals can become stale

## Closed-Loop Experiment Validation

- hypothesis-test candidates preserve experiment intent through execution
- outcomes emit experiment support/contradiction summaries
- mechanic extraction feeds those summaries back into hypothesis metadata and graph feedback

## Comparison Protocol

Run three modes with the same games, seeds, planner settings, and round budgets:

1. deterministic only
2. deterministic plus LLM enabled
3. deterministic plus LLM configured but gated tight

## Output Artifacts

- `deterministic_hypotheses.json`
- `llm_hypotheses.json`
- `hypothesis_agreement.json`
- `hypothesis_validation_summary.json`
- `path_to_victory_candidates.json`
- `llm_usage_summary.json`
- `hypothesis_lifecycle_summary.json`
- `experiment_results_summary.json`
- `hypothesis_comparison_report.json`
