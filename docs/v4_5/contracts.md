# v4.5 Contracts

v4.5 defines explicit typed contracts between control-plane stages.

## Core Contracts

- `AgentInput`
- `DiscoveryReport`
- `HypothesisReport`
- `PlanCandidateSet`
- `PlanDecision`
- `OutcomeReport`
- `BoardPerceptionReport`
- `LevelOptimizationReport`
- `GameOptimizationReport`
- `AdvisoryRequest`
- `AdvisoryResponse`

## Contract Rules

- every report includes `schema_version`
- every report includes `agent_name`
- every report includes `round_id`
- every decision/output includes rationale codes
- advisory outputs are marked advisory-only
- non-advisory contracts do not include LLM-specific fields

## Intent

These contracts separate:

- authoritative execution state
- planner/plugin proposals
- advisory suggestions
- offline optimization outputs

This keeps v4.5 deterministic at the control plane even when advisory support is enabled.
