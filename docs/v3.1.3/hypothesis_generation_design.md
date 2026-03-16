# Hypothesis Generation Design

This layer adds two advisory hypothesis generators above the dedicated mechanic graph:

- deterministic hypothesis generator
- optional LLM advisory generator

Neither becomes authoritative world state on generation. Both remain hypothesis-tier until later validation.

## Ownership

- deterministic generation is owned by deterministic rule modules and emits `deterministic_hypothesis`
- LLM generation is advisory only and emits `llm_hypothesis`
- cumulative graph facts remain owned by `MechanicGraphAgent`
- session-scoped proposal state is kept separately in `HypothesisRegistry`

## Shared Proposal Schema

Both sources use the shared proposal records:

- `HypothesisEdgeProposal`
- `HypothesisPathProposal`
- `HypothesisTestProposal`
- `HypothesisBundle`

All proposals carry source provenance, support refs, contradiction refs, confidence, novelty, validation requirement, and round/episode provenance.

## Validation Rules

- deterministic proposals are scored only from deterministic evidence
- LLM outputs must pass schema validation
- LLM proposals may reference only known node ids and allowed edge/path kinds
- malformed or unsupported LLM outputs are rejected
- LLM proposals are confidence-capped and remain hypothesis-tier

## Planner Priority Order

1. observed graph paths
2. validated deterministic paths
3. strongly supported deterministic hypotheses
4. deterministic test proposals
5. validated LLM test proposals
6. unvalidated LLM hypotheses as low-priority experiments

## Durable Memory Rules

- deterministic supported paths persist separately from LLM supported paths
- deterministic/LLM agreement persists separately
- repeated validated hypotheses can become durable aggregates
- contradicted LLM proposals are stored separately
- LLM proposals do not become durable-ready without later deterministic or observed validation

## Comparison Metrics

- proposal precision by source
- contradiction rate by source
- validated edge recall by source
- first correct prerequisite round by source
- first win after proposal by source
- unnecessary test count by source
- source agreement rate
- disagreement resolved by later evidence

## Gating Policy

LLM generation is allowed only when no strong supported mechanic path exists or ambiguity/failure/contradiction thresholds are crossed.

LLM generation is blocked when:

- a strong observed path exists
- a validated deterministic path exists
- call budget is exhausted
- there are no open mechanic questions

## Why Both Stay Hypothesis-Tier

The purpose of both generators is to propose explanations and tests, not to overwrite authoritative graph facts.

Only later direct support through normal graph merge and validation can strengthen a relation into observed graph state.
