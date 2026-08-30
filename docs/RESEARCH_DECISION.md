# RESEARCH_DECISION

`RESEARCH_DECISION.md` is the authoritative analysis and next-experiment specification.

It is written after analysing `EXPERIMENT_EVIDENCE.md` and is consumed by the next run as the declared intervention.

The analysis must explicitly state what must change, what must not change, what result is predicted, and what result falsifies the proposed explanation.

## Research loop position

```text
EXPERIMENT_EVIDENCE N
        ↓
causal analysis
        ↓
RESEARCH_DECISION N
        ↓
explicit implementation/change
        ↓
EXPERIMENT N+1
```

## Machine-readable decision metadata

The JSON object between the markers below is authoritative. The runtime captures it before the next experiment starts.

<!-- RESEARCH_DECISION_METADATA_BEGIN -->
{
  "change_id": "R001-T1",
  "change_type": "TELEMETRY",
  "target_hypothesis": "H05",
  "target_causal_edge": "M1N_TO_M2_TO_M3_ROLE",
  "target_files": [],
  "target_functions": [],
  "change": [],
  "must_not_change": [],
  "games": "research_1",
  "steps_per_game": 20000,
  "memory_policy": "REUSE",
  "primary_metric": "",
  "predicted_change": "",
  "minimum_meaningful_effect": "",
  "expected_unchanged_metrics": [],
  "falsifier": "",
  "decision_rule": []
}
<!-- RESEARCH_DECISION_METADATA_END -->

Replace the example values with the actual next experiment before running it.

## Experiment analysed

- experiment_id:
- revision:
- evidence file:

## Conclusion

Choose exactly one primary classification:

- `BUG`
- `EXPERIMENTAL_ARTIFACT`
- `PERFORMANCE_BOTTLENECK`
- `MECHANISM_FAILURE`
- `ARCHITECTURAL_LIMITATION`
- `HYPOTHESIS_CONTRADICTED`
- `INSUFFICIENT_EVIDENCE`

## Observations

List only facts supported by `EXPERIMENT_EVIDENCE.md`.

Separate experiment-local evidence from cumulative historical state.

## Earliest causal bottleneck

State:

- causal edge;
- status: broken / unresolved / passed;
- direct evidence;
- missing evidence if unresolved.

Do not diagnose a downstream hypothesis while an upstream mechanism required to test it is unresolved.

## Hypothesis assessment

For every affected Hxx use one of:

- `SUPPORTED`
- `CONTRADICTED`
- `UNTESTED`
- `INSUFFICIENT_EVIDENCE`

A bug fix does not validate a cognitive hypothesis.

## Competing explanations

Provide at least two explanations when evidence allows it.

For each:

- evidence for;
- evidence against;
- observation that would discriminate it from alternatives.

## Research decision

Select the smallest next experiment that most reduces uncertainty.

Prefer, in order:

1. missing telemetry;
2. bug verification/fix;
3. localized parameter intervention;
4. mechanism intervention;
5. architecture change.

Do not redesign the architecture while a bug, measurement artifact, budget effect, or untested existing mechanism remains a viable explanation.

## REQUIRED CHANGE

This section is mandatory and must match the machine-readable metadata.

### change_id

Unique stable identifier, for example:

- `R001-T1` telemetry
- `R002-B1` bug fix
- `R003-P1` parameter intervention
- `R004-M1` mechanism intervention
- `R005-A1` architecture change

### change_type

One of:

- `NONE`
- `TELEMETRY`
- `BUG_FIX`
- `PARAMETER_INTERVENTION`
- `MECHANISM_INTERVENTION`
- `ARCHITECTURE_CHANGE`

### target_files

Exact repository paths.

### target_functions

Exact functions/classes affected.

### change

Explicit behavior to add, remove, or modify. No vague implementation categories.

### must_not_change

Explicitly list mechanisms, parameters, or behavior that must remain fixed so the intervention is interpretable.

## Next experiment

Specify exactly:

- game preset
- steps per game
- actor/process settings if relevant
- memory policy
- control/baseline relationship
- single intervention being tested

Default research budget when no stronger reason exists:

```text
research_1
20,000 steps/game
12 games
240,000 new interaction steps
memory_policy=REUSE
```

## Preregistered predictions

Before implementation/run, specify:

- primary metric;
- current/baseline value when known;
- predicted direction/value;
- minimum meaningful effect;
- secondary metrics;
- metrics expected to remain unchanged.

## Falsifier

State the exact result that would reject the proposed explanation.

A prediction without a falsifier is not sufficient for causal research.

## Decision rule after next run

Use explicit rules, for example:

```text
IF eligible M2 opportunities are high but emitted M2 remains near zero without budget saturation
→ M2 mechanism failure becomes likely.

IF eligible opportunities are near zero
→ H05 remains untested; investigate upstream representation/normalization.

IF eligible candidates exist but disappear between proposal and commit
→ implementation bug becomes likely.
```

## Required outputs from next experiment

List the exact metrics that must appear in the next `EXPERIMENT_EVIDENCE.md` for the decision rule to be evaluable.
