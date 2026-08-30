# EXPERIMENT_EVIDENCE

`EXPERIMENT_EVIDENCE.md` is the factual output of one causal research run.

It is generated automatically under `<root>/research/EXPERIMENT_EVIDENCE.md` after every normal `continuous-run`.

Its purpose is to let the researcher determine whether an observed limitation is a bug, experimental artifact, performance bottleneck, mechanism failure, architectural limitation, hypothesis contradiction, or insufficient evidence.

It must not prescribe the next architecture or mechanism change.

## Research loop position

```text
RESEARCH_DECISION N
        ↓
explicit intervention
        ↓
EXPERIMENT N+1
        ↓
EXPERIMENT_EVIDENCE N+1
        ↓
causal analysis
        ↓
RESEARCH_DECISION N+1
```

## Persistent-memory rule

Memory is intentionally reused between runs.

Therefore cumulative memory/evidence totals are context only. Causal interpretation must primarily use:

- the pre-run start state;
- the post-run end state;
- experiment-local deltas;
- evidence appended during this run;
- current-run game activity;
- declared intervention metadata.

## Required structure

### Experiment identity

- `experiment_id`
- `parent_experiment_id`
- code revision
- exact command
- start/end timestamps
- exit code
- `memory_policy: REUSE`

### Applied intervention declaration

The run captures the machine-readable metadata from the existing `RESEARCH_DECISION.md` before execution.

If the declaration is absent or invalid, the evidence file must state that causal interpretation is limited.

### Start state

Captured before the experiment starts from the previous durable run state:

- memory watermark
- total memories
- edges
- evidence record count
- M0-M7 level counts
- formation telemetry
- verified-success state
- trajectory-optimizer state

### Experiment-local activity

Per game:

- steps
- wins
- failures
- levels completed
- resets

Also include automatic transfer experiments attempted/completed/passed.

### End state

Same state schema as the start state.

### Experiment deltas

At minimum:

- watermark delta
- memory-count delta
- edge-count delta
- evidence-count delta
- M0-M7 deltas

### Formation causal funnel

Production telemetry for:

```text
M1N
→ support eligible
→ candidate family groups
→ eligible M2 groups
→ M2 candidates emitted

M3 carriers
→ role groups
→ carrier-qualified groups
→ role candidates
```

Include production rejection counts and bounded rejection examples.

### Experiment-local evidence ledger

The run records the evidence-ledger byte boundary before execution. After the run, only rows appended beyond that boundary count as experiment-local evidence.

Report:

- evidence kinds
- causal interventions
- positive/negative/neutral effects
- hypothesis IDs when present
- distinct source games
- distinct target games

If the ledger was truncated or rewritten, explicitly mark the local slice unavailable rather than infer deltas.

### H01-H15 end status

Include end-of-run hypothesis status, but label it cumulative.

Do not attribute a cumulative status change to the current experiment unless experiment-local evidence supports that attribution.

### Integrity and possible confounders

Include:

- reporting cut
- optimizer state/backlog indicators
- adaptive-learning state
- whether the experiment-local ledger is available
- whether the intervention declaration was valid

### Cumulative state

Include cumulative memory and evidence only as context.

It must be explicitly marked as non-local causal evidence.

## Decision discipline

The researcher analysing this file must identify the earliest causal link supported as broken or unresolved and must not convert missing evidence into mechanism or hypothesis failure.
