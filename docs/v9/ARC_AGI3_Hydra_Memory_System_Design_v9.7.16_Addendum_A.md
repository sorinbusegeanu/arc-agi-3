# ARC-AGI-3 Hydra Memory System Design v9.7.16 --- Addendum A

## Efficient Matched Evaluation and Telemetry Reduction

**Status:** Design addendum\
**Applies to:** ARC-AGI-3 Hydra Memory System Design v9.7.16\
**Purpose:** Reduce H19 evaluation cost and telemetry noise without
weakening causal evaluation, reproducibility, or diagnostic coverage.

------------------------------------------------------------------------

# A1. H19 matched evaluation cadence

The full `broad` developmental workload remains the primary
heterogeneous training distribution.

Matched H19 evaluation is separated from ordinary developmental
sampling.

## A1.1 Normal developmental epochs

Normal epochs execute:

``` text
broad sampling once
→ publication once
→ ingestion once
→ developmental memory processing
→ HGT training
→ model candidate creation
```

A normal developmental epoch does not repeat the complete `broad`
workload solely to compare HGT-on against the parent/Hydra condition.

## A1.2 H19 evaluation epochs

H19 evaluation runs periodically and uses the existing fixed
matched-evaluation contract:

``` text
capture identical evaluation state
→ execute fixed TrialManifest with HGT-on
→ restore identical evaluation state
→ execute the same TrialManifest with comparison condition
→ compare behavioral outcomes
→ record H19 evidence
```

Default cadence:

``` text
every 3 developmental epochs
```

The cadence is configuration-controlled and recorded in the
`ExperimentManifest`.

A final H19 evaluation is also executed before declaring a training run
complete.

## A1.3 Reduced fixed TrialManifest

H19 evaluation does not replay the complete `broad` sampling workload.

It uses a fixed representative `TrialManifest` drawn from the
environment families already present in `broad`.

The manifest must preserve:

``` text
environment-family diversity
previously solved and unsolved tasks
cross-family transfer opportunities
different action-space and horizon regimes
different memory-development requirements
fixed start/reset states
fixed seeds
fixed per-trial horizons
fixed trial ordering
```

The same manifest is reused across evaluation epochs unless the
experiment definition explicitly changes.

The evaluation budget is independent of adaptive developmental sampling
budgets.

Default target:

``` text
10–20% of one broad developmental epoch
```

The exact trial count is derived once when the experiment is created and
becomes part of the immutable experiment definition.

## A1.4 Scientific validity

Reduced evaluation changes computational volume, not the H19 causal
contract.

Every matched comparison still requires:

``` text
identical starting state
identical ModelVersion baseline
identical TrialManifest
identical seeds
identical horizons
identical interaction opportunities
identical environment ordering
state restoration between conditions
```

Early termination discards unused horizon. It never grants an additional
reset or replacement trial.

Publication and ingestion occur for both matched branches because
downstream Hydra state and HGT-dependent behavior are part of the
intervention being evaluated. The reduced `TrialManifest` bounds this
duplicated cost.

The matched evaluation result must not alter the amount of developmental
interaction received by the normal `broad` training path.

------------------------------------------------------------------------

# A2. Evaluation scheduling

The runtime maintains two counters:

``` text
developmental_epoch
h19_evaluation_id
```

They are independent.

Example:

``` text
Development epoch 1  → broad training
Development epoch 2  → broad training
Development epoch 3  → broad training → H19 evaluation 1
Development epoch 4  → broad training
Development epoch 5  → broad training
Development epoch 6  → broad training → H19 evaluation 2
```

An H19 evaluation may also be triggered by:

``` text
candidate model promotion boundary
explicit scientific evaluation request
final run-completion validation
```

Such evaluations use the same fixed `TrialManifest`.

------------------------------------------------------------------------

# A3. Telemetry logging frequency

Dashboard rendering and persistent telemetry logging have separate
frequencies.

## A3.1 Dashboard refresh

Interactive dashboard refresh may remain relatively frequent for
operational visibility:

``` text
5 seconds
```

Dashboard refresh does not imply persistent log writes.

## A3.2 Persistent runtime telemetry

`telemetry/dashboard_metrics.jsonl` is persisted at:

``` text
30-second interval
```

This is the default periodic frequency.

A snapshot is also persisted immediately on significant events:

``` text
developmental epoch start
developmental epoch completion
matched branch completion
H19 comparison completion
transfer-validation completion
HGT training completion
model promotion or rejection
snapshot/recovery event
memory-governor warning or hard-limit event
runtime failure
```

Event-triggered records include an `event_type`.

Periodic records use:

``` text
event_type = "PERIODIC"
```

## A3.3 Change suppression

Periodic persistence may suppress a record when all primary scientific
metrics and all material operational metrics remain unchanged within
their configured tolerances.

Event-triggered records are never suppressed.

High-frequency performance debugging remains available through an
explicit diagnostic telemetry mode. Diagnostic mode is not the default
scientific-run configuration.

------------------------------------------------------------------------

# A4. Primary dashboard metric limit

The primary dashboard displays a maximum of **24 metrics**.

Detailed telemetry remains available in persisted telemetry and
diagnostic views. The 24-metric limit affects the primary human-facing
dashboard only.

## A4.1 Primary scientific metrics

The dashboard displays these 16 scientific/developmental metrics:

1.  `behavioral_success_rate`
2.  `current_run_wins`
3.  `current_run_levels_solved`
4.  `hgt_behavioral_gain`
5.  `hgt_selected_branch`
6.  `cross_family_transfer`
7.  `false_transfer_rate`
8.  `historical_retention`
9.  `M0_count`
10. `M1_count`
11. `M2_count`
12. `M3_count`
13. `M4_validated`
14. `M5_count`
15. `M6_count`
16. `M7_count`

## A4.2 Primary operational metrics

The dashboard displays these 8 operational metrics:

17. `sampled_steps`
18. `ingested_steps`
19. `sampling_backlog`
20. `sampling_rate`
21. `ingestion_rate`
22. `canonical_apply_latency`
23. `process_rss_gb`
24. `gpu_memory_used_gb`

Metric naming may use the existing canonical telemetry key where the
implementation already exposes an equivalent value.

## A4.3 Detailed metrics

Metrics removed from the primary dashboard remain available through
detailed telemetry, including:

``` text
queue depths
worker counts
individual latency percentiles
publication batch statistics
SHM utilization
WAL statistics
derivation counters
HGT batch statistics
training and validation losses
GPU allocator details
lifecycle counters
symbol-grounding submetrics
environment-viability diagnostics
per-game statistics
individual transfer counters
```

`game_results.log` remains authoritative for per-game behavioral
progression.

`environment_viability.log` remains authoritative for per-environment
viability history.

`dashboard_metrics.jsonl` remains the compact time-series source for
run-level scientific and operational assessment.

------------------------------------------------------------------------

# A5. Training continuation and stopping assessment

Periodic H19 evaluation supplies the principal causal signal for whether
additional HGT training is improving behavior.

Training continuation is assessed from trends across completed
evaluation points rather than high-frequency telemetry samples.

Primary assessment signals are:

``` text
hgt_behavioral_gain
behavioral_success_rate
current_run_wins
current_run_levels_solved
cross_family_transfer
false_transfer_rate
historical_retention
M4_validated / M5 / M6 / M7 development
```

A training run is not considered plateaued from repeated periodic
dashboard samples inside one sampling branch.

Plateau assessment operates across completed developmental epochs and
H19 evaluations.

------------------------------------------------------------------------

# A6. Acceptance criteria additions

Add the following acceptance criteria to v9.7.16:

129. Normal developmental epochs execute the `broad` workload once;
     complete broad HGT-on/HGT-off replay is not required every epoch.
130. H19 matched evaluation uses a fixed immutable `TrialManifest`
     independent of adaptive developmental sampling budgets.
131. The default H19 evaluation budget is bounded to 10--20% of a broad
     developmental epoch unless an experiment explicitly declares
     another budget.
132. H19 evaluation occurs periodically, default every three
     developmental epochs, and once before run-completion validation.
133. Both matched H19 branches preserve identical starting state, seeds,
     horizons, ordering, and interaction opportunities.
134. Publication and ingestion remain part of both H19 branches while
     their duplicated cost is bounded by the reduced `TrialManifest`.
135. Dashboard rendering and persistent telemetry logging use
     independent frequencies.
136. `dashboard_metrics.jsonl` defaults to a 30-second periodic
     persistence interval plus mandatory significant-event records.
137. Event-triggered scientific records are never suppressed by
     telemetry deduplication.
138. The primary dashboard contains no more than 24 metrics.
139. Detailed operational and scientific metrics remain available
     outside the primary dashboard.
140. Plateau and training-stop analysis uses completed epoch/evaluation
     points rather than high-frequency intra-branch telemetry samples.

------------------------------------------------------------------------

# A7. Resulting runtime structure

``` text
                 DEVELOPMENT

Broad heterogeneous sampling
          │
          ▼
 publish → ingest → Hydra development
          │
          ▼
      HGT training
          │
          ├───────────────┐
          │               │ every N epochs
          ▼               ▼
 next broad epoch    fixed H19 TrialManifest
                          │
                    ┌─────┴─────┐
                    ▼           ▼
                  HGT-on      comparison
                    │           │
                    └─────┬─────┘
                          ▼
                 causal H19 result


                 TELEMETRY

runtime state ──5 s──► interactive dashboard
      │
      ├──30 s────────► dashboard_metrics.jsonl
      │
      └──event────────► immediate durable telemetry

Primary dashboard: 24 metrics maximum
Detailed telemetry: retained for diagnostics and research analysis
```

This addendum preserves the v9.7.16 scientific controls while reducing
duplicated sampling/publication/ingestion work and making the dashboard
time series substantially easier to interpret.
