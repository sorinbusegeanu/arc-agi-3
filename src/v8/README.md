# ARC-AGI-3 v8 — Continuous Developmental Memory Runtime

v8 is a RAM-authoritative continuous developmental runtime. It does not change v7 behavior.

## Core execution model

- RAM is authoritative while a run is active.
- Sampling, M0→M7 derivation, canonical reduction, graph updates, and developmental peer operators run continuously.
- Canonical memory is partitioned across RAM shard writers by deterministic 128-bit `MemoryUid`.
- Hot-path messages are fixed-width binary shared-memory records; no JSON, pickle, SQLite, or filesystem writes are required for ordinary cognition.
- Actors read bounded-staleness shared-memory cognition indexes plus an actor-local recent overlay.
- M6 outcome and M7 strategy memory now participates in action selection; M1 remains the fallback.
- Disk persistence is an asynchronous latest-wins binary recovery observer.
- Controlled shutdown drains development, freezes evidence/reporting, writes a complete recovery snapshot, verifies it, then writes `RUN_COMPLETE.json`.

## Memory hierarchy

```text
M0 Episodes
M1 Contingencies
M2 Transformation Families
M3 Carrier hypotheses / contextual roles / functional roles
M4 Concept candidates and validated concepts
M5 Consequence structures
M6 Outcome-equivalence abstractions
M7 Strategies
```

Cross-cutting relations include:

```text
PROVENANCE
EXPLAINS
CONTEXT_REFINES
SIMILAR_TO
TRANSFER_CORRESPONDENCE
SUPERSEDES
LEADS_TO
PREFERENCE
```

## Developmental peer systems

The shared graph is continuously inspected by independent operators which emit canonical mutation proposals rather than mutating shard arenas directly:

```text
prediction-violation estimation
context refinement
carrier → role formation
bounded future-option estimation
explanatory reach / compression
held-out transfer candidacy and trial validation
concept validation
M6 outcome-equivalence refinement
M7 strategy reliability / alternatives
target-like preference estimation
planning / replanning
lifecycle / quarantine / reactivation / retirement staging
adaptive resource control
scientific evidence and H01-H15 reporting
```

### Prediction

Prediction error is inactive until a context/action contingency has sufficient support and stability. Contradictions then drive context refinement rather than immediate concept replacement.

### Future options

The actor's immediate available-action change remains early `OE0` evidence. The background estimator separately computes bounded learned `FO^k` over the discovered M1 context-transition graph with a small horizon and cached local reachability.

### Transfer and concepts

Structural cross-game recurrence creates transfer candidates only. It cannot validate a concept. `TransferValidator` requires an explicit held-out `memory-on` versus `memory-off` trial before empirical transfer evidence or validated concept status can be produced.

### Outcomes, strategies, replanning and preference

M6 identity is consequence-oriented rather than trajectory-oriented. M7 stores strategies separately from outcome identity. Actors can therefore change strategies while retaining an outcome representation. Preference is inferred separately from repeated comparable outcome evidence and never directly from a `WIN` terminal label.

## Incremental development

Repeated raw experience is reduced canonically at every level. Downstream stage propagation is dirty-key coalesced per canonical UID so one burst of repeated evidence does not enqueue an equivalent M0→M7 derivation path for every raw event. `DirtyKeyTracker` provides the persistent invalidation contract:

```text
last_processed_version
latest_required_version
queued_bit
```

## Lifecycle

Higher-level memories use hysteretic lifecycle states:

```text
CANDIDATE
PROBATION
ACTIVE
VALIDATED
QUARANTINED
RETIRE_PENDING
RETIRED
REACTIVATED
```

Scientific provenance is preserved independently of cognitive retirement.

## Adaptive resource control

The controller observes queue pressure and arena occupancy and can throttle actor production and reduce peer cadence/breadth before bounded RAM structures saturate. Full-ring pressure remains backpressure, not a fatal one-second timeout.

## Scientific reporting

Scientific evidence is append-only and causally timestamped. H01-H15 evaluators consume one immutable evidence cut associated with a graph digest. Each decision reports:

```text
raw_decision
quality_gate
dependency_gate
final_decision
evidence_count
blocker
```

Proxy/structural evidence may yield `PARTIALLY_VALID`; it never substitutes for a required held-out transfer, replanning, or stable preference contract.

The console hypothesis line is sourced from the current scientific evidence cut rather than from benchmark wins.

## Persistence

Ordinary learning never waits for disk.

Recovery snapshots use content-addressed 4 MiB chunks. Unchanged chunks are reused across snapshots, reducing write amplification. v8 can restore the previous v1 node snapshot representation and migrates it into the current RAM record shape on load.

```text
RAM evolution ───────────────► cognition
      │
      └──── async latest-wins snapshot ─► content-addressed recovery chunks
```

## Continuous ARC run

```bash
PYTHONPATH=src python -m v8 continuous-run \
  --root runs/v8/continuous \
  --games diverse \
  --steps-per-game 1000 \
  --actors 8 \
  --shards 4 \
  --stage-workers 2
```

Every progress interval prints dedicated game progress and H01-H15 lines, for example:

```text
[19:45] wins=50.0% levels_solved=50.0% solved_games=5/10 (ez01, ez02, ez03, ez04, ez05)
[19:45] hypotheses H01=VALID ... H15=PARTIALLY_VALID
```

At startup the graph source and loaded node count are also shown.

## Synthetic smoke run

```bash
PYTHONPATH=src python -m v8 smoke \
  --root runs/v8/smoke \
  --events 10000 \
  --shards 4 \
  --stage-workers 2
```

## Remaining scientific condition

Implementation availability is not the same as evidence availability. In a particular run, hypotheses that require held-out interventions, observed strategy ablation/recovery, or stable preference comparisons may correctly remain `INSUFFICIENT_EVIDENCE` until those opportunities occur. That is an evidence result, not a missing runtime subsystem.
