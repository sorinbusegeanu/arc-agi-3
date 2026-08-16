# ARC-AGI-3 v8 — Continuous Developmental Memory Runtime

v8 is a RAM-authoritative continuous developmental runtime. It does not change v7 behavior.

## Core execution model

- RAM is authoritative while a run is active.
- Sampling, M0→M7 derivation, canonical reduction, graph updates, and developmental peer operators run continuously.
- Canonical memory is partitioned across RAM shard writers by deterministic 128-bit `MemoryUid`.
- Hot-path messages are fixed-width binary shared-memory records; ordinary cognition does not require JSON, pickle, SQLite, or filesystem writes.
- Actors read bounded-staleness shared-memory cognition indexes plus an actor-local recent overlay.
- M6 outcome and M7 strategy memory participates directly in action selection; M1 remains the fallback.
- Disk persistence is an asynchronous content-addressed recovery observer.
- Controlled shutdown drains development, freezes evidence/reporting, compacts dependency-safe retired RAM, writes a complete recovery snapshot, verifies it, then writes `RUN_COMPLETE.json`.

## Memory hierarchy

```text
M0 Episodes
M1 Contingencies
M2 Transformation Families
M3 Carrier hypotheses / contextual roles / functional roles
M4 Concept candidates and validated concepts
M5 Consequence structures / world-model components
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
GAME_PROVENANCE
DEPENDS_ON
ENABLES
BLOCKS
OUTCOME_EQUIVALENT
```

`GAME_PROVENANCE` stores the exact source-game hash. Higher abstractions inherit exact provenance through graph lineage rather than relying on the compact 64-bit game mask.

## Developmental peer systems

The shared graph is continuously inspected by independent analyses which emit canonical mutation proposals rather than mutating shard arenas directly:

```text
prediction-violation estimation
context refinement
carrier → role formation
bounded typed graph-neighborhood similarity
bounded future-option estimation
ISF attention / replay prioritization
explanatory reach / compression
held-out transfer candidacy and automatic matched intervention
concept validation
M5 world-model integration
M6 outcome-equivalence merge/split/refinement
M7 strategy reliability / cost / alternatives
target-like preference estimation
planning / replanning / strategy-ablation probes
lifecycle / quarantine / reactivation / retirement
adaptive resource control
scientific evidence and H01-H15 reporting
```

The heavy read-only peer analyses run concurrently against one published graph view; canonical mutation remains shard-owned.

### Prediction

Actors compare each observed transition against the M1 outcome distribution visible **before** that transition. Prediction error therefore has causal temporal ordering. It remains scientifically inactive until the corresponding context/action contingency has sufficient support and stability. Contradictions drive context refinement rather than immediate concept replacement.

### ISF and replay

A capability-derived developmental `Stage_t` is inferred from the published graph before each peer interval and held fixed while that interval is scored. The ISF combines bounded option-structure impact, prediction error, prospective learning value, transfer prior, explanatory potential, and future-option value. A bounded replay/attention scheduler prioritizes violations, transfer opportunities, explanatory opportunities, future-option effects, and high-value developmental memories instead of processing only by raw support. Evidence created during the interval can affect only the next inferred stage.

### Bounded graph-neighborhood similarity

M3 role/contextual-role and M4 concept candidates receive deterministic radius-1 typed neighborhood descriptors. Candidate generation uses coarse structural buckets and hard limits; exact comparison is restricted to at most the configured candidate budget and unchanged descriptor versions are skipped.

High similarity never merges canonical identity. It produces only `SIMILAR_TO` evidence and a prospective transfer prior. A later held-out intervention is still required for empirical transfer or concept validation.

### Future options

The actor's immediate available-action change remains early `OE0` evidence. The background estimator separately computes bounded learned `FO^k` over the discovered M1 context-transition graph. Cache invalidation is local to changed nodes and their bounded reverse dependency neighborhood rather than clearing the entire cache.

### Transfer and concepts

Structural cross-game recurrence or bounded graph correspondence creates transfer candidates only. It cannot validate a concept. `continuous-run` automatically schedules bounded matched held-out experiments when a candidate has an admissible target game:

```text
same target game
same seed
same interaction budget
memory-on lineage restricted to the candidate
vs
memory-off M1 fallback
```

Positive held-out effects can validate transfer/concepts. Failed interventions are retained as negative scientific evidence.

### Outcomes, strategies, replanning and preference

M6 first retains bounded fine consequence variants. The outcome peer can merge recurring variants into a persistent coarse outcome class and rebind M7 alternatives to that class. If a coarse class fails validation it is quarantined, its `SUPERSEDES` effect stops participating in cognition, and its persistent fine members become usable again. M7 strategies pointing to an inactive coarse outcome are not selectable.

M7 stores observed attempts, success/reach statistics, and cost separately from outcome identity. Actor planning ranks admissible strategies by learned reliability and efficiency, with preference as a separate bonus. `QUARANTINED`, `RETIRE_PENDING`, and `RETIRED` strategies cannot guide action.

Actors can run explicit same-outcome strategy-ablation probes. H14 requires successful recovery after the primary strategy was deliberately excluded; an incidental strategy switch is only partial evidence.

Preference uses only clean comparisons where both outcomes were represented as reachable before choice and no existing preference influenced the choice. This prevents preference → choice → preference self-validation. Terminal `WIN` does not directly create preference.

## Incremental development

Every evidence contribution reaches its canonical owner, while downstream development uses persistent dirty accumulation. Repeated changes to one developmental path are coalesced into one notification carrying exact multiplicity, so support is not lost and raw events are not redundantly replayed through every higher stage.

```text
last_processed_version
latest_required_version
queued_bit
accumulated_multiplicity
```

## Lifecycle, forgetting and RAM recovery

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

Lifecycle fitness consumes the same fixed developmental Stageₜ weighting state used by attention/replay. Dependency-safe `RETIRE_PENDING` memories become `RETIRED`; retired memories are excluded from cognition immediately.

When node/edge pressure is high, or at final maintenance, v8 enters a short quiescent generation barrier, archives retired nodes and incident edges to:

```text
archive/retired_memory.jsonl
```

then rewrites the shared arenas densely and restarts shard writers so their local indexes are rebuilt. This physically reclaims bounded RAM while preserving retired scientific/provenance records outside the live cognitive graph.

## Adaptive resource control

The controller observes queue pressure and arena occupancy and adjusts:

```text
actor throttle
peer cadence
peer candidate budget
```

At high arena pressure it can also trigger dependency-safe retired-memory compaction. Full-ring pressure remains backpressure rather than a fatal one-second timeout.

## Scientific reporting

Scientific evidence is append-only, causally timestamped, restart-persistent, and associated with graph generation. H01-H15 evaluators enforce per-contract requirements including:

```text
required evidence kinds
minimum records
quality / normalization
causal intervention when required
held-out provenance when required
positive or negative effect direction
distinct target games
dependency gates
```

Each decision reports separately:

```text
raw_decision
quality_gate
dependency_gate
final_decision
evidence_count
blocker
```

Proxy/structural evidence may yield `PARTIALLY_VALID`; it never substitutes for required held-out transfer, causal replanning recovery, or stable clean preference evidence.

A reporting cut records:

```text
watermark
graph_generation
graph_digest
causally available evidence
H01-H15 decisions
```

The console hypothesis line is sourced from the current scientific evidence ledger rather than benchmark wins.

## Consistent persistence and restart

Ordinary learning does not wait for disk chunk serialization.

For a periodic recovery cut, actors are briefly frozen, canonical queues and developmental peers reach a fixed point, and the snapshot process copies all shard arenas into its own immutable RAM payloads. Actors resume as soon as that coherent capture is complete; content-addressed disk chunking continues asynchronously.

Recovery snapshots use 4 MiB content-addressed chunks. Unchanged chunks are reused across snapshots. Snapshot auxiliary state includes the scientific evidence ledger, transfer trials, preference probes, lifecycle state, similarity descriptor versions, peer dedupe/version state, and graph generation. Existing v8 node snapshots are migrated into the current RAM record shape on load while canonical v8 `MemoryUid` identity remains stable.

```text
RAM evolution ───────────────► cognition
      │
      └─ generation barrier ─► immutable RAM capture ─► async chunk persistence
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

Automatic transfer experiments are enabled by default and bounded by:

```text
--transfer-experiment-steps 32
--max-transfer-experiments 8
```

They can be disabled with `--no-automatic-experiments`.

Every progress interval, a dedicated reporting process prints game progress and
H01-H15 lines independently of actor-learning queue load. The first report is
emitted after the first interval (60 seconds by default), not at startup. For example:

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

## Scientific condition

Implementation availability is not evidence availability. A particular run may correctly leave hypotheses `INSUFFICIENT_EVIDENCE` when no admissible held-out target, clean preference comparison, strategy-ablation recovery, or other required event occurred. That is an evidence result, not a missing runtime subsystem.
