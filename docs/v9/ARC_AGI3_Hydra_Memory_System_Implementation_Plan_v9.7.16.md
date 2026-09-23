# ARC-AGI-3 Hydra Memory System Implementation Plan v9.7.16

**Target design:** `ARC_AGI3_Hydra_Memory_System_Design_v9.7.16.md`  
**Research contract:** `Research_problem_statement_v070.md`  
**Target codebase:** `src/v9` on `main`  
**Implementation style:** incremental cutover; existing runtime remains executable after every phase  
**Compatibility rule:** modify authoritative v9 modules; do not create runtime `*_v2.py` replacements  
**Test scope:** v9 only

---

# 1. Objective

Implement v9.7.16 without a big-bang rewrite.

The implementation must end with one common canonical memory substrate supporting two explicitly different scientific modes:

```text
ASYNC_DEVELOPMENT
    H17
    independent Hydra developmental processes
    atomic CanonicalStateHandle publication
    no schedule-identical trajectory guarantee
    learned HGT→Hydra developmental feedback disabled by default

MATCHED_REASONING
    H19
    immutable EpochInferenceView
    fixed TrialManifest
    EvidenceCut
    DevelopmentalCut
    TrainingCut
    PolicyProjectionCut
    deterministic actor-visible state
```

The implementation must preserve:

```text
M0-M7 developmental semantics
causal provenance
semantic abstention
existing HGT reasoning behavior
bounded runtime memory
crash/restart
current continuous-run CLI compatibility
```

---

# 2. Current-code starting point

The current `main` already contains useful foundations that should be retained and refactored rather than replaced:

```text
src/v9/runtime/canonical_commit.py
src/v9/runtime/canonical_commit_state.py
src/v9/runtime/canonical_commit_derivation.py
src/v9/runtime/bounded_indexes.py
src/v9/runtime/read_view.py
src/v9/runtime/chunked_snapshot.py
src/v9/runtime/snapshot_backend.py
src/v9/runtime/residency.py
src/v9/runtime/memory_governor.py
src/v9/runtime/publication_throughput.py
src/v9/runtime/shared_batch_transport.py
src/v9/runtime/parallel_memory_coordinator.py
src/v9/runtime/multiprocess.py
src/v9/runtime/epoch_runner.py
src/v9/runtime/environment_viability.py
src/v9/runtime/transfer_validation.py
src/v9/runtime/lifecycle.py
src/v9/hgt/epoch_dataset.py
src/v9/hgt/training.py
src/v9/hgt/matched_evaluation.py
```

Important current deltas against v9.7.16:

```text
ScientificConfig.design_version is still 9.7.9.

Actor policy refresh is step/time based.

Game budgets adapt from previous results.

HGT data is still epoch JSONL written directly from sampled transitions.

HGT training still has time/trigger-oriented configuration.

shared_batch_transport.py allocates individual shared-memory payloads.

Canonical state is still primarily mutable runtime state.

No CanonicalCommitWAL exists.

No immutable CanonicalStateHandle exists.

No authoritative ASYNC_DEVELOPMENT / MATCHED_REASONING split exists.

No deterministic DevelopmentalCut exists.

No TrialManifest, StructuralPriorProfile, GroundingCondition, or F1-F18 registry exists.
```

Before implementation, correct two documentation-only defects in the design source:

```text
Design predecessors must list v9.7.15 rather than v9.7.16 as predecessor.
Section 31 must say "v9.7.16 is operational when".
```

---

# 3. Implementation strategy

Use four implementation layers:

```text
Layer A — identity, persistence, atomic canonical state
Layer B — bounded high-throughput runtime transport
Layer C — scientific modes and deterministic visibility
Layer D — research controls, falsification, long-run validation
```

Do not implement H16/H18/H19 experiment machinery on top of mutable, non-WAL canonical state.  
Do not replace the actor pipeline before canonical recovery semantics are established.  
Do not switch HGT training to the new evidence authority before WAL evidence materialization works.

---

# 4. Phase 0 — Baseline and migration guardrails

## Goal

Freeze current behavior so each later phase can prove that it changes only the intended contract.

## Files

```text
src/v9/runtime/config.py
src/v9/runtime/runtime_integrity.py
src/v9/tests/
```

## Work

1. Set the new target design identifier in a feature-gated form, without changing runtime semantics yet.
2. Add a `v9.7.16 migration status` structure that records which architectural capabilities are active.
3. Capture baseline runtime fixtures for:
   - canonical M0-M7 counts;
   - existing batch publication;
   - actor policy snapshots;
   - HGT epoch training;
   - snapshot/restart;
   - lifecycle;
   - transfer validation;
   - memory-governor behavior.
4. Add an acceptance-matrix file mapping v9.7.16 acceptance criteria 1-128 to:
   - test;
   - runtime smoke;
   - long-run experiment;
   - not yet implemented.
5. Preserve the current `continuous-run` call without requiring additional CLI arguments.

## Tests

Extend existing:

```text
test_v978_runtime_acceptance.py
test_runtime_integrity_repairs.py
test_multiprocess_runtime.py
test_persistence_research_runtime.py
```

Add:

```text
test_v9716_acceptance_matrix.py
```

## Exit gate

Current v9 behavior is unchanged and the complete existing v9 test suite passes.

---

# 5. Phase 1 — Scientific identity and dual-mode contracts

## Goal

Introduce the scientific control plane before changing storage or runtime scheduling.

## New files

```text
src/v9/runtime/scientific_modes.py
src/v9/research/__init__.py
src/v9/research/experiment_manifest.py
```

## Modify

```text
src/v9/runtime/config.py
src/v9/runtime/epoch_runner.py
src/v9/runtime/multiprocess.py
src/v9/runtime/runtime.py
src/v9/__main__.py
```

## Core types

Implement:

```text
ScientificVisibilityMode
    ASYNC_DEVELOPMENT
    MATCHED_REASONING

LearnedDevelopmentalFeedbackProfile
    DISABLED
    ENABLED

ExperimentManifest
ExperimentManifestId
GroundingCondition
    C0_INTERACTION_ONLY
    C1_SYMBOLS_ONLY
    C2_ALIGNED
    C3_SHUFFLED

StructuralPriorProfile
ReasoningCondition
TrialSpec
TrialManifest
InteractionOpportunityManifest
```

## Deterministic identities

Implement deterministic derivation for:

```text
experiment_id
replicate_id
sampling_epoch_id
stable_environment_job_id
producer_id
environment_instance_id
episode_id
ScientificEvidenceId
```

Process-only identity remains separate:

```text
process_run_epoch
worker_id
local_lease_counter
```

## Compatibility

Default ordinary `continuous-run` behavior maps to:

```text
ScientificVisibilityMode.ASYNC_DEVELOPMENT
LearnedDevelopmentalFeedbackProfile.DISABLED
```

H19 experiment manifests explicitly select `MATCHED_REASONING`.

## Tests

Add:

```text
test_scientific_modes.py
test_experiment_manifest.py
test_scientific_identity_determinism.py
```

Required assertions:

```text
process restart does not change scientific ids
worker scheduling does not change scientific ids
same manifest produces same TrialManifest checksum
process_run_epoch never enters scientific ids
```

## Exit gate

Scientific identity and mode selection are fully deterministic but runtime behavior remains otherwise equivalent to current `main`.

---

# 6. Phase 2 — Immutable canonical store foundation

## Goal

Replace the assumption of one large mutable Python memory state with bounded immutable chunks and atomic root handles.

This is the highest-risk phase and must be completed before WAL cutover.

## New files

```text
src/v9/runtime/canonical_store.py
src/v9/runtime/canonical_transaction.py
```

## Modify

```text
src/v9/runtime/runtime.py
src/v9/runtime/canonical_commit.py
src/v9/runtime/canonical_commit_state.py
src/v9/runtime/canonical_commit_derivation.py
src/v9/runtime/read_view.py
src/v9/runtime/bounded_indexes.py
src/v9/runtime/residency.py
src/v9/runtime/snapshot_chunks.py
```

## Implement

### `CanonicalStateHandle`

```text
canonical_applied_lsn
graph_root
payload_root
level_index_root
signature_index_root
grounding_index_root
provenance_index_root
lifecycle_index_root
policy_source_index_root
schema_versions
generation
checksum
```

### Immutable chunk store

Initial chunk types:

```text
GraphChunk
PayloadChunk
LevelIndexChunk
SignatureIndexChunk
GroundingIndexChunk
ProvenanceChunk
LifecycleChunk
```

Bounds:

```text
entries/chunk <= 8,192
encoded bytes/chunk <= 64 MiB
```

### `TransactionOverlay`

Overlay contains only changed data:

```text
insert/update/delete maps
edge deltas
index posting deltas
tombstones
lifecycle changes
new chunk builders
```

Reads resolve:

```text
overlay
→ base CanonicalStateHandle
```

### Atomic publication

Replace separate mutable-root/sequence visibility with:

```text
new_handle = finalize_overlay(...)
publish_current_handle(new_handle)
```

Readers pin one handle and cannot mix roots from generations.

### Reclamation

Implement:

```text
handle pin/release
chunk reachability
bounded orphan queue
bounded GC cursor
```

## Migration method

Do not switch all memory types in one commit.

Implement in this order:

```text
1. handle abstraction around current authoritative roots
2. immutable low-level chunk backend
3. authoritative index roots
4. M0/M1 writes
5. M2-M4 derivation writes
6. M5-M7/lifecycle writes
7. remove direct mutable-root publication paths
```

During steps 1-6, compare old and new read results in shadow assertions.

## Tests

Add:

```text
test_canonical_store.py
test_canonical_handle_atomicity.py
test_transaction_overlay.py
test_canonical_chunk_gc.py
test_canonical_structural_sharing.py
```

Extend:

```text
test_canonical_batch_apply.py
test_canonical_scaling.py
test_mutation_publication_lineage.py
test_memory_development.py
```

Required tests:

```text
unchanged chunks retain identical ChunkIds
one transaction does not clone entire graph
reader never sees mixed-generation roots
old handle remains readable while pinned
GC never deletes pinned chunks
overlay abort leaves current handle unchanged
```

## Exit gate

All canonical runtime reads can execute through `CanonicalStateHandle`, and no authoritative commit requires cloning the full resident graph.

---

# 7. Phase 3 — Canonical WAL and recovery frontiers

## Goal

Make WAL the redo authority and separate durability from live canonical visibility.

## New file

```text
src/v9/runtime/canonical_wal.py
```

## Modify

```text
src/v9/runtime/memory_pipeline.py
src/v9/runtime/canonical_commit.py
src/v9/runtime/canonical_transaction.py
src/v9/runtime/parallel_memory_coordinator.py
src/v9/runtime/runtime.py
src/v9/runtime/snapshot.py
src/v9/runtime/chunked_snapshot.py
src/v9/runtime/snapshot_backend.py
```

## Implement WAL format

```text
WalGroupHeader
CanonicalCommitFrame[]
WalGroupCommitFooter
```

Each transaction carries:

```text
wal_lsn
wal_tx_id
previous_lsn
scientific identity/provenance
producer causal ranges
deterministic mutation intent
TrainingEvidenceRecord payloads
work/byte metadata
checksums
```

## Frontiers

Implement separately:

```text
wal_durable_lsn
canonical_applied_lsn
snapshot_applied_lsn
hgt_checkpoint_lsn
```

Invariant:

```text
wal_durable_lsn >= canonical_applied_lsn >= snapshot_applied_lsn
hgt_checkpoint_lsn <= wal_durable_lsn
```

## Commit path

```text
compiled transaction
→ WAL append
→ complete group footer
→ fsync
→ wal_durable_lsn advance
→ TransactionOverlay
→ finalize immutable roots
→ atomic CanonicalStateHandle publication
```

## Crash recovery

Implement startup scan:

```text
validate framing
truncate torn/invalid tail
restore snapshot handle
replay snapshot_applied_lsn+1 ... wal_durable_lsn
```

## Tests

Add:

```text
test_canonical_wal.py
test_wal_group_commit.py
test_wal_torn_tail.py
test_wal_replay_idempotence.py
test_persistence_frontiers.py
test_post_wal_pre_publish_recovery.py
```

Crash-injection cases:

```text
mid frame
before footer
after footer before fsync
after fsync before overlay
mid overlay
after overlay before root publication
after root publication
```

## Exit gate

Restart from snapshot + WAL reproduces the exact canonical scientific state and sequence frontiers.

---

# 8. Phase 4 — Persistent SignatureIndex and bounded derivation scheduling

## Goal

Remove unbounded in-memory signature/dirty state and eliminate derivation head-of-line blocking.

## New files

```text
src/v9/runtime/signature_index.py
src/v9/runtime/derivation_merge.py
```

## Modify

```text
src/v9/runtime/canonical_commit_derivation.py
src/v9/runtime/derivation_publication.py
src/v9/runtime/parallel_memory_coordinator.py
src/v9/runtime/publication_throughput.py
src/v9/runtime/runtime.py
```

## `SignatureIndexStore`

Persist:

```text
support
contradiction support
last derived support
derivation_dirty
priority data
persistent scan cursor
```

Resident state contains only:

```text
bounded delta
bounded page cache
bounded dirty priority window
saved refill cursor
```

## Derivation leases

Task identity:

```text
(signature, target_support, derivation_schema_version)
```

Lease fields:

```text
lease_epoch
attempt
deadline
```

Reference bounds:

```text
pending <= 4,096
in-flight <= 64 * workers * 4
completed rows <= 2,048
worker batch <= 64
```

## Merge algebra

Implement exact/idempotent merge for:

```text
evidence set union
integer/fixed-point sufficient statistics
monotonic maturity state
deterministic representative
stable canonical abstraction identity
```

Completion order cannot choose the result.

## Tests

Add:

```text
test_signature_index_store.py
test_signature_index_restart.py
test_derivation_leases.py
test_derivation_retry_idempotence.py
test_derivation_merge_algebra.py
test_derivation_schedule_permutation.py
```

Extend:

```text
test_derivation_publication_bound.py
test_canonical_pipeline_v2.py
```

## Exit gate

Derivation scheduling has a finite resident footprint, survives restart, and produces identical scientific abstractions under result-order permutations.

---

# 9. Phase 5 — Producer-affinity transport and shared-memory slab pools

## Goal

Move batching before multiprocessing serialization and establish finite byte-aware IPC.

## Modify

```text
src/v9/runtime/multiprocess.py
src/v9/runtime/shared_batch_transport.py
src/v9/runtime/parallel_memory_coordinator.py
src/v9/runtime/publication_throughput.py
src/v9/runtime/memory_pipeline.py
```

## Actor batching

Replace one-transition queue payloads with:

```text
TransitionBatchEnvelope
preferred rows = 32
max rows = 64
```

Flush on:

```text
full
episode boundary
actor completion
age bound
```

## Stable producer affinity

Shard only by stable producer/environment identity.

Do not use `producer_sequence` to choose a shard.

## Descriptor transport

Implement:

```text
TransportSlabPool
TransportSlabDescriptor
TransportBatchBundle
```

Actor serializes payload once into the slab.

Stage/shard/coordinator forward descriptors only.

Use fixed binary control messages rather than pickling encoded transition bytes at each hop.

## Producer causal admission

Per producer:

```text
next expected sequence
bounded gap window
held bytes
gap age
```

Protocol errors:

```text
duplicate/stale
gap timeout
overlapping ranges
bad checksum
```

## Shared-memory ownership

Track disjoint states:

```text
grant_free
worker_owned
coordinator_owned
```

Reference total:

```text
Hydra SHM <= 512 MiB
transport <= 128 MiB
compiled/derivation <= 384 MiB
```

## Tests

Add:

```text
test_transport_slab_pool.py
test_transport_descriptor_routing.py
test_producer_affinity.py
test_producer_causal_admission.py
test_shm_credit_ownership.py
test_shm_worker_crash_recovery.py
```

Extend:

```text
test_batched_publication_intake.py
test_publication_throughput.py
test_multiprocess_runtime.py
test_optimized_runtime_batching.py
```

## Exit gate

No high-volume transition payload is repeatedly pickled between actor/stage/shard/coordinator, and all IPC byte pools have enforced hard ceilings.

---

# 10. Phase 6 — Transaction work bounds and canonical latency control

## Goal

Make large canonical work finite without exposing partial transactions.

## Modify

```text
src/v9/runtime/memory_pipeline.py
src/v9/runtime/canonical_transaction.py
src/v9/runtime/canonical_commit.py
src/v9/runtime/publication_throughput.py
src/v9/runtime/config.py
```

## Transaction envelope

Estimate before WAL commit:

```text
rows
input bytes
materialized mutation bytes
write count
symbol count
derived relation count
grounding operations
work units
```

Reference limits:

```text
rows <= 1,024
materialized mutation <= 64 MiB
continuation mutation <= 16 MiB
```

## Continuation fragments

After WAL durability:

```text
private overlay
→ fragment 1
→ fragment 2
→ ...
→ final validation
→ atomic handle publication
```

Fragments target bounded execution/lock time but remain invisible until the full transaction commits.

## Oversized input

Explicit statuses:

```text
OVERSIZED_CANONICAL_TRANSACTION
OVERSIZED_CANONICAL_PRIMITIVE
```

Do not silently process unbounded work.

## Tests

Add:

```text
test_canonical_work_budget.py
test_canonical_continuation_fragments.py
test_canonical_oversized_quarantine.py
test_canonical_fragment_visibility.py
```

## Exit gate

Canonical p95/p99 latency is measurable and one oversized batch cannot monopolize the runtime invisibly.

---

# 11. Phase 7 — MATCHED_REASONING DevelopmentalCut

## Goal

Make every actor-visible developmental change deterministic under H19.

## New file

```text
src/v9/runtime/developmental_cut.py
```

## Modify

```text
src/v9/runtime/epoch_runner.py
src/v9/runtime/canonical_commit_derivation.py
src/v9/runtime/lifecycle.py
src/v9/runtime/transfer_validation.py
src/v9/runtime/runtime.py
src/v9/runtime/environment_viability.py
grounding/context modules used by runtime
```

## `DevelopmentalPipelineVersion`

Define a versioned ordered operator registry.

Include all actor-visible operators:

```text
M1 maturation
M2 family formation
carrier proposal/validation
role proposal/validation
context refinement
grounding maturation
transfer validation
M4 validation
future-option updates
M5 consequences
M6 outcome equivalence
M7 strategy/replanning
lifecycle
replay-allocation metadata
```

## Work selection

For each operator:

```text
eligible stable keys
→ deterministic priority tuple
→ stable sort
→ configured work-count budget
→ exact target evidence/support version
```

Every selected item ends as:

```text
APPLIED
NO_CHANGE
REJECTED_BY_CAUSAL_RULE
FAILED_DETERMINISTICALLY
QUARANTINED
```

## Gate background workers

When mode is `MATCHED_REASONING`:

```text
background developmental mutation cannot publish outside DevelopmentalCut
```

When mode is `ASYNC_DEVELOPMENT`:

```text
existing independent developmental publication remains active
```

## Tests

Add:

```text
test_developmental_cut.py
test_developmental_cut_operator_coverage.py
test_developmental_cut_schedule_independence.py
test_matched_reasoning_background_mutation_gate.py
```

Required invariant test:

Different worker completion orders produce the same post-cut `CanonicalStateHandle` checksum.

## Exit gate

All actor-visible developmental mutations are either inside `DevelopmentalCut` or explicitly deferred under MATCHED_REASONING.

---

# 12. Phase 8 — EpochInferenceView and bounded policy projection

## Goal

Freeze the complete actor-visible state for H19 without affecting H17.

## New files

```text
src/v9/runtime/epoch_inference_view.py
src/v9/runtime/policy_projection.py
```

## Modify

```text
src/v9/runtime/actor_policy.py
src/v9/runtime/actor_policy_cache.py
src/v9/runtime/runtime.py
src/v9/runtime/parallel_memory_coordinator.py
src/v9/runtime/multiprocess.py
src/v9/runtime/epoch_runner.py
```

## `EpochInferenceView`

Pin:

```text
ExperimentManifestId
sampling_epoch_id
CanonicalStateHandle
PolicyVersion
ModelVersion
stage_state
normalization_state
graph schema
feature schema
ScientificConfigId
checksum
```

## MATCHED_REASONING actor path

Remove time-based policy refresh semantics.

Actors use one view for the complete epoch/trial set.

## ASYNC_DEVELOPMENT actor path

May bind a newer complete `CanonicalStateHandle` according to existing/runtime-configured publication policy.

Do not apply the H19 epoch-freezing rule.

## Policy projection

Maintain bounded candidate projection:

```text
global action entries <= 524,288
strategy records <= 8,192
outcome records <= 8,192
target resident <= 256 MiB
hard <= 512 MiB
actor view <= 16 MiB
```

No graph-wide scan on the action path.

## Tests

Add:

```text
test_epoch_inference_view.py
test_epoch_view_pinning.py
test_policy_projection_bounds.py
test_policy_projection_no_graph_scan.py
test_async_mode_live_handle_binding.py
```

Extend:

```text
test_hgt_action_inference.py
test_curriculum_runtime.py
```

## Exit gate

H19 actors cannot observe mid-epoch memory/model/policy changes; H17 remains asynchronous.

---

# 13. Phase 9 — WAL-backed training evidence

## Goal

Replace epoch transition JSONL as the training authority.

## New file

```text
src/v9/runtime/training_evidence.py
```

## Modify

```text
src/v9/hgt/epoch_dataset.py
src/v9/hgt/training.py
src/v9/runtime/canonical_wal.py
src/v9/runtime/memory_pipeline.py
src/v9/runtime/lifecycle.py
src/v9/runtime/transfer_validation.py
grounding/deliberation evidence producers
```

## `TrainingEvidenceRecord`

Support evidence kinds:

```text
interaction
transition/consequence
memory formation
transfer
grounding
reasoning trace
delayed outcome
consolidation
derivation
invariance
strategy ranking
```

Every record includes:

```text
TrainingEvidenceId
scientific provenance
source canonical frontier
schema versions
label payload
quality/weight metadata
checksum
```

## Atomic materialization

Replace direct epoch JSONL authority with immutable:

```text
TrainingEvidenceSegment
TrainingEvidenceManifest
```

Materialization:

```text
WAL range
→ temp segment
→ validate
→ fsync
→ immutable rename
→ COW manifest
→ atomic manifest rename
→ hgt_checkpoint_lsn
```

Keep `epoch_dataset.py` initially as a compatibility reader over the new manifest; remove direct append authority only after training has cut over.

## Tests

Add:

```text
test_training_evidence_records.py
test_training_evidence_segments.py
test_hgt_materialization_atomicity.py
test_hgt_checkpoint_lsn.py
test_training_evidence_restart.py
```

## Exit gate

Deleting/rebuilding replay/index caches does not change the durable HGT training example set.

---

# 14. Phase 10 — Deterministic TrainingCut and HGT publication

## Goal

Remove timing-triggered scientific training semantics.

## New file

```text
src/v9/hgt/training_cut.py
```

## Modify

```text
src/v9/hgt/training.py
src/v9/hgt/matched_evaluation.py
src/v9/hgt/__init__.py
src/v9/runtime/epoch_runner.py
src/v9/runtime/config.py
```

## Remove as scientific semantics

Deprecate current concepts such as:

```text
hgt_examples_per_train_trigger
hgt_training_duty_cycle as a determinant of update count
wall-clock training windows
opportunistic scientific publication
```

They may remain only as execution scheduling hints if they do not change `TrainingCut`.

## `TrainingCut`

Freeze:

```text
evidence manifest
replay plan/order
RNG streams
optimizer state/hyperparameters
optimizer step count
microbatch plan
gradient accumulation
determinism mode
parent ModelVersion
evaluation manifest
```

## Deterministic kernel contract

For each required operation:

```text
deterministic CUDA
→ deterministic replacement
→ deterministic CPU fallback
→ NONDETERMINISTIC_UNSUPPORTED
```

Unsupported nondeterministic work cannot publish a deterministic H19 model.

## Publication

Candidate state:

```text
PLANNED
TRAINING
TRAINED
EVALUATING
ACCEPTED
REJECTED
FAILED
PUBLISHED
SUPERSEDED
```

Only the next `EpochInferenceView` publishes the selected model.

## Tests

Add:

```text
test_training_cut.py
test_training_cut_replay_determinism.py
test_training_cut_optimizer_count.py
test_training_cut_kernel_fallback.py
test_model_publication_epoch_gate.py
```

Extend:

```text
test_hgt_iterative_training.py
test_hgt_checkpoint_metadata_drift.py
test_hgt_self_describing_checkpoint.py
test_hgt_startup_restore.py
test_hgt_multitask_resources.py
```

## Exit gate

Running the same `TrainingCut` twice in deterministic scientific mode gives the configured deterministic-equivalence result and cannot publish mid-epoch.

---

# 15. Phase 11 — H17 asynchronous-development isolation

## Goal

Implement the actual H17 experiment rather than reusing H19 synchronization.

## Modify

```text
src/v9/runtime/scientific_modes.py
src/v9/runtime/runtime.py
src/v9/runtime/lifecycle.py
src/v9/runtime/transfer_validation.py
src/v9/runtime/epoch_runner.py
GNN→Hydra feedback call sites
```

## Baseline

Set:

```text
ScientificVisibilityMode = ASYNC_DEVELOPMENT
LearnedDevelopmentalFeedback = DISABLED
```

Independent developmental processes commit when their causal/version preconditions hold.

No:

```text
EvidenceCut
DevelopmentalCut
TrainingCut
EpochInferenceView requirement
```

is imposed for the baseline H17 path.

## Secondary factorial condition

Allow explicit:

```text
LearnedDevelopmentalFeedback = ENABLED
```

but report it separately.

All HGT-derived lifecycle/context/validation suggestions carry:

```text
ModelVersion
source handle/view
affected decision
Hydra evidence that accepted/rejected proposal
```

## H17 metrics

Implement:

```text
R_rev
R_churn
P_delta_t
created/retired/active memories
novel useful structures
prediction quality
transfer quality
memory growth
```

## Tests

Add:

```text
test_h17_async_development.py
test_h17_no_epoch_barrier.py
test_h17_hgt_feedback_isolation.py
test_h17_atomic_handle_visibility.py
```

## Exit gate

H17 can run independently of H19 and produces stability/plasticity telemetry without learned developmental feedback.

---

# 16. Phase 12 — Fixed matched H19 TrialManifest

## Goal

Remove the current adaptive-budget confound from matched H19 comparisons.

## Modify

```text
src/v9/runtime/epoch_runner.py
src/v9/runtime/environment_viability.py
src/v9/runtime/multiprocess.py
src/v9/hgt/matched_evaluation.py
```

## Replace matched-budget semantics

Current adaptive game budgeting may remain for ordinary `ASYNC_DEVELOPMENT`.

For `MATCHED_REASONING`, construct fixed:

```text
TrialSpec(
    start state / state reconstruction reference,
    environment seed,
    reset seed,
    actor/job identity,
    fixed horizon,
    timeout/truncation rule,
    curriculum stage,
    train/eval role
)
```

## Execution

```text
restore same start
→ run <= H interactions
→ early terminal
→ discard unused horizon
→ next fixed trial
```

Do not convert unused horizon into another episode.

## Conditions sharing a manifest

```text
HYDRA_ONLY
HYDRA_HGT_SINGLE_PASS
HYDRA_HGT_RECURSIVE
RANDOM_UNTRAINED_RELATIONAL
FROZEN_EARLY_MODEL
MODEL_ONLY_POLICY where feasible
```

## Tests

Add:

```text
test_trial_manifest.py
test_h19_matched_start_states.py
test_h19_unused_horizon_discard.py
test_h19_no_adaptive_budget_confounds.py
```

## Exit gate

Every H19 condition receives the same start states, trial list, horizons, and curriculum exposure.

---

# 17. Phase 13 — H18 StructuralPriorTransform

## Goal

Implement structural-prior ablations without topology/coordinate leakage.

## New module

```text
src/v9/research/structural_prior.py
```

## Modify

Environment adapter boundary used by:

```text
src/v9/runtime/multiprocess.py
environment adapter modules
action legality/translation layer
GNN feature builder
```

## Implement profiles

```text
S0 identity + temporal order
S1 + equality
S2 + coordinates
S3 + adjacency/topology
```

Modes:

```text
ORDINARY
FIXED_COORDINATE_PERMUTATION
CHANGING_COORDINATE_PERMUTATION
TOPOLOGY_WITHHELD
ADJACENCY_WITHHELD
```

## Formal transform

Observation:

```text
environment coordinate c
→ exposed coordinate π(c)
```

Coordinate-bearing action:

```text
exposed target c'
→ adapter executes π^-1(c')
```

When coordinates/topology are withheld, expose opaque target identity rather than original position.

## Anti-leak

Block access to:

```text
raw row/column
original tensor positions
shape-derived topology where ablated
adjacency helpers
connected components
adapter-side topology summaries
untransformed coordinate action ids
```

## Tests

Add:

```text
test_structural_prior_profiles.py
test_coordinate_permutation_roundtrip.py
test_h18_action_inverse_mapping.py
test_h18_topology_anti_leak.py
test_h18_tensor_index_anti_leak.py
```

## Exit gate

Ablating/permuting structure changes only learner-visible structure, not the underlying environment transition dynamics.

---

# 18. Phase 14 — H16 C0/C1/C2/C3 grounding controls

## Goal

Turn symbolic grounding controls into actual experimental conditions.

## New/modify

```text
src/v9/research/experiment_manifest.py
symbol ingestion/grounding modules
src/v9/hgt/grounding_objectives.py
src/v9/runtime/memory_pipeline.py
grounding evaluation/reporting
```

## Conditions

```text
C0 interaction only
C1 symbols only
C2 aligned interaction + symbols
C3 shuffled interaction + symbols
```

## C3 anti-leak

Preserve declared marginals but remove alignment keys:

```text
cross-stream timestamp
shared episode id
shared producer id
shared sequence position
common provenance id
adapter alignment metadata
```

System bookkeeping may retain them outside learner-visible representations.

## G0-G5

Represent and report:

```text
G0 interaction grounding
G1 aligned symbols
G2 descriptive-symbol linkage
G3 prospective symbols
G4 novel grounded composition
G5 symbol-mediated learning about unexperienced interactions
```

## Evaluation

Implement both directions:

```text
symbol → interaction
interaction → symbol
```

plus:

```text
C2-C0
C2-C1
C2-C3
novel composition
symbol-removal persistence
```

## Tests

Add:

```text
test_h16_grounding_conditions.py
test_h16_c3_alignment_destroyed.py
test_h16_c3_no_metadata_leak.py
test_h16_bidirectional_transfer.py
```

## Exit gate

C0-C3 are auditable separate conditions rather than labels within one mixed dataset.

---

# 19. Phase 15 — ResearchPredictionRegistry and F1-F18

## Goal

Make falsification executable rather than descriptive.

## New file

```text
src/v9/research/prediction_registry.py
```

## Implement

```text
ResearchPredictionRegistry
DevelopmentalMilestoneLedger
PredictionResult
    SUPPORTED
    VIOLATED
    INSUFFICIENT_EVIDENCE
```

## Registry entries

Implement explicit mappings for:

```text
F1  Semantic-Prior Necessity
F2  Object-First Emergence
F3  Appearance-Dominated Transfer
F4  World-Model Necessity
F5  Prediction-Violation Allocation
F6  Future-Option Contribution
F7  Explanatory Reach
F8  Context Refinement
F9  Empirical Transfer Validation
F10 Developmental Ordering
F11 Outcome/Strategy Separation
F12 Emergent Target-Like Structure
F13 Efficiency Emergence
F14 Conditional Compression
F15 Architectural Thrashing
F16 Grounded Symbolic Emergence
F17 Developmental Stability
F18 Structural-Prior Dependence
H19-REJECT
```

## Milestones

Record first evidence-qualified:

```text
M1 stable contingency
M2 family
carrier
role
M4 candidate
validated M4 concept
future-option motif
planning/replanning effectiveness
M5/M6/M7 late structures
```

The runtime records order; it never forces it.

## Tests

Add:

```text
test_prediction_registry.py
test_developmental_milestone_ledger.py
test_falsification_status_semantics.py
test_f1_f18_registry_completeness.py
```

## Exit gate

Every paper falsification criterion resolves to concrete stored evidence or `INSUFFICIENT_EVIDENCE`.

---

# 20. Phase 16 — Snapshot and global durable-storage governance

## Goal

Make immutable canonical state sustainable across long runs.

## New file

```text
src/v9/runtime/storage_governor.py
```

## Modify

```text
src/v9/runtime/chunked_snapshot.py
src/v9/runtime/snapshot_backend.py
src/v9/runtime/canonical_store.py
src/v9/runtime/canonical_wal.py
src/v9/runtime/memory_governor.py
src/v9/hgt/epoch_dataset.py
model checkpoint management
```

## Snapshot

Snapshot pins exactly one `CanonicalStateHandle`.

The manifest records:

```text
snapshot_applied_lsn
canonical roots
index roots
scientific identity
schema versions
checksums
```

Snapshot lock work is limited to root/builder handle capture.

## Durable classes

Govern:

```text
WAL
canonical chunks
snapshots
TrainingEvidenceSegments
model checkpoints
optimizer checkpoints
evaluation artifacts
SignatureIndex segments
policy/epoch-view manifests
```

## WAL reclamation

```text
wal_reclaim_lsn =
    min(
        snapshot_applied_lsn,
        hgt_checkpoint_lsn,
        other registered durable consumers
    )
```

## Reference-aware GC

Never delete objects referenced by:

```text
active readers
EpochInferenceViews
recovery snapshots
TrainingCuts
published model provenance
required scientific checkpoints
```

## Tests

Add:

```text
test_storage_governor.py
test_reference_aware_gc.py
test_wal_consumer_retention.py
test_snapshot_handle_consistency.py
test_snapshot_chunk_reclamation.py
test_storage_pressure_drain.py
```

## Exit gate

Long-running durable data has finite configured ceilings and cannot delete live recovery/scientific dependencies.

---

# 21. Phase 17 — Whole-system memory governor and telemetry

## Goal

Bound the complete process tree, shared memory, overlays, snapshots, WAL buffers, and durable pressure.

## Modify

```text
src/v9/runtime/memory_governor.py
src/v9/runtime/parallel_memory_coordinator.py
src/v9/runtime/publication_throughput.py
src/v9/telemetry/
dashboard telemetry modules
```

## Memory accounting

Use mutually exclusive categories:

```text
process-tree private/USS
tracked SHM
snapshot external buffers
other explicitly tracked external buffers
```

Also observe:

```text
RSS
swap
MemAvailable
durable bytes by class
filesystem free bytes/fraction
```

## Sampling

```text
frequent:
    cheap RSS / SHM / MemAvailable / swap

slower:
    USS reconciliation
```

## Governor states

```text
NORMAL
COMPACTING
HARD_PRESSURE_DRAIN
RECOVERING
```

Hard pressure:

```text
stop new actor admission
reduce publication
drain WAL/canonical pipeline
prioritize compaction/GC
avoid optional snapshots/checkpoints
fail safely before host/disk exhaustion
```

## Telemetry additions

At minimum:

```text
wal_durable_lsn
canonical_applied_lsn
snapshot_applied_lsn
hgt_checkpoint_lsn
WAL retained/reclaimable bytes
transport SHM
compiled SHM
canonical overlay bytes
canonical fragment p50/p95/p99
policy projection bytes
EpochInferenceView id/checksum
DevelopmentalCut id/status
TrainingCut id/status
SignatureIndex cache
derivation lease counts/retries
durable storage by class
H17 churn/reversal/persistence
TrialManifest utilization/discarded horizon
```

## Tests

Extend:

```text
test_telemetry_v976.py
test_v978_telemetry_concurrency.py
test_dashboard_telemetry_log.py
test_lifecycle_compaction.py
```

Add:

```text
test_v9716_governor_accounting.py
test_v9716_frontier_telemetry.py
```

## Exit gate

The dashboard exposes enough data to identify memory, WAL, canonical, derivation, HGT, policy, or storage bottlenecks independently.

---

# 22. Phase 18 — Restart semantics for both scientific modes

## Goal

Prove restart correctness after all major components exist.

## Modify

```text
src/v9/runtime/snapshot.py
src/v9/runtime/chunked_snapshot.py
src/v9/runtime/canonical_wal.py
src/v9/runtime/canonical_store.py
src/v9/runtime/epoch_runner.py
src/v9/runtime/scientific_modes.py
```

## Common recovery

```text
load ExperimentManifest
restore latest complete snapshot handle
validate/truncate WAL
replay WAL
restore current CanonicalStateHandle
restore producer sequences
restore TrainingEvidenceManifest
reclaim stale process-run SHM
```

## ASYNC_DEVELOPMENT

Restore:

```text
developmental cursors
leases
dirty state
H17 parameter/perturbation manifest
feedback profile
```

Then resume independent workers.

## MATCHED_REASONING

Restore:

```text
last complete EpochInferenceView
incomplete EvidenceCut/DevelopmentalCut/TrainingCut manifests
exact model/policy/root checksums
```

Finish/reconstruct deterministic cut work before launching new actors.

## Tests

Add:

```text
test_v9716_restart_async.py
test_v9716_restart_matched.py
test_restart_mid_developmental_cut.py
test_restart_mid_training_cut.py
test_restart_epoch_view_integrity.py
```

## Exit gate

Restart cannot silently mix generations, modes, models, policies, roots, or scientific identities.

---

# 23. Phase 19 — End-to-end reproducibility, crash, and soak validation

## Goal

Prove the architecture rather than merely passing unit tests.

## 23.1 Full v9 regression

Run the complete v9 suite after every implementation phase.

Do not run v8 regression tests.

No phase may merge with known v9 failures attributed to the change.

## 23.2 MATCHED_REASONING schedule permutation

Run the same experiment manifest repeatedly with varied:

```text
actor scheduling
stage/shard timing
ingest worker count
derivation worker timing
training execution timing
```

Require identical:

```text
scientific evidence ids
TrialManifest checksum
EvidenceCut
DevelopmentalCut
TrainingCut
EpochInferenceView checksum
selected ModelVersion
final scientific canonical state
```

Operational timing/LSNs may differ only where the design explicitly permits them.

## 23.3 H17 perturbation matrix

Vary:

```text
developmental update rates
hysteresis
thresholds
peer rates
worker scheduling
```

Measure stability/plasticity distributions rather than exact trajectory equality.

## 23.4 Crash-injection matrix

Cover:

```text
torn WAL group
post-WAL/pre-overlay
mid overlay
post-overlay/pre-handle publish
post-handle/pre-snapshot
partial TrainingEvidenceSegment
post-segment/pre-manifest
actor/stage/shard worker death
SHM worker death
snapshot writer death
derivation lease expiry
mid DevelopmentalCut
mid TrainingCut
disk hard pressure
RAM hard pressure
```

## 23.5 Long-run soak

Reference acceptance workload:

```text
>= 10 equivalent epochs
>= 1,000,000 admitted transitions when environments permit
```

Require:

```text
no monotonic unbounded resident-memory trend
no monotonic unbounded SHM trend
no unbounded WAL growth
no unbounded SignatureIndex cache growth
no unbounded derivation state
no unbounded policy projection
no unbounded snapshot external buffers
no unbounded durable-data class
final-three-epoch normalized throughput within 20% of warm-up median
```

## Exit gate

All applicable v9.7.16 acceptance criteria have evidence in the acceptance matrix.

---

# 24. Phase 20 — Cutover and cleanup

## Goal

Remove obsolete paths only after their replacements are proven.

## Remove/deprecate

Only after equivalent v9.7.16 functionality is active:

```text
direct epoch-transition JSONL training authority
scientific hgt_examples_per_train_trigger semantics
scientific hgt_training_duty_cycle update-count semantics
time/step actor policy refresh in MATCHED_REASONING
adaptive previous-result game budgeting in matched H19 experiments
direct mutable canonical publication paths
unbounded derivation scheduling state
per-message transition serialization through actor/stage/shard IPC
independent lifecycle-triggered HGT training
```

Do not remove compatibility readers/checkpoint migration until existing run artifacts needed for analysis can still be loaded.

## Configuration version

Only at final cutover:

```text
ScientificConfig.design_version = "9.7.16"
```

and the default generated config hash changes intentionally.

## Final checks

```text
all v9 tests pass
runtime smoke passes
ASYNC_DEVELOPMENT smoke passes
MATCHED_REASONING deterministic smoke passes
snapshot/WAL restart smoke passes
HGT deterministic training smoke passes
broad-run performance smoke passes
```

---

# 25. Recommended PR / commit sequence

Keep changes reviewable and bisectable.

```text
PR 01  scientific manifests + mode contracts
PR 02  CanonicalStateHandle + immutable chunk store
PR 03  TransactionOverlay + atomic canonical publication
PR 04  CanonicalCommitWAL + recovery
PR 05  SignatureIndexStore + leased derivation
PR 06  transport slab pool + producer-affinity causal admission
PR 07  transaction work/byte/latency bounds
PR 08  DevelopmentalCut
PR 09  EpochInferenceView + bounded policy projection
PR 10  TrainingEvidenceRecord + segment materializer
PR 11  deterministic TrainingCut + model publication
PR 12  H17 isolation
PR 13  matched TrialManifest
PR 14  H18 StructuralPriorTransform
PR 15  H16 C0-C3 controls
PR 16  ResearchPredictionRegistry F1-F18
PR 17  snapshot/storage governor
PR 18  telemetry + whole-host governor
PR 19  dual-mode restart
PR 20  crash/reproducibility/soak acceptance
PR 21  legacy cleanup + final 9.7.16 cutover
```

Each PR must leave `main` runnable.

---

# 26. Dependency graph

```text
Phase 1  scientific contracts
   ↓
Phase 2  canonical store
   ↓
Phase 3  WAL
   ├───────────────┐
   ↓               ↓
Phase 4 derivation Phase 5 transport
   └───────┬───────┘
           ↓
Phase 6 transaction bounds
           ↓
Phase 7 DevelopmentalCut
           ↓
Phase 8 EpochInferenceView
           ↓
Phase 9 TrainingEvidence
           ↓
Phase 10 TrainingCut
      ┌────┴─────┐
      ↓          ↓
Phase 11 H17   Phase 12 H19 trials
                 ↓
              Phase 13/14
              H18 / H16
                 ↓
              Phase 15 research registry

Canonical store + WAL + training evidence
                 ↓
              Phase 16 storage governance
                 ↓
              Phase 17 telemetry/governors
                 ↓
              Phase 18 recovery
                 ↓
              Phase 19 acceptance
                 ↓
              Phase 20 cleanup
```

---

# 27. Highest-risk implementation areas

## 27.1 Canonical COW migration

Risk:

```text
hidden direct mutations bypass CanonicalStateHandle
```

Mitigation:

```text
shadow reads
mutation assertions
root checksum comparison
temporary direct-mutation guard
incremental memory-level migration
```

## 27.2 MATCHED_REASONING completeness

Risk:

```text
one developmental subsystem still mutates actor-visible state outside DevelopmentalCut
```

Mitigation:

```text
central mutation gate
operator registry
test that every canonical mutation declares:
    mode
    cut id or async origin
    WAL transaction
```

## 27.3 Deterministic HGT

Risk:

```text
CUDA nondeterminism or timing-dependent replay changes model outputs
```

Mitigation:

```text
TrainingCut hashes
fixed replay plan
deterministic algorithm checks
CPU fallback
checkpoint equivalence tests
```

## 27.4 H18 leakage

Risk:

```text
topology survives through tensor storage, action ids, or adapter metadata
```

Mitigation:

```text
adapter-boundary transform
learner-input audit
opaque targets when required
explicit inverse action transform
negative leakage tests
```

## 27.5 Persistent-data growth

Risk:

```text
the RAM leak becomes a WAL/chunk/training-evidence/model-checkpoint disk leak
```

Mitigation:

```text
class-level byte accounting
reference-aware GC
consumer-frontier WAL retention
hard filesystem pressure gates
long-run soak
```

---

# 28. Definition of complete

v9.7.16 implementation is complete only when all of the following are true:

```text
1. Canonical state publishes through immutable CanonicalStateHandle.
2. WAL is the redo authority and crash recovery is proven.
3. Producer causality and IPC memory are bounded.
4. Derivation state is persistent/bounded and merge-order independent.
5. ASYNC_DEVELOPMENT and MATCHED_REASONING have distinct enforced semantics.
6. MATCHED_REASONING uses EvidenceCut → DevelopmentalCut → TrainingCut.
7. H19 uses fixed matched TrialManifest trials.
8. HGT training evidence is WAL-backed and singular in authority.
9. HGT scientific training is deterministic or explicitly invalidated.
10. H17 baseline has no learned HGT→Hydra feedback.
11. H16 C0-C3 and H18 structural-prior controls are executable.
12. F1-F18 and H19 rejection are directly traceable to evidence.
13. Snapshot, WAL, training evidence, model checkpoints, and indexes are storage-bounded.
14. Whole-system RAM/SHM/swap pressure is bounded.
15. Dual-mode restart is proven.
16. Full v9 regression is clean.
17. Crash matrix is clean.
18. MATCHED_REASONING schedule-variation reproducibility is clean.
19. H17 perturbation experiments run without hidden synchronization.
20. Long-run soak shows bounded memory/storage and stable normalized throughput.
```
