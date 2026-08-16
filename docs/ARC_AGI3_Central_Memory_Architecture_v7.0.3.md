# ARC-AGI-3 Central Memory Architecture

## System Design and Implementation Plan — v7.0.3

**Target:** clean-break v7 architecture with canonical RAM cognition, immutable published generations, v7-native durable storage, batch-first access, bounded candidate growth, lock-free parallel read scalability, single-owner canonical writes, incremental parallel derivation, optional GPU acceleration, and a controlled path to remote sampling.

**Compatibility policy:** **none.** v7 does not provide runtime, schema, API, or reporting backward compatibility with v6. v6 is retained only as a scientific and performance reference implementation.

---

# 1. Executive Summary

v7 is a clean architectural break from v6.

The central change is not simply moving SQLite data into RAM. The design separates five distinct concerns that were previously mixed together:

1. **CanonicalMemoryWriter** — the single authoritative mutation owner.
2. **MemoryReadView** — immutable, cognition-oriented read state used by workers and evaluators.
3. **EvidenceStore** — durable historical evidence and provenance, separated from active semantic memory.
4. **DurableGenerationStore** — transactional persistence of committed generations.
5. **CandidateProvider** — bounded candidate generation/ranking, initially CPU and optionally GPU later.

The core execution model is generation-oriented:

```text
Published Generation N
    immutable
    read by workers/reports/GPU

Mutable Generation N+1
    owned by CanonicalMemoryWriter
    receives batched evidence and mutations

commit
    ↓

Published Generation N+1
Mutable Generation N+2
```

This gives the system MVCC-like behavior without implementing a general MVCC database.

The design deliberately avoids several expensive or risky systems projects in v7:

- no legacy v6 schema compatibility;
- no generic `MemoryStore` pretending every backend behaves identically;
- no custom WAL/database engine;
- no adaptive semantic eviction;
- no distributed canonical memory;
- no sharding;
- no synchronous global RPC for every memory read;
- no GPU-first vector representation.

The implementation is split into:

- **v7.0** — clean-break architecture, v7-native schema, numeric IDs, canonical writer, immutable read views, evidence separation, batch-first APIs, generation commit, RAM M1-M6, bounded candidates.
- **v7.1** — compact memory segments, zero/low-copy worker read views, parallel read scaling, batched/vectorized algorithms, elimination of loop-heavy hot paths.
- **v7.2** — GPU acceleration only for a measured candidate-comparison bottleneck.
- **v7.3** — remote sampler nodes while retaining one canonical memory owner.
- **v8+** — distributed canonical memory, sharding, adaptive paging, custom durability, or replicated ordering only if measured v7 limits require them.

---

# 2. Why a Clean Break Is Preferable

The current v6 implementation is valuable as an experimental reference but is not a desirable architectural constraint for v7.

Current code couples cognition and persistence:

- `MemorySubstrate` owns SQLite schema creation, nodes, edges, evidence, scores, lifecycle and commit behavior.
- `MemoryQueryEngine` traverses graph structures and still performs direct SQL reads in hot scoring paths.
- reporting creates physical read-only SQLite evidence snapshots.
- higher-order derivation contains substantial Python iteration even where portions of transfer processing already use `ProcessPoolExecutor`.

A compatibility layer would preserve too many of these assumptions.

v7 therefore treats v6 only as:

```text
scientific oracle
performance baseline
regression comparison source
```

and not as:

```text
runtime API
schema contract
storage format
directory layout
query contract
```

---

# 3. Architectural Principles

## 3.1 Canonical cognition is RAM-first

M1-M6 active semantic state is canonical in RAM during execution.

Disk is durability, not the operational query engine.

## 3.2 Read and write models are different

The best representation for mutation is not necessarily the best representation for cognition.

Therefore:

```text
Canonical write state != MemoryReadView
```

The writer may maintain mutable maps and append buffers while readers consume compact indexes and arrays.

## 3.3 Evidence history is not active cognition

Evidence and provenance may grow monotonically. Active semantic memory should not.

The architecture separates:

```text
Active Semantic Memory
Evidence Ledger
```

## 3.4 APIs are batch-first

The primary API unit is a batch, not one node, edge, score or query.

## 3.5 Candidate search is bounded before acceleration

No CPU/GPU optimization may recreate unrestricted pairwise comparison.

## 3.6 Published generations are immutable

Workers, reports and optional GPU indexes consume immutable generation snapshots.

## 3.7 Parallel reads are preferred over parallel canonical writes

Reads scale horizontally. Canonical writes remain logically serialized in v7.

This keeps semantics deterministic while removing the read-side bottleneck.

---

# 4. Target Logical Architecture

```text
                        ┌──────────────────────────┐
                        │     Sampling Workers     │
                        │        10 - 100+         │
                        └───────┬─────────┬────────┘
                                │         │
                     read-only  │         │ batched evidence
                                │         │
                                v         v
                    ┌────────────────┐   ┌─────────────────────────┐
                    │ MemoryReadView │   │ CanonicalMemoryWriter   │
                    │ Generation N   │   │ Mutable Generation N+1  │
                    │ immutable      │   │                         │
                    ├────────────────┤   │ nodes / edges / scores  │
                    │ contingency idx│   │ lifecycle / promotion   │
                    │ role idx       │   │ candidate scheduling    │
                    │ concept idx    │   │ batch mutation ingress  │
                    │ FO idx         │   └───────────┬─────────────┘
                    │ failure idx    │               │
                    └────────────────┘               │ commit
                                                     v
                                        ┌─────────────────────────┐
                                        │ Generation Publisher    │
                                        │ freeze + compact + index│
                                        └───────────┬─────────────┘
                                                    │
                         ┌──────────────────────────┼─────────────────────────┐
                         │                          │                         │
                         v                          v                         v
                  MemoryReadView N+1        DurableGenerationStore      EvidenceStore
                         │                    v7-native SQLite             SQLite/columnar
              ┌──────────┼─────────┐
              │          │         │
              v          v         v
           workers     reports   CandidateProvider
                                   │
                                   ├── CPU indexed
                                   └── GPU optional v7.2
```

---

# 5. Core Contracts

The earlier generic `MemoryStore` is removed.

## 5.1 CanonicalMemoryWriter

Purpose: authoritative mutation only.

Primary operations:

```text
apply_mutation_batch(batch)
apply_score_batch(batch)
apply_edge_batch(batch)
apply_lifecycle_batch(batch)
apply_promotion_batch(batch)
append_support_batch(batch)

begin_generation()
commit_generation()
abort_generation()
```

There should be no cognition-oriented graph traversal API on this interface.

## 5.2 MemoryReadView

Purpose: immutable, generation-specific cognition queries.

Primary operations:

```text
generation_id()

get_nodes(ids[])
get_scores(ids[])

score_inputs(actions[])
lookup_contingencies(context_action_keys[])
lookup_action_evidence(actions[])
lookup_roles(keys[])
lookup_concepts(keys[])
lookup_future_option(keys[])
lookup_failure_evidence(keys[])

neighbors(node_ids[], relation_type)
query_candidates(batch)
```

ReadView APIs return compact arrays/records rather than storage-engine rows.

## 5.3 EvidenceStore

Purpose: durable historical support and provenance.

```text
append_evidence_batch(batch)
append_transfer_trials(batch)
append_contradiction_batch(batch)
append_promotion_evidence(batch)
append_lifecycle_history(batch)

read_evidence(refs[])
read_provenance(refs[])
```

Active semantic memories store compact evidence references and summaries.

## 5.4 DurableGenerationStore

Purpose: transactionally persist v7 generations.

```text
persist_generation_delta(...)
mark_generation_committed(...)
load_generation(...)
load_latest_committed(...)
```

## 5.5 CandidateProvider

Purpose: bounded candidate generation.

```text
family_candidates(batch)
role_candidates(batch)
concept_candidates(batch)
transfer_candidates(batch)
```

Implementations:

```text
IndexedCpuCandidateProvider
GpuCandidateProvider      # v7.2 only
```

---

# 6. Numeric Identity Model

Hot paths should not manipulate large string identifiers.

## 6.1 Internal identity

```text
MemoryId = uint64
```

Each memory record includes compact metadata:

```text
memory_id
level_id
type_id
created_generation
updated_generation
status_flags
```

## 6.2 External/provenance identity

Human-readable or stable canonical keys remain available in the evidence/provenance layer.

Example:

```text
canonical_key_hash
canonical_key_ref
source_run
source_game
source_epoch
source_global_step
```

This avoids carrying strings such as `M3:role:...` through every RAM graph and GPU structure.

---

# 7. v7-Native Durable Schema

No v6 compatibility schema is retained.

A minimal durable model should be designed around v7 semantics.

Suggested tables:

```text
memory_instances
generations
generation_batches

memory_nodes
memory_edges
memory_scores

memory_evidence_refs
memory_promotions
memory_lifecycle

evidence_records
transfer_trials
contradiction_records
provenance_records
```

Key properties:

- integer IDs;
- generation ranges;
- typed integer enums where practical;
- compact payload references instead of repeated JSON;
- append-heavy evidence tables;
- active-state tables optimized for generation restore;
- no requirement to recreate v6 table names or column semantics.

SQLite remains the first durability engine because its transaction and recovery behavior are useful, not because its schema must resemble v6.

---

# 8. Generation Model

## 8.1 Double-generation execution

At any point:

```text
Generation N
    immutable published state

Generation N+1
    mutable writer-owned frontier
```

Workers read N while N+1 is being built.

The system does not allow readers to inspect partially mutated canonical structures directly.

## 8.2 Commit sequence

```text
1. stop accepting mutations for current frontier cutoff
2. drain accepted batch queue up to cutoff
3. finalize derived state required for commit
4. persist generation delta transactionally
5. persist evidence references
6. compact/freeze read structures
7. publish Generation N+1 atomically
8. start mutable Generation N+2
```

## 8.3 Read staleness

Workers can operate against the latest published generation rather than the latest individual mutation.

This introduces bounded staleness but removes lock/RPC contention.

The maximum staleness must be measured as:

```text
published_generation_lag
published_step_lag
published_wall_time_lag_ms
```

If specific cognition requires immediate selected-action feedback, that small local/session state may be handled separately from the global published read view.

---

# 9. Active Semantic Memory vs Evidence Ledger

This separation is central to controlling growth.

## 9.1 Active semantic memory contains

```text
M1-M6 active nodes
validated active semantic relationships
current scores
current lifecycle state
current support summaries
compact evidence references
current candidate indexes
```

## 9.2 Evidence ledger contains

```text
individual observations
historical support events
transfer attempts
failed transfer attempts
contradictions
promotion evidence
full provenance
historical lifecycle transitions
cold rejected candidates
```

## 9.3 Canonical records should reference evidence

Example semantic record:

```text
memory_id
support_count
evidence_ref_start
evidence_ref_count
transfer_success_summary
prediction_summary
explanatory_summary
```

not:

```text
large repeated provenance JSON
all historical transfer rows
all source interaction details
```

This prevents evidence growth from directly becoming active RAM growth.

---

# 10. RAM Physical Design

## 10.1 Mutable frontier

Current-generation changes use append-friendly structures:

- hash maps;
- integer-key maps;
- append arrays;
- per-relation delta buffers;
- dirty-bit arrays;
- batch-local temporary vectors.

## 10.2 Stable compact segments

Published/stable state uses:

- contiguous arrays;
- columnar score arrays;
- packed node metadata;
- sorted integer indexes;
- typed adjacency arrays;
- CSR-like structures only where read-heavy and immutable.

## 10.3 Generational consolidation

```text
Stable Segment N
       +
Mutable Frontier N+1
       ↓
Generation commit
       ↓
Stable Segment N+1
```

Do not mutate large compressed CSR arrays in place.

## 10.4 Memory arenas

Suggested arenas:

```text
NodeArena
ScoreArena
M1ContingencyArena
M2FamilyArena
M3RoleArena
M4ConceptArena
M5WorldModelArena
M6StrategyArena
StableEdgeArena
MutableEdgeArena
IndexArena
```

This allows precise measurement of RAM cost by semantic class.

---

# 11. Cognition-Oriented Read Model

The current code often reconstructs answers by traversing generic graph relationships.

v7 should maintain read indexes corresponding to actual cognitive questions.

## 11.1 Required initial indexes

```text
(context_signature, action) -> contingency IDs

action -> interaction/evidence summary
action -> future-option aggregate
action -> failure/contradiction aggregate

(context_signature, action, family) -> bounded role candidate IDs
(context_signature, action) -> bounded fallback role candidate IDs
family -> role IDs

role -> concept IDs

context/task signature -> M6 strategy IDs

(source_id, relation_type) -> target IDs
(target_id, relation_type) -> source IDs
```

## 11.2 Why this matters

A RAM graph traversal is faster than a SQLite graph traversal, but an indexed direct lookup is faster than both.

v7 should not merely move v6 algorithms into RAM.

It should remove avoidable graph reconstruction from the hot path.

---

# 12. Access-Time Optimization

The target is not only lower storage latency. It is fewer memory operations, fewer indirections, fewer Python objects, fewer RPC boundaries and fewer repeated calculations.

## 12.1 Access hierarchy

Preferred order:

```text
1. worker-local immutable array/index
2. shared-memory immutable index
3. canonical RAM owner
4. durable SQLite
5. cold evidence lookup
```

Normal action scoring should stay almost entirely at levels 1-2.

## 12.2 Batch read amplification

Instead of:

```text
for action:
    lookup contingency
    lookup future option
    lookup failures
    lookup roles
    lookup concepts
```

use:

```text
batch = all available actions
read_view.score_inputs(batch)
```

One call should return all required arrays.

## 12.3 Structure-of-arrays over object-of-structures

Prefer:

```text
node_ids[]
future_option_delta[]
transfer_score[]
replay_priority[]
```

over millions of Python objects each carrying dictionaries.

Benefits:

- less memory;
- sequential access;
- better CPU cache locality;
- easier vectorization;
- direct NumPy/PyTorch interoperability;
- cheaper shared-memory publication.

## 12.4 Precomputed aggregate indexes

Some values currently recomputed from lower-level evidence during each query should be incrementally maintained.

Examples:

```text
action_future_option_sum
action_future_option_count
action_positive_count
action_negative_count
action_failure_count
action_contradiction_flag
```

Then action scoring reads O(1) aggregates rather than rescanning all action evidence.

## 12.5 Batch complete action scoring

The unit of cognition-side scoring is the full available-action set, not one action at a time.

Instead of calling prediction, future-option, failure, role and concept retrieval separately for every action, the read view should assemble one compact scoring matrix:

```text
available_actions[]
    ↓
score_inputs(actions[])
    ↓
contingency features
future-option aggregates
failure/contradiction aggregates
bounded role candidates
role -> concept expansion
    ↓
vectorized action scoring
```

Role matches are computed once per action batch and reused by prediction, role scoring and concept scoring. Concept expansion must never trigger a second global role search.

Target complexity should depend on action count and a bounded candidate size K, not total M3/M4 population:

```text
O(actions × bounded_K)
```

## 12.6 Avoid repeated serialization

Do not serialize JSON/DTO payloads for same-host worker reads.

Use:

- shared arrays;
- memoryviews;
- fixed-layout binary records;
- integer IDs;
- offset/length pairs.

Serialization should exist mainly at process/host boundaries.

---

# 13. Current Loop-Heavy Bottlenecks to Eliminate

Inspection of the current v6 code shows several patterns that should not be reproduced in v7.

## 13.1 Role lookup scans all M3 roles

Current `MemoryQueryEngine.find_similar_roles()` iterates all M3 `FunctionalRoleMemory` nodes and then performs nested edge traversal for each role.

Conceptually:

```text
for role in all_roles:
    for carrier in carriers_for_role:
        for family in families_for_carrier:
        for context in contexts_for_carrier:
        for interaction in interactions_for_carrier:
            ...
```

This scales with global M3 population.

### v7 replacement

Maintain direct candidate indexes:

```text
(context_signature, action, family) -> bounded role IDs
(context_signature, action) -> bounded fallback role IDs
family -> role IDs
```

Then rank only the bounded candidate set.

Target complexity changes from approximately:

```text
O(all_roles × neighborhood_work)
```

to:

```text
O(candidate_roles)
```

where candidate roles are bounded.

## 13.2 Concept matching repeatedly invokes role search

Current `find_concept_matches()` loops over all concepts and can call `find_similar_roles()` while processing concept role links.

This can effectively nest concept population over role population.

### v7 replacement

Compute role matches **once per action batch**:

```text
role_matches_by_action = batch_role_lookup(actions)
```

Then concept lookup becomes:

```text
matched_role_ids -> concept IDs
```

through an inverted index.

No repeated global role search.

## 13.3 Future-option action scoring rescans action evidence

Current future-option and failure evidence paths iterate action edges and perform score lookups per evidence node.

### v7 replacement

Maintain incremental action aggregates in the read model.

When evidence changes:

```text
update action_future_option_sum/count
update action_positive/negative counts
update action_failure_count
update action_contradiction_count
```

When scoring:

```text
one indexed aggregate read
```

This converts an evidence-length-dependent read into O(1) or O(number_of_actions).

## 13.4 Higher-order derivation still contains Python nested loops

Current higher-order code already uses `ProcessPoolExecutor` for transfer chunks, which is useful, but substantial work remains in Python iteration:

- role/family membership counting;
- concept-role structure expansion;
- world-model relation creation;
- transfer validation scans;
- per-role/per-family set unions;
- row-by-row link insertion.

Examples in current code include operations equivalent to:

```text
for family in candidate_families:
    role_count = sum(1 for role in roles if family in role_links[role])

for role in roles:
    for carrier in carriers_for_role:
        insert_link(...)

for concept:
    for role:
        for transfer_attempt:
            ...
```

### v7 replacement

Prefer:

- inverted indexes;
- integer bitsets where populations are dense enough;
- sorted integer intersections;
- vectorized membership counts;
- batch edge insertion;
- batch transfer evaluation;
- pre-grouped arrays;
- one-time per-generation indexes.

---

# 14. Algorithmic Batching Strategy

Every major derivation stage should be explicitly classified as one of:

```text
scalar unavoidable
batchable CPU
parallel batch CPU
vectorizable SIMD/NumPy
GPU candidate
```

No performance-critical function should remain an accidental Python loop without this classification.

## 14.1 M1 contingency operations

Batchable:

- canonical-key generation;
- support increments;
- score updates;
- context/action lookup.

Use vectorized hash/key lookup and batch mutations.

## 14.2 M2 family formation

Potentially batchable:

- signature grouping;
- support aggregation;
- member assignment.

Use integer-coded signatures and grouped/sorted arrays.

## 14.3 M3 role candidate generation

High-priority optimization:

```text
carrier neighborhoods
    ↓
signature/index extraction
    ↓
group candidate role neighborhoods
    ↓
bounded structural comparisons
```

Avoid comparing every role with every carrier.

## 14.4 M4 concept candidate generation

Use role->concept inverted indexes and batched structural summaries.

Do not independently reconstruct each concept's entire source graph.

## 14.5 M5 world-model construction

Current family-count and link creation loops should be converted to precomputed membership arrays and batch insertion.

## 14.6 M6 strategy comparison

Trajectory summary records should be vectorized around:

```text
outcome signature
cost
future-option gain
loop ratio
blocked ratio
repeated-state ratio
```

Candidate comparisons should group by comparable outcome signature before ranking efficiency.

---

# 15. Parallel Execution Model

Parallelism must be designed by dependency, not added indiscriminately.

## 15.1 Sampling workers

Highly parallel.

Each environment/game rollout can run independently except for read-view refresh and evidence submission.

Target:

```text
N workers
shared immutable read generation
independent rollout
batch evidence submission
```

## 15.2 Memory reads

Highly parallel.

Published `MemoryReadView` is immutable, so readers need no locks.

Ideal implementation:

```text
many processes
    ↓
same memory-mapped/shared immutable arrays
```

This is the largest safe source of parallelism.

## 15.3 Canonical mutations

Logically serialized in v7.

However, work can be split into:

```text
parallel preprocessing
    ↓
ordered canonical apply
```

Example:

workers/process pool may precompute:

- normalized IDs;
- signatures;
- candidate features;
- evidence summaries;

then the writer applies compact batches.

## 15.4 Persistence writes

Do not allow many processes to write SQLite independently.

Use:

```text
one durable writer
large transactions
```

Parallel SQLite writers would reintroduce contention.

Parallelize serialization/preparation, not commit ownership.

## 15.5 Higher-order derivation

Stages with no dependency between candidates should use chunked process parallelism.

But multiprocessing is useful only when:

```text
compute_per_chunk >> serialization/process overhead
```

Large immutable arrays should preferably be shared/memory-mapped rather than copied into each process.

Current transfer multiprocessing should not pass large nested structures such as complete role rows or profile caches to every worker job. v7.1 workers should attach once to the immutable generation arrays and receive compact work descriptors only:

```text
start_index
end_index
candidate IDs / offsets
```

This avoids repeated pickling, copy amplification and per-process memory multiplication.

Stage-level dependencies remain explicit:

```text
M2 families
   ↓
M3 roles
   ↓
transfer validation
   ↓
M4 concepts
   ↓
M5 world-model components
```

These dependent stages are not run concurrently against mutable state. Parallelism is applied within a stage over independent dirty candidate chunks, followed by a deterministic merge.

## 15.6 Reporting

H01-H12 independent evaluators can continue in parallel, reading the same immutable generation.

Shared generation indexes should be constructed once, not once per evaluator.

## 15.7 Four execution domains

v7 separates concurrency into four domains with different correctness rules:

```text
SAMPLING
40-100+ independent processes
          │
          v
READING
lock-free shared immutable generation
all workers parallel
          │
          v
DERIVATION
parallel chunks over dirty candidate sets
          │
          v
WRITING
single ordered canonical batch writer

GPU (v7.2)
candidate batch -> GPU ranking -> IDs -> CPU validation
```

The design does not use a central synchronous read RPC for normal same-host cognition. The shared immutable read view is attached directly by workers. The canonical writer is centralized because mutation ordering must remain deterministic; read scaling and derivation parallelism provide the primary concurrency gains.

---

# 16. Parallel Read Design

## 16.1 Lock-free published state

A published read view is immutable.

Readers need:

- no write locks;
- no SQLite connection;
- no manager proxy;
- no mutable Python shared objects.

## 16.2 Atomic generation publication

Use a small atomic publication record:

```text
published_generation_id
read_view_manifest_path/shared_segment_id
```

Workers periodically check whether a newer generation exists.

## 16.3 Worker refresh

Worker does:

```text
if generation_id changed:
    attach new read view
    release old view when no longer in use
```

No full process restart.

## 16.4 Read fan-out

One published generation can be read by:

- 50+ samplers;
- report evaluators;
- candidate workers;

without multiplying RAM if memory-mapped/shared structures are used correctly.

---

# 17. Write-Side Throughput

The canonical writer can still become a bottleneck.

v7 therefore minimizes its work.

## 17.1 Writer should not perform expensive derivation inline

The writer's critical path should mainly be:

```text
validate batch envelope
resolve IDs
apply compact updates
mark dirty sets
enqueue derivation work
```

Expensive candidate generation should happen outside the mutation lock/critical section.

## 17.2 Per-generation dirty sets

Maintain:

```text
dirty_M1
dirty_M2
dirty_M3
dirty_M4
dirty_M5
dirty_M6
dirty_edges
dirty_scores
```

Derivation processes only dirty neighborhoods where scientifically valid.

Do not recompute all memory levels each generation.

## 17.3 Coalescing

If the same memory receives 500 support increments in one batch interval:

avoid:

```text
500 mutations
```

use:

```text
memory_id -> +500 support
```

Similarly coalesce:

- last_seen max;
- first_seen min;
- score deltas;
- repeated edge support increments;
- duplicate edge additions into one edge plus accumulated support.

---

# 18. Incremental Derivation

A major late-epoch bottleneck is full or broad recomputation against a growing lifetime graph.

v7 should make incremental derivation a first-class invariant.

## 18.1 Dirty-neighborhood derivation

New M1 changes should identify affected:

```text
families
carriers
roles
concepts
world-model components
strategies
```

Only these neighborhoods become derivation candidates.

## 18.2 Dependency graph

Maintain lightweight reverse dependencies:

```text
M1 -> M2
M2 -> M3
M3 -> M4
M4 -> M5
trajectory/outcome -> M6
```

When a lower level changes, propagate dirty flags upward.

## 18.3 Periodic full validation

Incremental operation may drift.

Therefore:

```text
every generation: incremental derivation
every N generations: optional full consistency validation
```

Full validation is scientific checking, not the normal runtime path.

---

# 19. Data Structures for Fast Set Operations

Higher-order reasoning performs many set operations.

Python `set[str]` is flexible but expensive.

v7 should progressively use integer structures.

## 19.1 Sorted integer arrays

Good for:

- compact membership;
- intersections;
- merge joins;
- deterministic output.

## 19.2 Dense bitsets

Useful only where ID domains are sufficiently compact/dense.

Fast operations:

```text
AND   intersection
OR    union
popcount overlap count
```

This can drastically accelerate role/family/concept overlap calculations.

## 19.3 Roaring bitmap-style structures

Potentially useful for sparse large ID sets.

Not mandatory in v7.0; evaluate in v7.1.

---

# 20. Bounded Candidate Search

Candidate generation remains explicitly bounded.

```text
population
    ↓
cheap exact/index filters
    ↓
bounded candidate pool
    ↓
structural scoring
    ↓
top-K unvalidated frontier
    ↓
semantic validation
    ↓
validated relationships
```

K applies only to the unvalidated frontier.

Validated relationships may exceed K.

Audit every relation class:

```text
population
prefiltered
scored
retained
validated
rejected
budget_exhausted
```

---

# 21. GPU Boundary — v7.2

GPU is introduced only after CPU batching and indexing.

This ordering matters because many apparent GPU candidates may disappear once the algorithm is converted from global Python loops to bounded indexed batch operations.

GPU criteria:

1. CPU candidate stage remains material after v7.1.
2. Candidate features can be expressed as dense/sparse batches.
3. CPU reference ranking exists.
4. top-K recall is measurable.
5. H2D/D2H cost is smaller than saved compute.

GPU returns candidate IDs/scores only.

No GPU output is semantic evidence.

---

# 22. Bottleneck Taxonomy

v7 performance work should classify bottlenecks into five classes.

## A. Boundary bottlenecks

Examples:

- SQL call per node;
- manager/RPC call per read;
- JSON serialization per object.

Fix:

- batch;
- shared arrays;
- immutable read views.

## B. Algorithmic bottlenecks

Examples:

- all-role scans;
- repeated concept->role->carrier traversal;
- global candidate cross-products.

Fix:

- indexes;
- inverted indexes;
- bounded candidate sets;
- incremental derivation.

## C. Python execution bottlenecks

Examples:

- nested Python loops;
- millions of dictionaries;
- repeated set construction.

Fix:

- arrays;
- vectorized operations;
- bitsets;
- compiled libraries;
- process chunks where appropriate.

## D. Synchronization bottlenecks

Examples:

- central read RPC;
- shared manager proxies;
- multiple SQLite writers;
- locks around read structures.

Fix:

- immutable publication;
- single writer;
- lock-free reads.

## E. Memory-bandwidth/cache bottlenecks

Examples:

- object pointer chasing;
- duplicated string IDs;
- scattered adjacency dictionaries.

Fix:

- numeric IDs;
- contiguous arrays;
- structure-of-arrays;
- packed adjacency.

---

# 23. Performance Instrumentation Required Before/Throughout v7

## 23.1 Per-function hot-path timing

Add timing/counters for:

```text
contingency_lookup
action_evidence_lookup
role_lookup
concept_lookup
future_option_lookup
failure_lookup
candidate_generation
candidate_validation
promotion
world_model_derivation
durable_flush
generation_publish
```

## 23.2 Loop counters

Every major derivation should record:

```text
items_outer_loop
items_inner_loop
pair_comparisons
membership_tests
set_unions
edge_traversals
rows_materialized
```

This exposes accidental O(N²) behavior.

## 23.3 Batch efficiency

```text
average_batch_size
p50_batch_size
p95_batch_size
mutations_before_coalesce
mutations_after_coalesce
queries_before_batch
queries_after_batch
```

## 23.4 Parallel efficiency

```text
worker_cpu_utilization
worker_wait_time
queue_depth
process_pool_serialization_bytes
task_duration
pool_overhead
speedup_vs_1_worker
```

A parallel stage that gives <1.5x speedup at 4 workers should be examined rather than blindly scaled.

---

# 24. Versioned Implementation Plan

# v7.0 — Clean-Break Core

## Phase 0 — Baseline

Capture v6 performance and semantic outputs.

Required:

- phase timings;
- SQL operations;
- RSS;
- edge/population growth;
- current query counts;
- process/worker wait;
- major loop counts.

## Phase 1 — New v7 identity and schema

Implement:

- `src/v7`;
- numeric memory IDs;
- memory instance;
- generation identity;
- v7-native durable schema;
- no migration adapter.

## Phase 2 — Core contracts

Implement:

- `CanonicalMemoryWriter`;
- `MemoryReadView`;
- `EvidenceStore`;
- `DurableGenerationStore`;
- `CandidateProvider`.

## Phase 3 — Active semantic RAM

Implement:

- M1-M6 active records;
- score arrays;
- mutable edge frontier;
- cognition-oriented indexes;
- deterministic recent-M0 budget.

## Phase 4 — Double-generation model

Implement:

- immutable Generation N;
- mutable Generation N+1;
- atomic publish;
- worker generation tracking.

## Phase 5 — Batch-first mutation and durability

Implement:

- evidence batches;
- mutation coalescing;
- dirty sets;
- one SQLite durable writer;
- large transactions;
- batch IDs;
- generation commit markers.

## Phase 6 — Evidence separation

Move historical provenance/support detail to EvidenceStore.

Canonical RAM stores compact evidence references/summaries.

## Phase 7 — Bounded CPU candidates

Implement relation-specific indexed CPU candidate providers and audit counters.

## Phase 8 — Generation-based reporting

Reports consume immutable v7 generation/read-view + evidence interfaces.

No v6 report database compatibility.

### v7.0 exit criteria

- zero runtime dependency on v6 storage/schema;
- hot cognition reads from MemoryReadView;
- no direct SQL in cognition;
- active semantic state separate from evidence history;
- transaction count reduced by >=10x where applicable;
- no unrestricted candidate cross-product;
- deterministic scientific results pass new v7 fixtures and v6 reference comparison where comparable.

---

# v7.1 — Access-Time, Parallelism and Algorithm Optimization

## Phase 1 — Compact arenas

Convert hot populations to:

- numeric arrays;
- structure-of-arrays;
- packed adjacency;
- compact offsets.

Measure bytes per semantic unit.

## Phase 2 — Shared/zero-copy read views

- publish immutable numeric arrays through shared memory or memory mapping;
- attach sampler and derivation workers directly to the same generation;
- prohibit Manager-proxy/RPC reads in normal same-host hot paths;
- pass only IDs/ranges/offsets to process workers rather than large Python role/profile structures.

Publish immutable generation structures through:

- shared memory;
- mmap;
- or equivalent low-copy mechanism.

Eliminate Manager proxy and per-read RPC usage.

## Phase 3 — Batch action scoring

One batch operation per worker decision point:

```text
all available actions
    ↓
contingency + FO + failure + role + concept inputs
    ↓
vectorized score
```

## Phase 4 — Remove repeated global scans

Replace:

- all-role scans;
- all-concept scans;
- repeated role lookup inside concept lookup;
- edge-by-edge score retrieval.

with dedicated indexes and aggregates.

## Phase 5 — Incremental higher-order derivation

Use dirty dependency propagation rather than full generation recomputation.

## Phase 6 — Vectorized set/membership operations

Evaluate:

- sorted integer merges;
- NumPy membership;
- bitsets/bitmap structures.

## Phase 7 — Parallel derivation chunks

Parallelize compute-heavy independent candidate chunks.

Avoid copying full graph dictionaries to every process.

Use shared immutable generation data.

### v7.1 exit criteria

- worker read throughput scales close to linearly until CPU/memory bandwidth becomes limiting;
- central writer is not materially loaded by read requests;
- action scoring no longer scales with total M3/M4 population;
- major higher-order stages operate on dirty/bounded candidate sets;
- loop comparison counts demonstrate sub-quadratic active behavior;
- RAM bytes per semantic unit materially improve.

---

# v7.2 — GPU Candidate Acceleration

Only after v7.1 profiling.

Steps:

1. identify dominant candidate kernel;
2. implement CPU exhaustive/reference batch;
3. define versioned features;
4. implement GPU CandidateProvider;
5. incremental dirty synchronization;
6. measure top-K recall and full phase speedup.

Exit:

- >=2x affected end-to-end phase speedup;
- target recall met;
- CPU semantic validation unchanged.

---

# v7.3 — Remote Samplers

Scale rollout CPU across hosts.

Keep:

- one canonical writer;
- one canonical generation publisher;
- immutable remote read snapshot or bounded query cache;
- batched network evidence.

Do not distribute canonical semantic memory.

---

# v8 — Conditional High-Risk Work

Only after v7 measured limits.

Potential:

- distributed canonical memory;
- memory sharding;
- adaptive semantic paging;
- custom WAL/checkpoint engine;
- multi-owner writes;
- distributed generation barriers;
- replicated identity/order service;
- multi-GPU distributed candidate service.

---

# 25. Recommended v7 Module Structure

```text
src/v7/
    memory/
        ids.py
        schema.py
        writer.py
        read_view.py
        generation.py
        publisher.py
        durable_store.py
        evidence_store.py

        arenas/
            nodes.py
            scores.py
            edges.py
            contingencies.py
            families.py
            roles.py
            concepts.py
            world_models.py
            strategies.py

        indexes/
            contingency.py
            action_evidence.py
            role.py
            concept.py
            future_option.py
            failure.py
            adjacency.py

        candidates/
            base.py
            cpu.py
            gpu.py          # v7.2

        transport/
            local.py
            remote.py       # v7.3

    cognition/
        query_engine.py
        action_scoring.py

    derivation/
        dependencies.py
        families.py
        roles.py
        transfers.py
        concepts.py
        world_models.py
        strategies.py

    reporting/
        generation_reader.py
        evidence_reader.py

    cli.py
```

This explicitly prevents one new `MemorySubstrate` monolith from emerging.

---

# 26. Highest-Priority Bottlenecks to Attack

The implementation order should follow measured algorithmic leverage rather than adding parallelism first.

## 1. Repeated M3/M4 global scans

Eliminate global role scans and concept-to-role recursive search. Use `(context_signature, action, family)` bounded role indexes and `role_id -> concept_ids` inverted indexes.

## 2. Scalar action scoring

Score all available actions as one batch. Assemble contingency, future-option, failure, role and concept features once and reuse role matches across scoring components.

## 3. Per-evidence future-option/failure reads

Maintain incremental per-action aggregates so scoring reads compact O(1) summaries rather than traversing evidence edges and performing N+1 score lookups.

## 4. Read-side synchronization and IPC

Use immutable shared/memory-mapped `MemoryReadView` generations directly from 40-100+ workers. Avoid central read RPC, Manager proxies and repeated serialization.

## 5. Broad higher-order recomputation

Use dirty dependency propagation and process only affected M2-M6 neighborhoods during normal generations. Reserve full scans for periodic validation.

## 6. Python nested loops and set-heavy algorithms

Replace family/role/concept membership scans, repeated `set` unions and row-by-row link generation with inverted indexes, sorted integer arrays, bitsets where appropriate, grouped/vectorized operations and batch insertion.

## 7. Multiprocessing copy/pickling overhead

Transfer and higher-order worker processes attach once to immutable generation arrays and receive ranges/IDs only. Do not pickle complete role rows or profile caches per chunk.

## 8. Fine-grained canonical mutations and commits

Coalesce repeated mutations and apply them through one canonical writer and one large durable transaction per flush/generation boundary. Parallelize preprocessing, not canonical commit ownership.

## 9. Python object/string memory overhead

Replace hot object graphs with numeric IDs, structure-of-arrays layouts and compact segments to reduce cache misses and permit vectorization/shared-memory reads.

## 10. Candidate comparison compute

Only after the previous changes are measured should GPU acceleration be introduced for remaining bounded candidate-ranking kernels.

# 27. Explicit Performance Targets

These are engineering targets, not assumptions.

| Area | Target |
|---|---|
| Hot contingency lookup | O(1) indexed |
| Future-option action evidence | O(1) aggregate per action |
| Failure evidence | O(1) aggregate per action |
| Role lookup | O(K) bounded candidates, not O(all roles) |
| Concept lookup | proportional to matched roles/concepts, not all concepts |
| Read locking | none on published generation |
| Worker read RPC | none in normal same-host hot path |
| Durable writer count | 1 |
| Mutation transactions | batched, >=10x fewer than equivalent fine-grained path |
| Higher-order derivation | dirty/bounded neighborhoods |
| Candidate cross-products | forbidden |
| Read-view publication | atomic generation swap |
| RAM representation | measured bytes/record, progressively compact |
| GPU | only if CPU batch/index path remains bottleneck |

---

# 28. Final Design Position

The clean-break v7 design is intentionally different from both v6 and the earlier v7.0 proposal.

It is not:

> SQLite semantics hidden behind a generic store.

It is not:

> one central Python service handling every read and write.

It is not:

> the same graph algorithms moved into RAM.

It is:

> **a generation-oriented memory system with a single canonical mutation owner, immutable parallel read views, compact active semantic state, a separate evidence ledger, batch-first APIs, direct cognition indexes, incremental derivation, and bounded candidate computation.**

The performance strategy is equally important:

> **first remove unnecessary work, then batch the remaining work, then parallelize independent work, and only then accelerate the residual numeric bottleneck on GPU.**

This sequence minimizes implementation risk while maximizing the chance that late-epoch runtime approaches a bounded marginal cost rather than scaling with the entire lifetime memory.
