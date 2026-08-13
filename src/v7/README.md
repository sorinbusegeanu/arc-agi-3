# v7 clean foundation

This package is a clean-break implementation of the v7 memory architecture. It intentionally has no runtime or schema compatibility layer with `src/v6`.

Implemented:

- uint64-style numeric memory IDs and M0-M6 levels;
- single-owner `CanonicalMemoryWriter` with batch coalescing and dirty sets;
- atomic batch validation for node and edge mutations;
- immutable generation-specific `MemoryReadView` snapshots;
- v7-native SQLite schema and single-writer durable generation store;
- separate append-only `EvidenceStore`;
- bounded CPU `CandidateProvider` contract;
- cognition-oriented indexes for contingencies, bounded roles and role-to-concept expansion;
- incremental per-action future-option/failure/contradiction aggregates;
- batch `score_inputs()` retrieval across all available actions;
- generation prepare/finalize/abort lifecycle;
- durable-before-publish `GenerationCommitCoordinator`;
- monotonic `GenerationPublisher` with reader refresh-by-generation semantics;
- transport contract plus direct-reference local transport for single-process use;
- file-backed `MmapReadViewTransport` for cross-process immutable generation attachment without Manager proxies or read RPCs;
- content-addressed read-view files with digest validation and explicit generation release;
- retry-safe publication after a durable generation has already committed;
- incremental M1->M6 dependency registration and dirty-neighborhood propagation;
- deterministic per-level `DirtyDerivationPlan` snapshots and consumption;
- read-only compact node and score columns backed by numeric buffers;
- packed typed adjacency using source/relation rows plus offset/length target segments;
- `MemoryReadView` hot node/score/neighbor reads routed through compact arenas;
- deterministic bounded `DerivationTaskPlanner` chunks for dirty M2-M6 work;
- worker-side generation attachment once per worker and task descriptors containing only levels and MemoryIds.

Current mmap transport still reconstructs the immutable Python read model on attach before compact arenas are built. The next physical-layout step is to mmap the numeric arena segments directly so worker hot arrays become zero/low-copy rather than reconstructed.

Next implementation slices:

- direct mmap/shared numeric arena segments;
- vectorized/batched higher-order derivation kernels consuming `DerivationTask` chunks;
- deterministic merge of derived mutation batches back into the single canonical writer;
- GPU candidate ranking only after residual CPU bottlenecks are measured;
- remote samplers later.

Scientific M1-M6 derivation semantics are intentionally not copied wholesale from v6. Stabilized v6 behavior remains the scientific oracle while v7 infrastructure and execution architecture evolve independently.
