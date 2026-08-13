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
- deterministic per-level `DirtyDerivationPlan` snapshots and consumption.

Current mmap transport reconstructs the immutable Python read model from a memory-mapped generation blob on attach. The next physical-layout phase will replace object reconstruction in hot arrays with compact columnar/offset structures while preserving the same transport/publication contract.

Next implementation slices:

- compact columnar arenas and packed adjacency;
- zero/low-copy numeric read-view segments for hot worker queries;
- incremental derivation workers consuming dirty plans;
- vectorized/batched higher-order derivation kernels;
- GPU candidate ranking only after residual CPU bottlenecks are measured;
- remote samplers later.

Scientific M1-M6 derivation semantics are intentionally not copied wholesale from v6. Stabilized v6 behavior remains the scientific oracle while v7 infrastructure and execution architecture evolve independently.
