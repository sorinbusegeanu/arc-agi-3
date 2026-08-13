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
- transport contract plus direct-reference local transport for publication tests and single-process use;
- retry-safe publication after a durable generation has already committed.

Next implementation slices:

- shared-memory/mmap read-view transport replacing the local direct-reference transport for multi-process sampling;
- incremental M1->M6 dirty dependency propagation;
- compact columnar arenas and packed adjacency;
- vectorized/batched higher-order derivation kernels;
- GPU candidate ranking only after residual CPU bottlenecks are measured;
- remote samplers later.

Scientific M1-M6 derivation semantics are intentionally not copied wholesale from v6. Stabilized v6 behavior remains the scientific oracle while v7 infrastructure and execution architecture evolve independently.
