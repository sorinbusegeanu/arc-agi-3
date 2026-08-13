# v7 clean foundation

This package is a clean-break implementation of the v7 memory architecture. It intentionally has no runtime or schema compatibility layer with `src/v6`.

Implemented:

- uint64-style numeric memory IDs and M0-M6 levels;
- single-owner `CanonicalMemoryWriter` with batch coalescing and dirty sets;
- deterministic canonical-key resolution for duplicate discoveries from parallel workers;
- clean v7 scientific constructors for M1 contingencies, M2 families, M3 roles, M4 concepts, M5 world models and M6 strategies;
- `MemoryLearningPipeline` connecting episode evidence and higher-order derivation to canonical memory;
- atomic batch validation for node and edge mutations;
- immutable generation-specific `MemoryReadView` snapshots;
- v7-native SQLite schema and single-writer durable generation store;
- separate append-only `EvidenceStore`;
- bounded CPU `CandidateProvider` contract;
- packed cognition indexes for contingencies, bounded roles, role-to-concept expansion and action aggregates;
- vectorized action scoring over packed aggregate arrays;
- generation prepare/finalize/abort lifecycle;
- durable-before-publish `GenerationCommitCoordinator`;
- monotonic `GenerationPublisher` with reader refresh-by-generation semantics;
- direct-reference local transport for single-process use;
- legacy complete-view mmap transport for compatibility tests;
- content-addressed segmented mmap transport for nodes, scores, adjacency and cognition indexes;
- mmap-backed packed cognition indexes, avoiding JSON reconstruction on worker hot paths;
- explicit generation release with shared content-segment retention across generations;
- incremental M1->M6 dependency registration and dirty-neighborhood propagation;
- deterministic per-level `DirtyDerivationPlan` snapshots and consumption;
- read-only compact node and score columns backed by numeric buffers;
- packed typed adjacency using source/relation rows plus offset/length target segments;
- incremental arena publication that reuses unchanged node, score, adjacency and cognition sections;
- deterministic bounded `DerivationTaskPlanner` chunks for dirty M2-M6 work;
- worker-side generation attachment once per process;
- vectorized derivation inputs using dense numeric arrays for each dirty task chunk;
- bounded `ParallelDerivationExecutor` using process workers and mmap generation attachment;
- deterministic `DerivedMutationBatch` merge back into the single canonical writer independent of worker completion order.

The six implementation phases following the v7 foundation are now represented end-to-end:

1. M1-M6 scientific derivation constructors and pipeline.
2. Writer-owned deterministic canonical ID allocation.
3. Packed and mmap-backed cognition indexes.
4. Vectorized/batched cognition and action scoring.
5. Bounded parallel derivation execution with deterministic merge.
6. Incremental generation publication with unchanged section reuse and content-addressed mmap segments.

Remaining major work is runtime integration rather than foundation architecture: retention/replay/promotion-demotion lifecycle, evidence/provenance/transfer policies, durable restart/restore, v7 runner/CLI, H01-H12 reporting integration, and performance validation against v6. GPU ranking and remote samplers remain intentionally later phases.

Scientific M1-M6 behavior is clean-break v7 code and does not import v6 runtime modules. Stable v6 behavior remains an external scientific reference for differential validation.
