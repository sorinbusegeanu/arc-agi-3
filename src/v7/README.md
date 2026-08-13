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
- content-addressed segmented mmap transport for nodes, scores, adjacency and cognition indexes;
- mmap-backed packed cognition indexes;
- incremental M1->M6 dependency registration and dirty-neighborhood propagation;
- deterministic bounded derivation tasks and process execution;
- deterministic derived-batch merge into the canonical writer;
- configurable retention fitness, promotion, demotion and replay;
- append-only lifecycle evidence, provenance, transfer trials and contradictions;
- empirical concept validation kept separate from transfer priors.

Integration block completed:

1. Durable runtime snapshots restore committed generation state, canonical IDs, nodes, scores, edge support, cognition indexes, dependency graph and lifecycle flags.
2. `V7Runtime` and `arc-agi3-v7`/`python -m v7` provide an end-to-end JSONL episode ingestion and generation-commit path with restart support.
3. Strict H01-H12 evidence contracts enforce missing evidence and proxy-only evidence as `INSUFFICIENT_EVIDENCE` while preserving raw decision, quality gate and dependency gate separately.
4. Differential scientific artifact comparison operates on exported artifacts without importing the v6 runtime.
5. Performance validation covers memory size, generation commit latency, derivation throughput, action-selection latency, mmap attach latency and parallel derivation throughput, with like-for-like baseline comparison.

Unit coverage for these blocks is under `src/v7/tests/test_v7_integration_blocks.py`.

Scientific M1-M6 behavior is clean-break v7 code and does not import v6 runtime modules. Stable v6 behavior remains an external scientific reference for differential validation.

Remaining work is experiment integration and measured validation: execute the v7 unit/runtime suite, feed real ARC-AGI-3 environment traces, generate H01-H12 evidence from those runs, and collect like-for-like v6/v7 performance measurements. GPU ranking and remote samplers remain later optional phases.
