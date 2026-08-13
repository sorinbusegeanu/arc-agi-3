# v7 clean foundation

This package is a clean-break implementation of the v7 memory architecture. It intentionally has no runtime or schema compatibility layer with `src/v6`.

Implemented in the first foundation slice:

- uint64-style numeric memory IDs and M0-M6 levels;
- single-owner `CanonicalMemoryWriter` with batch coalescing and dirty sets;
- immutable generation-specific `MemoryReadView` publication;
- v7-native SQLite schema and single-writer durable generation store;
- separate append-only `EvidenceStore`;
- bounded CPU `CandidateProvider` contract;
- generation metadata and deterministic publication.

Not implemented in this slice:

- v6 scientific derivation semantics for M1-M6;
- shared-memory/mmap read-view transport;
- compact columnar arenas;
- incremental M2-M6 dependency propagation;
- action-scoring indexes;
- GPU candidate ranking;
- remote samplers.

Those are intentionally staged after the architectural contracts are stable and, for scientific logic, after the corresponding v6 behavior is stable enough to serve as an oracle.
