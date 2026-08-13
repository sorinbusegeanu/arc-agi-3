# v7 implementation rules

`src/v7/` is a clean-break architecture. Rules here override v6 compatibility guidance for files under this directory.

- Do not import from `v6` in production v7 modules.
- Do not preserve v6 SQLite table names, schemas, runtime APIs, report layouts, or migration paths.
- v6 may be used only by external differential/reference tests.
- Canonical active M1-M6 cognition is RAM-first; SQLite is durability/evidence storage, not a hot query engine.
- Keep one logical canonical writer. Parallelize preprocessing and derivation, not canonical mutation ownership.
- Published `MemoryReadView` generations are immutable and must remain safe for concurrent readers.
- Primary hot-path APIs are batch-first. Do not add scalar loops that repeatedly cross storage/RPC boundaries.
- Persist dirty generation deltas rather than rewriting complete active state each generation.
- Keep historical evidence/provenance separate from active semantic state.
- Candidate generation must be bounded before semantic validation; do not introduce unrestricted all-pairs search.
- Prefer numeric IDs and compact arrays/indexes in hot paths. Avoid repeated string/JSON processing in cognition.
- Do not add GPU acceleration until the corresponding bounded CPU path and measurement baseline exist.
- Keep scientific semantics deterministic and validate them against stable reference fixtures rather than runtime compatibility layers.
