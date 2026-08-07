# Scope and remaining integration

Implemented in this drop-in:

- strict H01-H12 contracts and standardized decision envelopes;
- explicit DERIVE then read-only REPORT execution;
- additive universal identity, edge lifecycle, promotion audit and lifecycle schemas;
- persistent context-split lineage and strategy-reuse tables;
- bounded in-memory graph insertion;
- contradiction split proposals;
- reusable M6 strategy records and retrieval through `MemoryController`;
- one golden decision-envelope fixture for every H.

Not forced by this drop-in:

- `memory_query_enabled` and `memory_action_selection_enabled` remain opt-in;
- `V6System` continues to use its existing query, promotion and lifecycle objects
  directly; the new controller is an available facade rather than a mandatory
  replacement;
- historical repair modules already loaded by the repository's existing
  `sitecustomize.py` are not rewritten into every large evaluator/substrate
  module by this archive.

These are compatibility choices, not silent claims of full integration.
