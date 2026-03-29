Status: implemented and verified
Scope: tests doc: test tt01 contracts spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `tt01` Contract Test Spec

The `tt01` contract tests must cover:

- valid reset observation
- valid legal movement step
- valid transition record
- valid step result
- raw payload preservation
- monotonic step index
- no fabricated coordinate or interaction payloads unless the real local game exposes them
- terminal mapping preservation if encountered

Additional requirement:

- all legality checks must come from the actual `available_actions` surface
