Status: implemented and verified
Scope: tests doc: test ul01 contracts spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ul01` Contract Test Spec

These tests must cover:

- valid reset observation
- valid legal movement step
- valid interaction step only if the real local game exposes that action
- valid transition record
- valid step result
- raw payload preservation
- monotonic step index
- terminal mapping preservation if success or failure occurs during the test
- no fabricated coordinate action payloads

Additional requirements:

- available actions must come only from the real environment surface
- legality checks must use the actual current `available_actions`
- if the local game exposes movement only, the tests must not invent an interaction action
