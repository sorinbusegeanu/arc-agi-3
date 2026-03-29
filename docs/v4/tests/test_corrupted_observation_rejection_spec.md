Status: implemented and verified
Scope: tests doc: test corrupted observation rejection spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Corrupted-Observation Rejection Test Spec

The tests must cover:

- missing required authoritative field
- invalid raw state enum or unsupported raw state value
- observation payload shape mismatch
- inconsistent `available_actions` structure
- metadata mismatch between observation and environment metadata if the contract enforces it

The tests must assert that:

- parsing fails closed
- no guessed defaults are inserted
- no policy action is emitted from an invalid authoritative observation
- failure is localized to observation acquisition or state parsing
