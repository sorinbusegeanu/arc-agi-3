Status: implemented and verified
Scope: tests doc: test terminal mapping mismatch spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Terminal-Mapping Mismatch Test Spec

The tests must cover:

- raw environment state indicates non-terminal but an injected path tries to mark terminal
- raw environment state indicates terminal success but the derived result disagrees
- raw environment state indicates terminal failure but the derived result disagrees
- score or progress deltas alone never create terminal state

The tests must assert that:

- the validator rejects inconsistent terminal derivation
- source attribution is preserved in the error
- the step-result record cannot persist an invalid terminal mapping as authoritative truth
