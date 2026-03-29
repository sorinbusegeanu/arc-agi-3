Status: implemented and verified
Scope: click perception doc: pt01 gate
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Purpose**
`pt01` is the first Phase 4 hard gate. It proves exact click-state build, exact click transition semantics, exact short-horizon search, and Stage 2 loop integration for a deterministic click family.

**Gate Conditions**
The gate passes only when:
- `pt01` typed-state build passes
- `pt01` transition-model tests pass
- `pt01` search and selection tests pass
- `ClickSolverPolicyV4` solves at least one verified real `pt01` level end to end in the Stage 2 loop
- no forbidden runtime dependencies are present

`sy01` work begins only after this gate passes.
