Status: implemented and verified
Scope: closed loop doc: loop invariants
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 Loop Invariants

## Invariant List

- one live environment only
- one authoritative observation per step boundary
- one authoritative action per executed step
- one authoritative transition record per step
- one authoritative step result per step
- memory update only after step result exists
- no inferred symbolic structure in authoritative state
- no branch winner selection
- no merge or reconcile phase
- every failure must be localizable to one stage

## Rationale

These invariants keep the Stage 2 runtime narrow enough to verify directly against the environment contract. They also prevent v3.1-style branching, merge, and symbolic promotion logic from re-entering the control path.

## Violation Handling

If an invariant is violated:

- the runtime must fail closed
- the failure must be assigned to one Stage 2 failure bucket
- the ledger must record the stop status and failure bucket

## Test Expectations

Tests must prove:

- only one live env instance is used
- one step creates exactly one transition and one step result
- memory updates happen after step results
- no branch or merge API exists in the public runtime surface
- failures can be localized to one Stage 2 bucket

