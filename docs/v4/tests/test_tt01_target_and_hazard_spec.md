Status: implemented and verified
Scope: tests doc: test tt01 target and hazard spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `tt01` Target-And-Hazard Test Spec

These tests must be grounded in the real local `tt01` game definition.

They must verify, using only direct environment truth:

- there is at least one observable task structure beyond pure movement
- a deterministic local action script or exact short policy can make real progress toward the target condition
- hazard or blocking structure is real and not inferred from heuristics
- a clearly bad path or hazard-contact or blocked path does not falsely count as success
- terminal success or failure comes only from the environment state

The tests must not use:

- POIs
- blackboard state
- hypotheses
- chain objectives
- score-only success inference

Note:

- if the real local game does not expose terminal failure on hazard contact, the tests must reflect the actual behavior and treat hazards as real blocking structure rather than inventing a failure mode
