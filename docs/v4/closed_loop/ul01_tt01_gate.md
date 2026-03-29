Status: implemented and verified
Scope: closed loop doc: ul01 tt01 gate
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 `ul01` / `tt01` Gate

## Purpose

This gate defines the next Stage 2 proving step after the `ez01` through `ez04` tutorial games. It requires the Stage 2 runtime to remain contract-correct and single-loop while handling slightly richer real environment structure.

## Why `ez01`–`ez04` Were Insufficient

The tutorial directional games proved:

- real environment loading
- valid authoritative observation, transition, and step-result handling
- single-loop runtime behavior
- deterministic movement-only win coverage

They did not prove:

- dependency ordering
- task structure beyond pure directional walking
- collection structure with blocking layout constraints
- rejection of bad or incomplete action orderings in a nontrivial game

## What `ul01` Must Prove

`ul01` must prove that the Stage 2 loop can handle movement plus a real dependency sequence from direct environment truth only.

Specifically it must show:

- the environment exposes a key-door dependency structure in live observations
- a deterministic local action script can satisfy the true dependency order
- an incorrect order, such as trying to pass the door before collecting the key, does not falsely succeed
- the runtime still produces valid authoritative records on every executed step
- no legacy v3.1 branch, merge, hypothesis, or durable runtime path is active

## What `tt01` Must Prove

`tt01` must prove that the Stage 2 loop can handle movement plus observable target-collection structure and blocking hazard or obstacle structure from direct environment truth only.

Specifically it must show:

- there is real task structure beyond pure directional walking
- a deterministic local script can make real progress toward target collection
- clearly bad or blocked paths do not falsely count as success
- authoritative terminal handling remains grounded in environment state only
- the runtime stays in one continuous live session with one ledger record per executed action

## Pass Criteria

This gate passes only when:

- all `ul01` and `tt01` Stage 2 smoke and contract tests pass
- at least one verified deterministic dependency-sequence success path passes for `ul01`
- at least one verified deterministic target-progress or success path passes for `tt01`
- invalid-action, corrupted-observation, and terminal-mapping mismatch tests pass
- no legacy v3.1 runtime path is active
- every executed step yields valid observation, action, transition, and step-result records
- failures remain localizable to one Stage 2 bucket

## Failure Buckets

The active failure buckets remain the Stage 2 buckets:

- observation acquisition
- state parsing
- action selection
- action execution
- transition building
- step-result derivation
- local-memory update
- stop-condition handling

## Non-Goals

This gate does not require:

- POI extraction
- blackboard state
- mechanic graphs
- hypotheses
- chain objectives
- durable memory
- Phase 3 planner behavior

## Exit Rule For Moving To Phase 3

Phase 3 must not begin until the `ul01` / `tt01` gate passes in full. Passing requires both richer real-game families and the fault-injection rejection tests to be green.
