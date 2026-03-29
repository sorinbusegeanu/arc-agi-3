Status: implemented and verified
Scope: movement doc: va01 gate
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `va01` Gate

## Purpose

Phase 3E proves exact solving for the real local `va01` coverage family on top of the Stage 2 loop.

## Mechanic Scope

- the objective is to visit every walkable cell at least once
- revisits are allowed
- walls block motion and shrink the coverage-eligible cell set
- there is no separate exit goal beyond full coverage
- no LLM, VLM, RL, blackboard, POIs, hypotheses, or mechanic graph are allowed in the action loop

## Required Typed-State Additions

- explicit coverage-eligible cell set
- explicit current covered-state set
- exact wall layout
- start-cell coverage inclusion, because the real game marks the start visited immediately

## Required Transition Semantics

- one primitive movement action resolves to one deterministic neighbor move or block
- entering a new eligible cell adds it to the covered set
- revisiting a covered cell leaves coverage unchanged
- completion is reached exactly when covered cells equal coverage-eligible cells

## Required Search-State Representation

- avatar position
- coverage-eligible cell set
- current covered-state set
- blocking wall cells

## Required Solver-Policy Behavior

- build exact `va01` typed state from Stage 2 parsed state
- search over coverage-state successors, not position-only successors
- return only legal primitive actions or a short executable prefix
- replan after every executed action

## Pass Criteria

- `va01` typed-state build passes on at least one verified real level
- `va01` transition-model tests match checked real env semantics
- `va01` search uses coverage-state successors
- `MovementSolverPolicyV4` solves at least one verified real `va01` level inside the Stage 2 loop
- no legacy `v3_1` runtime surfaces are active
- no forbidden runtime dependencies are present

## Failure Buckets

- builder
- adapter
- transition
- search
- policy

## Non-Goals

- revisit-forbidden path variants
- hazard-bearing coverage variants
- any separate-goal coverage family not present in the real local `va01` implementation

## Exit Rule For Moving To `pb01`

`pb01` work begins only after the `va01` gate passes. `va01` is the proof that the solver handles exact visited-state and coverage-state planning before adding movable-object state.
