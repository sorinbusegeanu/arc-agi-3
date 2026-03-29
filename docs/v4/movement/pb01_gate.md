Status: implemented and verified
Scope: movement doc: pb01 gate
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `pb01` Gate

## Purpose

Phase 3F proves that the Stage 2 loop can solve the real local `pb01` family with exact push-state reasoning.

## Mechanic Scope

- one player
- one pushable block
- one target pad
- wall blockers
- legal primitive movement and one-cell pushes only

## Required Typed-State Additions

- exact pushable-block position
- exact target cell
- exact wall blocker cells
- exact step limit when exposed by the local level config

## Required Transition Semantics

- ordinary legal movement when no push occurs
- legal push only when the adjacent block exists and the cell beyond it is in bounds and not blocked
- blocked push leaves avatar and block positions unchanged
- successful push updates avatar and block positions exactly once
- completion occurs only when the block reaches the target cell

## Required Search-State Representation

- avatar position
- pushable-block position
- fixed blockers and bounds
- target cell
- any exposed step-budget constraint needed for exact failure modeling

## Required Solver-Policy Behavior

- build exact `pb01` typed state from real observation and environment config
- search over push-state successors, not avatar position only
- return only legal primitive action or short plan prefix
- replan after each executed action
- use no LLM, VLM, or RL component in the action loop

## Pass Criteria

- `pb01` typed-state build passes on at least one verified real local level
- `pb01` transition-model checks match real environment push semantics
- search distinguishes equal avatar positions under different block positions
- `MovementSolverPolicyV4` solves at least one verified real `pb01` level end to end inside the Stage 2 loop
- no legacy `v3_1` runtime surfaces are active

## Failure Buckets

- builder
- adapter
- transition
- search
- policy

## Non-Goals

- multi-block sokoban
- pulling
- deadlock heuristics beyond exact local search
- non-movement reasoning layers

## Exit Rule For Phase 3 Movement Consolidation

Phase 3 movement consolidation begins only after the `pb01` gate passes alongside the earlier `ul01`, `fs01`, `tp01`, `ic01`, and `va01` gates.
