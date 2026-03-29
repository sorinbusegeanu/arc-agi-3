Status: implemented and verified
Scope: movement doc: ic01 gate
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ic01` Gate

## Purpose

Phase 3D proves exact solving for the real local `ic01` ice-slide family on top of the Stage 2 loop.

## Mechanic Scope

- one primitive movement action starts one straight-line slide
- slide continues until the next cell would be out of bounds, a gray wall, or a red hazard
- the avatar stops on the last legal cell before that blocker
- the yellow goal is not a blocker
- no LLM, VLM, RL, blackboard, POIs, hypotheses, or mechanic graph are allowed in the action loop

## Required Typed-State Additions

- explicit slide mode
- exact traversable slide cells
- explicit wall positions
- explicit red hazard positions
- exact goal cells when directly derivable

## Required Transition Semantics

- one action resolves to one deterministic landing cell
- stop-before-OOB semantics
- stop-before-wall semantics
- stop-before-red-hazard semantics
- goal handling is evaluated from the final landing cell only

## Required Search-State Representation

- avatar position
- exact traversable slide surface
- explicit blocking wall and hazard cells
- goal cells

## Required Solver-Policy Behavior

- build exact `ic01` typed state from Stage 2 parsed state
- search over slide-resolved successors only
- return only legal primitive actions or a short executable prefix
- replan after every executed action

## Pass Criteria

- `ic01` typed-state build passes on at least one verified real level
- `ic01` transition-model tests match checked real env semantics
- `ic01` search uses slide-resolved landing states
- `MovementSolverPolicyV4` solves at least one verified real `ic01` level inside the Stage 2 loop
- no legacy `v3_1` runtime surfaces are active
- no forbidden runtime dependencies are present

## Failure Buckets

- builder
- adapter
- transition
- search
- policy

## Non-Goals

- torus wrapping
- capped slides
- mixed friction surfaces
- any ice variant not present in the real local `ic01` implementation

## Exit Rule For Moving To `va01`

`va01` work begins only after the `ic01` gate passes. `ic01` is the proof that the solver handles exact forced multi-cell deterministic motion before adding coverage-history state.
