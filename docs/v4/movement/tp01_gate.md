Status: implemented and verified
Scope: movement doc: tp01 gate
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `tp01` Gate

## Purpose

Phase 3C proves that the exact movement solver handles `tp01` symmetric teleporter pairs inside the Stage 2 loop.

## Mechanic Scope

- cardinal movement only
- symmetric teleporter pairs
- immediate warp on teleporter entry
- yellow target completion after movement and any warp resolution

## Required Typed-State Additions

- teleporter endpoint positions
- exact teleporter pair list
- explicit directional pair map for both warp directions
- directly derived target cells

## Required Transition Semantics

- legal primitive movement reaches a tentative destination
- entering a teleporter endpoint warps immediately to its paired endpoint
- either endpoint of a pair uses the same rule
- goal completion is checked after warp resolution
- walls continue to block before warp logic runs

## Required Search-State Representation

- avatar position
- teleporter endpoint set
- exact teleporter pair map

`tp01` does not add hidden mutable mechanic bits beyond the fixed pair map, so the search state is usually position plus the explicit teleporter mapping.

## Required Solver-Policy Behavior

- build `tp01` typed state exactly from Stage 2 parsed state plus environment-backed config
- search over teleporter-resolved successors only
- emit one legal primitive action or a short plan prefix
- replan after every executed action

## Pass Criteria

- exact `tp01` typed-state build succeeds on at least one verified real level
- transition-model checks match real local warp semantics
- search returns legal teleporter-using plans where needed
- `MovementSolverPolicyV4` solves at least one verified real `tp01` level end to end inside the Stage 2 loop
- no legacy `v3_1` runtime surfaces are active
- no LLM, VLM, RL, blackboard, POI, hypothesis, or mechanic-graph dependency appears in the live action loop

## Failure Buckets

- builder
- adapter
- transition
- search
- policy

## Non-Goals

- one-way teleporters
- single-use teleporters
- hidden teleporter state
- learned heuristics
- branch or merge runtime behavior

## Exit Rule For Moving To `ic01`

`ic01` work starts only after the `tp01` gate passes. `tp01` is the proof that the solver handles exact non-local spatial transitions correctly before adding slide-ray movement.
