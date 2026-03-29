Status: implemented and verified
Scope: Phase 3 movement overview
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Phase 3 Overview

## Purpose

Phase 3 adds an exact movement-family solver on top of the Stage 2 single-session loop.

## Scope

- exact typed-state solving for movement-first local families
- deterministic family adapters
- deterministic local transition modeling
- exact or bounded exact search
- Stage 2 policy integration through one action-at-a-time replanning

## Supported Game Families

- `ul01`
- `fs01`
- `tp01`
- `ic01`
- `va01`
- `pb01`

## Excluded Game Families

- non-movement symbolic tasks
- language-conditioned tasks
- open-ended VLM tasks
- RL-trained control policies

## Typed Movement State

Phase 3 introduces `MovementTypedStateV4`, a solver-state layer derived from Stage 2 parsed state. It keeps direct control fields separate from family option fields such as key bits, switch activation bits, teleporter pairs, coverage masks, and push-block positions.

## Family Adapters

Each supported family has one explicit adapter. Adapters consume `ParsedStateV4`, use direct observations plus known local family layouts, and fail closed when required state cannot be reconstructed.

## Exact Transition Model

`MovementTransitionModelV4` applies one primitive movement action to one typed state. It handles blocking, key pickup, door opening, switch activation, teleport jumps, ice sliding, coverage updates, and one-block pushing without stepping the live environment.

## Search Layer

Phase 3 uses BFS by default for exact shortest-path search on small spaces, supports A* when a clearly admissible heuristic is available, and supports bounded exact search for larger spaces.

## Solver Policy

`MovementSolverPolicyV4` is a Stage 2-compatible policy head. It builds a family typed state, runs exact search, returns one primitive action or a short plan prefix, and replans after every executed action.

## Integration With Stage 2 Loop

- one live environment instance per session remains unchanged
- no probe pass
- no directed pass
- no branch execution
- no merge or reconcile phase
- at most one action is executed before the loop re-enters observation and parsing

## Failure Buckets

- typed-state build
- family adapter reconstruction
- transition application
- exact search
- policy decision packaging

At the Stage 2 runtime boundary, these surface as action-selection failures unless the environment-facing contract fails earlier.

## Success Criteria

- exact typed-state reconstruction for each supported family
- exact transition semantics on checked local env cases
- exact or bounded search returns legal primitive plans
- `ul01` solves at least one verified real level end to end in the Stage 2 loop
- later family gates pass in rollout order

## Non-Goals

- no LLM in the action loop
- no VLM in the action loop
- no RL in the action loop
- no blackboard
- no POIs
- no hypotheses
- no mechanic graph
- no durable memory
