Status: implemented and verified
Scope: movement doc: phase3 movement consolidation
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Phase 3 Movement Consolidation

## Purpose

This document defines the consolidation gate after all six Phase 3 movement families have individual hard gates.

## Covered Families

- `ul01`
- `fs01`
- `tp01`
- `ic01`
- `va01`
- `pb01`

## What Is Now Proven

- exact typed-state construction for each supported movement family
- exact local transition semantics for each family
- exact search-state expansion beyond avatar position where the mechanic requires it
- Stage 2 loop compatibility with live replanning after every executed action
- absence of legacy `v3_1` branch, merge, hypothesis, and blackboard machinery in the movement path

## What Is Still Not Covered

- non-movement families
- learned heuristics or learned policies
- durable memory coupling
- planner abstractions outside the exact movement solver path
- larger optimization work beyond the proved local gates

## Required Consolidation Test Runs

- full movement suite
- isolated v4 suite
- Phase 3 family regression matrix
- Phase 3 movement consolidation gate
- focused Stage 2 closed-loop regression checks

## Required Gate Set

- `ul01` gate
- `fs01` gate
- `tp01` gate
- `ic01` gate
- `va01` gate
- `pb01` gate

## Regression Requirements

- all family gates must pass together in one run
- no Stage 2 closed-loop regression is allowed
- no forbidden runtime dependency may appear in any movement path
- deterministic equal-input behavior must remain stable

## Exit Rule To Phase 4

Phase 4 begins only after the full Phase 3 movement consolidation gate passes.
