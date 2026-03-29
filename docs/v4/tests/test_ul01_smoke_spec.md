Status: implemented and verified
Scope: tests doc: test ul01 smoke spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ul01` Smoke Test Spec

## Purpose

These tests prove that the Stage 2 runtime can run the real local `ul01` environment in one continuous live session while preserving the v4 contract boundary.

## Setup

The tests must:

- instantiate the real local `ul01` environment
- build the Stage 2 v4 loop
- use one live environment instance for the whole session
- run a short bounded session

## Assertions

The tests must assert that:

- reset yields a valid `V4Observation`
- executed steps produce valid `V4TransitionRecord` objects
- executed steps produce valid `V4StepResult` objects
- the ledger appends exactly one record per executed action
- the session remains one continuous live session
- success is not assumed yet

## Excluded Legacy Behavior

The tests must verify that no legacy v3.1 runtime machinery is involved:

- no round runner
- no blackboard merge
- no hypothesis registry
- no chain manager
- no probe or directed split
- no durable flush path

## Failure Localization Expectations

If the smoke run fails, the failure must localize to one Stage 2 bucket only, not to a combined symbolic or branch-selection diagnosis.
