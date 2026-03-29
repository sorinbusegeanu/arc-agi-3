Status: implemented and verified
Scope: tests doc: test tt01 smoke spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `tt01` Smoke Test Spec

## Purpose

These tests prove that the Stage 2 runtime can run the real local `tt01` environment in one continuous live session while preserving valid authoritative records.

## Required Test Properties

The tests must:

- instantiate the real local `tt01` environment
- run the Stage 2 v4 loop in one continuous live session
- produce valid observation, transition, and step-result records
- use no legacy v3.1 runtime surfaces
- append one ledger record per executed action
- not assume success yet

## Additional Note

`tt01` must exercise nontrivial choice structure beyond the tutorial directional games.
