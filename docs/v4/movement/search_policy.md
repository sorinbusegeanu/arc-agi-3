Status: implemented and verified
Scope: movement doc: search policy
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Search Policy

## Purpose

Phase 3 search chooses exact movement plans over typed solver states.

## Search Algorithms

- BFS for exact shortest-path search on small spaces
- A* only when a clearly admissible heuristic exists
- bounded search for larger spaces or smoke runs

## Plan Representation

- ordered primitive action ids
- small interruptible action prefixes only

## Action-Prefix Execution Rule

- the solver may return one primitive action
- or a short plan prefix
- the runtime still executes at most one action before the next observe-parse-decide boundary

## Replan Rule

- replan after every executed action
- do not execute long fixed plans blindly

## Legality Rule

- returned actions must come from the current legal movement-action surface
- search expands actions in deterministic order

## Failure Handling

- `found`
- `exhausted`
- `bound_exhausted`

When search fails, the policy may fall back to a deterministic legal primitive action for bounded proving runs.

## Non-Goals

- no planner-ranker bundles
- no opaque candidate trees in the public policy payload
- no environment stepping during search
