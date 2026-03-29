# Stage 2 Runtime Package

## Purpose

`src/v4/runtime` implements the Stage 2 short closed loop.

## Runtime Loop

The loop is:

1. observe
2. parse current state
3. choose action or short plan
4. execute one action
5. record transition
6. derive step result
7. update local memory
8. append ledger
9. evaluate stop condition
10. repeat

## Public Surface

- `EnvSessionV4`
- `LoopControllerV4`
- `StepLedgerRecordV4`
- `SessionSummaryV4`
- stop-condition entry points

## Boundaries

- environment-facing records come only from `v4.agentContract`
- parsed state comes only from `v4.state`
- control decisions come only from `v4.policy`
- session-local memory comes only from `v4.memory`

## What Is Explicitly Excluded

- blackboard
- POI extraction
- mechanic graph
- hypotheses
- ranker
- durable memory
- probe or directed branching
- postrun export logic in the runtime path

## Dependency Rules

- runtime may depend on `v4.agentContract`, `v4.state`, `v4.policy`, and `v4.memory`
- runtime must not depend on v3.1 blackboard or hypothesis runtime surfaces

