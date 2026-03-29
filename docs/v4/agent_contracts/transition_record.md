Status: implemented and verified
Scope: agent contracts doc: transition record
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Transition Record

## Purpose

The v4 transition record is the authoritative environment-facing record for one executed action. It exists for replay, debugging, and audit without embedding planner belief state.

## Schema

The record contains:

- pre-action observation
- chosen action
- post-action observation
- legality status
- execution status
- terminal signal
- optional step index
- optional locally generated timestamp

## Required Fields

- `pre_observation`
- `action`
- `post_observation`
- `action_legal`
- `execution_status`
- `terminal_signal`

## Optional Fields

- `step_index`
- `timestamp_ms`

These optional fields are allowed only if they are directly available from the wrapper or are locally generated bookkeeping fields. They are not environment truth.

## Invariants

This section is normative.

- Pre and post observations are raw-authoritative snapshots or references to them.
- Action is exactly the executed action.
- Transition record does not store promoted symbolic interpretation as fact.
- Missing fields must remain null or absent rather than guessed.
- `terminal_signal` is derived from the raw post-observation state.
- `action_legal` is determined only from the pre-observation `available_actions` plus payload validation.

## Replay Requirements

The record must support deterministic replay of the environment-facing exchange by preserving:

- the exact pre-observation snapshot
- the exact executed action id and payload
- the exact post-observation snapshot

Replay support does not require planner beliefs, hypotheses, candidate sets, or symbolic summaries.

## Prohibited Inferred Fields

The transition record must not contain:

- planner rationale
- POI hits
- hypothesis labels
- mechanic-edge labels
- promoted entities
- predicted reward
- candidate rankings
- symbolic chain state
