Status: implemented and verified
Scope: agent contracts doc: per step result record
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Per-Step Result Record

## Purpose

The per-step result record is a compact derived record built from a transition record. It stays close to the environment truth surface while dropping full observation payloads when a small execution summary is enough.

## Relationship To Observation And Transition Record

The per-step result record is derived from one authoritative transition record.

It summarizes:

- what action was executed
- whether the action was legal
- what raw environment state changed from and to
- whether the step ended the episode
- whether reset is now required

## Minimal Schema

- `action`
- `action_legal`
- `terminal_signal`
- `raw_state_before`
- `raw_state_after`
- `levels_completed_delta`
- `win_levels_delta`
- `reset_required`
- `coordinate_payload`

## Authoritative Fields

These are authoritative because they come directly from the transition record’s authoritative observations and executed action:

- executed action id and enum name
- coordinate payload, if present
- legality status
- raw state before
- raw state after
- terminal signal derived from raw state
- reset required flag
- raw `levels_completed` delta
- raw `win_levels` delta

## Derived-But-Allowed Fields

The record may derive simple arithmetic deltas from directly observed numeric fields, such as:

- `levels_completed_delta`
- `win_levels_delta`

Those remain allowed because they are direct arithmetic on authoritative numeric observations, not inferred semantics.

## Prohibited Fields

The per-step result record must not include:

- POI hits
- candidate rankings
- planner rationale
- hypothesis labels
- mechanic-edge labels
- reward shaping terms unless stored in a separate non-authoritative analysis extension
- solver beliefs
- symbolic promotions
