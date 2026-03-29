Status: implemented and verified
Scope: agent contracts doc: validation errors
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Validation And Adapter Errors

## Purpose

`v4` errors exist to explain exactly why an authoritative environment-facing payload was rejected and which source field caused the rejection.

## Error Categories

- missing required field
- unknown authoritative field
- invalid enum or action id
- illegal action for current observation
- invalid coordinate payload
- invalid terminal derivation
- invalid transition record
- metadata mismatch
- adapter source incompatibility

## Validation Failure Principles

- Fail closed on authoritative input violations.
- Report the offending source field explicitly.
- Distinguish validation failures from adapter source incompatibility.
- Do not repair invalid authoritative input silently.

## Source Attribution Rule

Every validation or adapter error must identify the closest source field that triggered the failure, such as:

- `raw_payload.state`
- `available_actions[2]`
- `payload.x`
- `env.info.game_id`

## Fail-Closed Rule

If authoritative input is incomplete, contradictory, or incompatible with the contract, extraction must fail rather than guessing a replacement value.

## Logging And Reporting Expectations

- Log the error category.
- Log the source field.
- Log enough context to reproduce the failure without dumping unrelated planner state.
- Preserve the original message when re-raising from adapters into extraction boundaries.

## Non-Goals

The error model is not a planner debugging framework and not a heuristic diagnosis system. It is only for authoritative contract and adapter failures.

## Error Table

| error name | trigger | severity | recoverable yes/no | required context fields |
| --- | --- | --- | --- | --- |
| `V4MissingFieldError` | required authoritative field missing | error | no | source field, source object |
| `V4UnknownFieldError` | unknown authoritative field present in strict payload | error | no | source field, raw keys |
| `V4InvalidActionError` | unknown action id or action name mismatch | error | no | action id, action name |
| `V4IllegalActionError` | action not present in current observation `available_actions` | error | yes | action id, available actions |
| `V4InvalidPayloadError` | payload missing required keys or violates coordinate bounds | error | no | action id, payload field |
| `V4InvalidTerminalSignalError` | terminal signal does not match raw state | error | no | raw state, derived signal |
| `V4InvalidTransitionError` | transition invariants violated | error | no | action, pre state, post state |
| `V4MetadataMismatchError` | metadata contradicts authoritative observation or wrapper source | error | yes | metadata field, observed value |
| `V4AdapterError` | adapter cannot consume the source object shape | error | yes | source object type, failing field |
| `V4ValidationError` | generic contract validation failure | error | depends | source field, failing value summary |

