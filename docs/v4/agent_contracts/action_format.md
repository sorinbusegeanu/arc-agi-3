Status: implemented and verified
Scope: agent contracts doc: action format
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Action Contract

## Canonical Action Record

The v4 action record is the smallest stable representation of one action request sent across the solver boundary.

It contains:

- a primitive discrete action id
- the exact engine enum name for that id
- an optional payload object
- an optional reasoning or annotation sidecar

## Primitive Discrete Action Id And Enum Name

The local engine exposes these authoritative `GameAction` members:

- `RESET` with id `0`
- `ACTION1` with id `1`
- `ACTION2` with id `2`
- `ACTION3` with id `3`
- `ACTION4` with id `4`
- `ACTION5` with id `5`
- `ACTION6` with id `6`
- `ACTION7` with id `7`

No additional authoritative action names are allowed in `v4`.

## Optional Payload Object

Simple actions carry no required payload beyond optional engine fields such as `game_id`.

The only complex action exposed by the local engine is `ACTION6`, whose payload schema is:

- `x: int`
- `y: int`
- optional `game_id: str`

Locally, `x` and `y` are validated by the engine in the inclusive range `0..63`.

## Reasoning Or Annotation Sidecar

The local engine `ActionInput` supports an optional `reasoning` blob. In `v4`, that field is preserved only as a non-authoritative sidecar.

It may be stored for audit or debugging, but it must never be treated as authoritative environment state or as proof that an action was legal, useful, or intended by the environment.

## Authoritative Action Fields

This section is normative.

The authoritative action fields are only:

- action id
- action enum name
- payload, when required by the engine for the chosen action

The reasoning sidecar is not authoritative.

## Action Legality Rule

Action legality is based only on `available_actions` returned by the current observation.

An action is legal if and only if:

- its action id appears in the observation’s `available_actions`
- its payload matches the engine schema for that action id

Legality must not be inferred from planner preferences, prior episodes, action aliases, or wrapper-level heuristics.

## Action Serialization Rules

The canonical serialized shape is:

- `action_id`
- `action_name`
- `payload`
- `reasoning`

Serialization must preserve:

- the exact integer action id
- the exact engine enum name
- the payload object exactly as supplied after validation
- null for absent payload rather than a guessed default payload object

## Invalid-Action Handling Contract

If an action is not present in `available_actions`, or if its payload violates the engine schema, the contract must reject it before treating it as an authoritative executed action.

If a wrapper still attempts execution and receives a resulting observation, the transition record must distinguish:

- the requested action
- legality status
- whether execution was attempted

The contract must not rewrite the invalid action into a nearby legal action.

## Reset Handling Contract

The local engine exposes `RESET` as action id `0`.

The local engine also exposes `full_reset` on the returned frame, and its reset path distinguishes full-game reset from level reset internally. `v4` therefore allows reset handling to record whether the resulting observation reports `full_reset = true` or `false`.

`v4` does not invent separate reset action names beyond the engine’s single `RESET` action.

## Explicit Prohibitions

- No invented action names beyond `RESET` and `ACTION1` through `ACTION7`.
- No synthetic legality rules beyond observation-exposed `available_actions`.
- No guessed payload for non-coordinate actions.
- No inferred grid-coordinate reinterpretation of `ACTION6` payload unless the environment itself exposes that mapping.
