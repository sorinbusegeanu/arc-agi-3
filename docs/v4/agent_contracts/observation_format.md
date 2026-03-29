Status: implemented and verified
Scope: agent contracts doc: observation format
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Observation Contract

## Raw Observation Object

The raw observation object returned by the local engine is `FrameDataRaw`.

In the local engine, `reset()` and `step()` return a `FrameDataRaw` instance whose validated fields are:

- `game_id: str`
- `state: GameState`
- `levels_completed: int`
- `win_levels: int`
- `action_input: ActionInput`
- `guid: str | None`
- `full_reset: bool`
- `available_actions: list[int]`

`FrameDataRaw` also carries a runtime `frame` property backed by a private `_frame` list of numpy arrays. The local engine populates that property before returning the observation.

## Exact Local Field List

The effective local observation field set is:

- `game_id`
- `frame`
- `state`
- `levels_completed`
- `win_levels`
- `action_input`
- `guid`
- `full_reset`
- `available_actions`

No additional per-step authoritative fields are assumed.

## Mandatory Fields In v4

The v4 contract requires these fields on every observation:

- `raw_object_name`
- `raw_payload`
- `game_id`
- `frame`
- `state`
- `levels_completed`
- `win_levels`
- `full_reset`
- `available_actions`

`action_input` is required when present in the raw engine object, which is the observed local behavior.

## Passthrough Optional Metadata

These fields are preserved but remain simple passthrough metadata:

- `guid`
- `action_input.reasoning`

`guid` is authoritative only as returned metadata, not as a semantic identifier invented by solver code.

## Normalized Grid or Frame Payload

The authoritative raw frame payload is the `frame` carried by `FrameDataRaw`.

For stable internal use, v4 normalizes the frame into an ordered plane stack:

- outer dimension: frame plane index
- middle dimension: `y`
- inner dimension: `x`
- element type: integer cell or pixel value as returned by the underlying array payload

Normalization must preserve the raw payload losslessly at the content level. The raw payload must remain available alongside the normalized shape.

## Available Actions

Available actions are authoritative only when directly returned in `available_actions`.

In the local engine this is a `list[int]` of legal `GameAction` ids for the current frame. Legal action availability must not be inferred from planner state, action names, or historical heuristics.

## Score, Progress, And State Fields

The local frame object directly exposes:

- `state`
- `levels_completed`
- `win_levels`

Those are authoritative. No separate reward, truncation, or score field is part of the observed `FrameDataRaw` contract.

## Coordinate System Convention

The local frame object does not expose an explicit named coordinate-system field.

The only directly exposed coordinate payload surface is the `ACTION6` action payload with `x` and `y`. Local wrapper code treats `y` as increasing downward when rendering or describing frames, and `ACTION6` payloads are passed through as screen or display coordinates rather than solver-invented grid coordinates.

`v4` therefore records coordinates exactly as provided by the engine-facing action payload and does not reinterpret them as semantic grid coordinates unless the environment itself exposes that mapping.

## Observation Validation Rules

- The raw observation object must map to the local `FrameDataRaw` field set.
- `game_id` must be a string.
- `state` must be one of the local `GameState` values.
- `levels_completed` and `win_levels` must be integers.
- `full_reset` must be boolean.
- `available_actions` must be a list of integers.
- `frame` must be present and normalizable to an ordered plane stack.
- Missing optional metadata remains null or absent; it must not be guessed.
- Unknown authoritative fields are rejected rather than silently accepted.

## Authoritative Observation Subset

This subsection is normative.

The authoritative observation subset is restricted to fields directly returned by the environment on the local `FrameDataRaw` object:

- `game_id`
- `frame`
- `state`
- `levels_completed`
- `win_levels`
- `action_input`
- `guid`
- `full_reset`
- `available_actions`

Nothing inferred from those fields is itself authoritative. Derived entities, POIs, mechanic labels, topology, reward predictions, or planner interpretations are excluded from the authoritative observation subset.

## Observation Normalization

This subsection is normative.

The stable internal v4 observation shape must:

- preserve the raw `FrameDataRaw` payload content losslessly
- carry the raw object name as `FrameDataRaw`
- expose a normalized `frame` plane stack in `[plane][y][x]` order
- expose the directly observed scalar fields unchanged in meaning
- preserve `available_actions` exactly as returned by the environment

Normalization may change container types for stability, such as converting mutable lists or arrays into immutable tuples, but it must not invent fields or discard raw fields present on the local observation object.

## Explicit Exclusions

The observation contract explicitly excludes:

- POIs
- hypothesis labels
- mechanic graphs
- entity canonicals
- planner beliefs
- reward shaping
- candidate scores
- inferred topology
- symbolic promotions
