Status: implemented and verified
Scope: agent contracts doc: serialization
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Serialization Contract

## Purpose

This document defines JSON serialization expectations for the `v4` authoritative contract models and their near-authoritative companions.

## File Or Object Scope

The serialization contract applies to:

- `V4Observation`
- `V4Action`
- `V4AuthoritativeState`
- `V4TerminalSignal`
- `V4TransitionRecord`
- `V4StepResult`
- `V4EnvironmentMetadata`

## Field Naming Policy

- Use stable snake_case field names.
- Preserve authoritative engine field names where possible.
- Do not rename `game_id`, `state`, `available_actions`, or `full_reset` during serialization.

## Enum Encoding Policy

The project commonly serializes dataclass payloads as plain JSON-friendly dicts. For `v4`:

- contract version is serialized as its string value
- terminal status is serialized as a string
- engine enum values already normalized into the contract are serialized as strings or integers already carried by the contract model

`v4` JSON does not depend on Python enum object serialization.

## Null Or Absent Field Policy

- Absent data must remain null or absent.
- Missing data must never be guessed during serialization.
- Optional fields should not be promoted to required by emitters.

## Raw Payload Preservation Rule

The authoritative raw payload preserved inside `V4Observation` and `V4EnvironmentMetadata` must not be dropped lossily during serialization.

That means:

- no dropping `frame` from the raw observation payload
- no dropping direct wrapper metadata fields from the raw metadata payload
- no rewriting unavailable fields into defaults

## Version Placement

When a serialized artifact needs an explicit contract version, place it at the top object level as a separate version field rather than rewriting the meaning of authoritative fields.

The individual models themselves remain version-neutral data carriers. Version tagging belongs to the containing artifact or envelope.

## Compatibility Rules

- Additive optional fields are compatible only when they remain optional and non-authoritative.
- Changing authoritative field meaning is not compatible.
- Changing action encoding is not compatible.
- Changing terminal encoding is not compatible.

## Prohibited Serialization Shortcuts

- No lossy dropping of raw authoritative payload.
- No guessed defaults for absent fields.
- No serialization of planner or analysis fields inside authoritative models.
- No collapsing raw observation snapshots into summary hashes only.
- No converting illegal actions into legal ones during serialization.

## Local Convention Notes

The local project frequently serializes dataclasses through `asdict()` and writes JSON dicts directly. Some artifact writers use `sort_keys=True`, but deterministic key ordering is not a universal project-wide requirement. `v4` therefore permits deterministic ordering when a writer opts into it, but does not require it as part of the core contract.

