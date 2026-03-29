Status: implemented and verified
Scope: agent contracts doc: environment metadata
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Environment Metadata

## Purpose

`v4` static environment metadata captures only environment-level facts that are exposed by the local wrapper or engine outside the per-step `FrameDataRaw` stream.

It exists to separate:

- static wrapper metadata
- static action-set metadata

from:

- per-step authoritative observations
- inferred planner or analysis state

## Authoritative Static Metadata

The local wrapper surface exposes authoritative static metadata through two real sources:

- `env.info`
- `env.action_space`

Observed local wrapper metadata fields on `env.info` include:

- `game_id`
- `title`
- `description`
- `local_dir`
- `date_downloaded`

Observed local action-set metadata on `env.action_space` is a list of `GameAction` enum members. Each member directly exposes:

- enum name
- integer id
- whether the action is complex

## Optional Passthrough Metadata

Some wrapper metadata is exposed inconsistently across local environments and must therefore remain optional:

- `title`
- `description`
- `local_dir`
- `date_downloaded`

These are authoritative only when directly present on the wrapper metadata object.

## Forbidden Inferred Metadata

Static metadata must not include inferred or promoted fields such as:

- POIs
- mechanic summaries
- planner summaries
- inferred grid topology
- inferred control affordances beyond the real action set
- estimated reward structure
- inferred task difficulty

## Reset Semantics

Reset semantics are not static metadata fields on `env.info`.

The local engine exposes reset behavior dynamically through:

- the `RESET` action in the action set
- the per-step `full_reset` field on returned `FrameDataRaw`

`v4` therefore documents reset capability as part of action-set metadata and reset outcome as part of per-step observation, not as a separate static metadata field.

## Action-Set Metadata

The static action-set contract is grounded in `env.action_space`.

Observed local behavior:

- `env.action_space` is a list of `GameAction` enum members
- each action exposes `name`
- each action exposes integer `value`
- complex action capability is exposed via `is_complex()`

If `ACTION6` appears in `env.action_space`, the environment supports the complex coordinate action.

## Coordinate-Bounds Metadata

The wrapper does not expose free-form coordinate bounds as a separate metadata field.

The only directly exposed coordinate bound in the local engine is the `ComplexAction` schema used by `ACTION6`, which validates:

- `x` in `0..63`
- `y` in `0..63`

If the action set does not include `ACTION6`, coordinate-bounds metadata must remain absent.

## Frame Or Grid Shape Metadata

The local wrapper does not expose stable frame or logical grid shape as static environment metadata.

Current project code derives width and height from actual observations rather than static metadata. `v4` therefore keeps frame or grid shape out of static metadata.

## Episode Or Level Identifiers

The observed static wrapper metadata exposes `game_id`.

No separate static episode id, level id, or level index is exposed on the local metadata surface reviewed here. Those fields must remain absent from the static metadata contract.

## Validation Rules

- `game_id` is required when static metadata is extracted.
- Metadata fields must come from direct wrapper or engine exposure only.
- `action_ids` and `action_names` must match the real `env.action_space`.
- `coordinate_bounds` may be present only when the real action set includes the complex coordinate action.
- Fields not directly exposed by the wrapper or engine must remain null or absent.

## Serialization Rules

- Preserve raw source metadata payload losslessly as far as the underlying wrapper object allows.
- Normalize directly exposed fields into stable JSON-friendly scalar values.
- Keep absent optional metadata null or absent.
- Do not serialize advisory or inferred metadata inside the authoritative static metadata model.

## Normative Table

| field name | source | required yes/no | authoritative yes/no | notes |
| --- | --- | --- | --- | --- |
| `game_id` | `env.info.game_id` | yes | yes | Primary static environment identifier |
| `title` | `env.info.title` | no | yes | Optional descriptive metadata |
| `description` | `env.info.description` | no | yes | Optional descriptive metadata; observed as absent in some local wrappers |
| `local_dir` | `env.info.local_dir` | no | yes | Optional local package path metadata |
| `date_downloaded` | `env.info.date_downloaded` | no | yes | Optional wrapper metadata; often surfaced as a datetime-like value |
| `action_ids` | `env.action_space[*].value` | no | yes | Direct static action-set ids |
| `action_names` | `env.action_space[*].name` | no | yes | Direct static action-set names |
| `coordinate_action_id` | complex member in `env.action_space` | no | yes | Present only when the environment exposes a complex action |
| `coordinate_bounds` | `ACTION6` `ComplexAction` schema | no | yes | Present only when the complex action is exposed |
| `raw_payload` | wrapper metadata object plus action-space source | yes | yes | Raw source preservation container |

