# Board Perception Contracts

## Typed Outputs

- `BoardObject`
- `BoardState`
- `BoardPerceptionReport`

## BoardObject

`BoardObject` contains only:

- `object_id`
- `object_type`
- `bbox`
- `center`
- `position_x`
- `position_y`
- `color`

## BoardState

`BoardState` includes:

- schema version
- round id
- board geometry summary
- source observation window size
- avatar object optional
- object list
- traversable background representation
- blocking background representation
- hazard representation optional
- hud region optional
- progress bar region optional
- lives region optional
- advisory-only flag
- gap list / uncertainty notes

## BoardPerceptionReport

`BoardPerceptionReport` is the wrapper returned by the module and includes:

- `schema_version`
- module name
- `round_id`
- `board_state`
- rationale codes
- explicit gaps
- advisory-only marker

