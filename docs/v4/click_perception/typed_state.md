Status: implemented and verified
Scope: click perception doc: typed state
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Common Required Fields**
- `game_family`
- `game_id`
- `level_index`
- `static_bounds`
- `clickable_cells`
- `legal_action_ids`
- `terminal_status`
- `step_depth`
- `visual_grid`

**Family-Specific Optional Fields**
- `rotation_tiles`, `target_rotations_by_type` for `pt01`
- `reflection_axis_x`, `reflection_pairs`, `mirror_source_cells`, `mirror_target_cells`, `placed_mirror_cells` for `sy01`
- `fill_regions`, `filled_region_indexes` for `ff01`
- `sequence_order`, `sequence_progress`, `clickable_color_cells` for `sq01`
- `active_mole_cells`, `mole_click_radius` for `wm01`
- `memory_slot_colors`, `hidden_slots`, `revealed_slots`, `matched_slots`, `slot_geometry` for `mm01`

**Forbidden Fields**
- POIs
- blackboard entities
- hypotheses
- planner scores
- learned rewards
- chain state
- durable symbolic abstractions

**Field Table**
| field name | source | required yes/no | authoritative yes/no | notes |
| --- | --- | --- | --- | --- |
| `clickable_cells` | direct observation plus env config | yes | no | solver candidate surface |
| `visual_grid` | direct observation frame | yes | no | exact control backing view |
| `legal_action_ids` | authoritative observation | yes | yes | current env-exposed action ids |
| `rotation_tiles` | direct observation plus `pt01` config | no | no | tile positions, types, rotations |
| `reflection_axis_x` | `sy01` level config | no | no | fixed vertical divider column in the local implementation |
| `reflection_pairs` | `sy01` pattern plus fixed reflection rule | no | no | explicit source-to-target mirror mapping |
| `mirror_target_cells` | `sy01` config | no | no | fixed reflected targets on the right side |
| `fill_regions` | `ff01` config | no | no | exact enclosed interiors |
| `sequence_order` | `sq01` config | no | no | required click order |
| `active_mole_cells` | directly visible `wm01` frame state | no | no | visible-only mole state |
| `memory_slot_colors` | `mm01` config | no | no | exact pair-color layout |

`ClickTypedStateV4` is solver state. It is derived from authoritative observation and metadata, but it is not itself authoritative env truth.

For `sy01`, search/selection state must include mirrored-state progress, not clicked-cell identity only. The placed right-side block set is part of solver state.
