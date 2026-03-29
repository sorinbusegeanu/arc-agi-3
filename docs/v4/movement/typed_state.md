Status: implemented and verified
Scope: movement doc: typed state
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `MovementTypedStateV4`

## Common Required Fields

- avatar position when directly derivable from the current or previous observation
- traversable-cell set
- current legal movement actions
- terminal status for the current level solve model
- step depth
- static bounds

## Family-Specific Optional Fields

- key inventory bits
- door states
- switch and toggle bits
- teleporter mapping
- slide mode
- coverage mask
- pushable block positions
- hazard positions
- target cells

### `fs01` Requirements

- switch positions must be represented explicitly
- switch activation must be represented as exact state bits over those positions
- door positions must be represented while the door is still closed
- door open/closed state must be represented explicitly
- switch-to-door linkage must remain absent unless the real game data directly exposes more than the observed global all-switches-open-the-door rule
- `fs01` search state is not position-only; avatar position alone is insufficient

### `tp01` Requirements

- teleporter endpoint positions must be represented explicitly
- teleporter pair mapping must be represented explicitly
- any warp-on-enter behavior flag must remain absent unless a real local config surface exposes it directly
- target cells must be represented when directly derivable
- `tp01` search state must include avatar position and the explicit teleporter pair map
- symmetric fixed teleporter pairs usually add no hidden state beyond position plus the fixed pair map

### `ic01` Requirements

- the exact slide surface must be represented explicitly
- red hazard cells must be represented explicitly
- traversability must be represented in the same stop-before-block form used by the real slide rule
- goal cells must be represented when directly derivable
- no extra slide-rule flags may be invented unless a real local config surface exposes them
- `ic01` search uses deterministic slide-resolved successors
- the state must be sufficient to reproduce the exact landing cell for each primitive action

### `va01` Requirements

- the exact coverage-eligible cell set must be represented explicitly
- the current covered-state representation must be explicit
- the start-cell coverage rule must be represented because the real local game marks the start cell visited immediately
- no separate completion goal field may be invented unless a real local config surface exposes one
- wall blockers relevant to coverage progression must remain explicit
- `va01` search state must include coverage progress, not position only

### `pb01` Requirements

- exact pushable-object positions must be represented explicitly
- target cells must be represented explicitly when directly derivable from the real local level config
- blocking cells relevant to push legality must remain explicit
- completion must be representable from explicit block-position and target-cell state only
- `pb01` search state must include pushable-object positions, not avatar position only

## Forbidden Fields

- POIs
- blackboard entities
- hypotheses
- planner scores
- learned rewards
- chain state
- durable symbolic abstractions

## Field Table

| field name | source | required yes/no | authoritative yes/no | notes |
| --- | --- | --- | --- | --- |
| `common.game_family` | game id stem and caller family selection | yes | no | solver-routing field |
| `common.game_id` | current observation | yes | yes | copied from authoritative observation |
| `common.level_index` | `levels_completed` | yes | yes | current live level index |
| `common.avatar_position` | direct frame decode, with previous-observation fallback where special tiles hide avatar | yes | no | solver-state field derived from direct observation |
| `common.traversable_cells` | family adapter plus static local layout | yes | no | exact movement surface for the current family |
| `common.current_legal_actions` | observation available-actions | yes | yes | movement actions exposed by env |
| `common.terminal_status` | typed-state level-completion semantics | yes | no | level-solve terminal status, not raw env terminal only |
| `common.step_depth` | Stage 2 parsed-state step index | yes | yes | search/debug depth anchor |
| `common.static_bounds` | local family level layout | yes | no | exact movement bounds |
| `common.blocked_cells` | family adapter plus current dynamic blockers | no | no | walls, closed doors, hazards |
| `common.target_cells` | local family layout or directly derivable goal cells | no | no | used for search and validation |
| `common.hazard_positions` | directly observed or fixed hazard cells | no | no | only when relevant |
| `family.key_inventory_bits` | key-presence reconstruction | no | no | `ul01` |
| `family.door_positions` | direct observation plus family rules | no | no | `ul01`, `fs01` |
| `family.door_open` | family state reconstruction | no | no | `ul01`, `fs01` |
| `family.switch_positions` | local level layout | no | no | `fs01` |
| `family.activated_switch_bits` | direct observation plus family rules | no | no | `fs01` |
| `family.door_state_bits` | exact family state reconstruction | no | no | `fs01` explicit open/closed bit |
| `family.teleporter_endpoint_positions` | real level config plus directly observed portal cells | no | no | `tp01` |
| `family.teleporter_pairs` | local level layout | no | no | `tp01` |
| `family.teleporter_pair_map` | real level config | no | no | `tp01` explicit symmetric warp map |
| `family.slide_mode` | family type | no | no | `ic01` |
| `family.ice_cell_positions` | direct board layout after removing walls and red hazards | no | no | `ic01` exact slide surface |
| `family.coverage_eligible_cells` | direct board layout after removing walls | no | no | `va01` exact visit-all target |
| `family.coverage_mask` | direct trail and avatar observation | no | no | `va01` current covered-state |
| `family.pushable_block_positions` | direct frame decode | no | no | `pb01` exact movable-object position |
| `family.push_target_cells` | real local level config | no | no | `pb01` explicit target representation |
| `family.step_limit` | real local level config | no | no | `pb01` only |
