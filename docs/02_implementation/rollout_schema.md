# Rollout Batch Schema

This documents the `TRAJECTORY_BATCH_V1` structure written by `RolloutCollector`.

## Top Level
- `schema_version`: `"TRAJECTORY_BATCH_V1"`
- `episodes`: list of episode records

## Episode Record
- `game_id`: string
- `seed`: int
- `steps`: list of step records
- `done`: bool
- `num_steps`: int

## Step Record (PPO-relevant)
- `step_idx`: int
- `h_t`: recurrent state embedding (serialized)
- `mode_id`: int
- `mode_logits`: controller logits (serialized)
- `action_ids`: list of discrete action ids
- `action_index`: selected discrete action index
- `coord_candidates`: list of coord candidates
- `chosen_coord_index`: int or null
- `chosen_coord_tag`: string or null

### Old-policy statistics for PPO
- `old_logp_mode`: float
- `old_logp_action`: float
- `old_logp_coord`: float (`0.0` when coord not used)
- `old_value`: float
- `old_mode_entropy`: float
- `old_action_entropy`: float

### PPO masks
- `mask_valid_step`: int (`1` for valid step)
- `mask_has_coord`: int (`1` if coord decision exists else `0`)

### Reward and transition
- `reward`: float
- `reward_terms`: dict
- `reward_aux`: dict (may contain `mode_target`, `mode_weight`)
- `done`: bool
- `available_actions_mask`: list[int]
- `state_hash_before`: string or null
- `state_hash_after`: string or null
- `transition_event`: compact transition summary
