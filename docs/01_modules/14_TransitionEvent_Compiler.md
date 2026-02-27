TransitionEvent_Compiler
0) Scope and non-goals

Scope: produce TransitionEventV1 from raw observations + FP_Analyst reports, with strict determinism and explicit handling of multi-frame / multi-grid cases.

Non-goals: any hypothesis logic, action selection, synthesis, exploration, planning, learning.

1) Inputs

Required:

prev_observation (step t-1 raw env payload)

observation (step t raw env payload)

action_taken (normalized action key)

fp_prev_report (FP_Analyst output for t-1)

fp_curr_report (FP_Analyst output for t)

ctx: game_id, seed, step_idx

Optional:

env_frame_names if environment provides named grids/frames (else derived from FP_Analyst normalization)

2) Output: TransitionEventV1 (canonical, versioned)

Codex must define a stable dataclass/JSON schema:

schema_version = "TRANSITION_EVENT_V1"

game_id, seed, step_idx

action_key:

kind: "RESET"|"SIMPLE"|"COORD"

id: "ACTION1"..

x,y if coord

state_hash_before, state_hash_after

frame_policy:

primary_grid_name

grid_names_used[]

aggregation: "PRIMARY_ONLY"|"UNION_DIFFS"

grid_delta (computed under frame_policy):

changed_cells_count

changed_bbox (y0,x0,y1,x1) or null if none

changed_colors: list of {from,to,count} (top-M capped, stable sorted)

palette_added[], palette_removed[] (sorted)

event_signatures:

list of {sig_id, confidence} sorted by (confidence desc, sig_id asc)

object_deltas (optional if tracking enabled in FP_Analyst):

moved[]: {object_id, dy, dx}

appeared_count, disappeared_count, split_count, merge_count

meta_delta:

available_actions_before[], available_actions_after[] (sorted)

terminal_before/after if present (else null)

reward_before/after if present (else null)

meta_keys_used[] (sorted)

3) Deterministic compilation rules
3.1 State hashing (must be pinned)

state_hash_* must be computed from:

the ordered set of normalized grids (names + values)

plus a whitelisted subset of meta keys:

available_actions

terminal, reward (if present)

explicitly exclude any volatile/debug fields.
Rule: if a meta key is missing, treat as null (do not omit).

3.2 Multi-frame / multi-grid policy

Default frame_policy:

primary_grid_name = first grid in FP_Analyst grids[] order (stable)

grid_names_used = [primary_grid_name]

aggregation = "PRIMARY_ONLY"

If config enables multi-grid:

grid_names_used = all FP_Analyst grid names (stable order)

aggregation = "UNION_DIFFS" meaning:

changed_cells_count = sum over grids

changed_bbox = bbox-union over grids (if any changed)

changed_colors aggregated across grids

palette_added/removed computed from union palettes

3.3 Event signatures source of truth

Use FP_Analyst diff_summary.event_signatures as input.

Normalize to the sig_id namespace and sort deterministically.

If FP_Analyst missing diff_summary (e.g., first step), set signatures = [{sig_id:"unknown", confidence:1.0}].

3.4 Changed_bbox conventions

bbox is inclusive-exclusive: (y0,x0,y1,x1) where y1/x1 are one past max index.

If no changed cells, bbox is null.

3.5 changed_colors cap

Keep top M=12 by count, stable tie-break (from,to).

4) Interfaces

compile(prev_observation, observation, action_taken, fp_prev, fp_curr, cfg, ctx) -> TransitionEventV1

to_json(event) -> dict with compact stable ordering

5) Configuration

enable_multigrid = false

hash_meta_whitelist = ["available_actions","terminal","reward"]

changed_colors_topM = 12

6) Logging

Emit a single line per compilation with:

hashes, action_key, changed_cells_count, top signature, grid policy
