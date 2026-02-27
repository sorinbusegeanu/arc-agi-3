## Memory Schema (v1.0)

Deterministic, JSON-serializable snapshot produced by `memory_snapshot()` and saved as `memory.json`.

## Persistent Store (v1.0) — SQLite

On-disk store under `memory_dir/`:

* `memory.sqlite` (primary store, WAL enabled)
* `journal/run_summaries.jsonl` (optional)
* `locks/store.lock`
* `schema.json` (optional human mirror)

SQLite tables (normative):

1) `schema_meta`

* `schema_version`
* `created_ts`
* `last_compaction_ts`
* `feature_flags`

2) `action_priors_global`

Key: `action_key`

* `attempts_total`
* `effect_total`
* `no_effect_total`
* `avg_changed_cells`
* `avg_bbox_area`
* `last_seen_ts`

3) `action_priors_by_signature`

Key: `(task_signature_v1, action_key)`

* same fields as global

4) `game_memory`

Key: `game_id`

* `success_count`
* `attempt_count`
* `last_success_ts`
* `best_known_programs` (serialized)
* `known_noop_signatures` (optional, serialized)
* `failure_histogram_json`

5) `failure_histograms`

Key: `(task_signature_v1, failure_label)`

* `count`
* `params` (optional)

6) `candidate_priors` (optional)

Key: `(task_signature_v1, candidate_signature_v1)`

* `times_considered`
* `times_accepted`
* `times_rejected`
* `avg_score`
* `last_score`
* `reject_histogram_json`

7) `agent_calibration` (optional)

Key: `(task_signature_v1, agent_id, role)`

* `suggestions_count`
* `accepted_count`
* `led_to_progress_count`
* `led_to_win_count`

8) `events_run_summary_v1` (optional)

* `run_summary_v1_json`
* `ingested_ts`
* `run_id`

## RUN_SUMMARY_V1 (ingestion unit)

Required fields:

* `schema_version`: `"RUN_SUMMARY_V1"`
* `task_signature_v1`
* `game_id`
* `seed`
* `run_id`
* `win`
* `progress_metrics`
* `action_efficacy`
* `hypothesis_outcomes`
* `mechanic_posterior_evolution`
* `failure_labels`

### Top-level

* `version`: string, `"1.0"`
* `per_action`: map `action_key -> ActionEfficacy`
* `per_state_action`: map `"state_hash|action_key" -> StateActionEfficacy`
* `coord_heatmaps`: map `action_id -> { "x,y" -> CoordStat }`
* `event_sig_window`: list of per-step `{sig -> count, "_total" -> n}` (max `K_long`)
* `object_delta_window`: list of per-step `{event -> count}` (max `K_long`)
* `template_stats`: map `hypothesis_id -> TemplateStats`
* `recent_actions_by_state`: map `state_hash -> [action_key]` (length `K_short`)
* `mechanic_by_fingerprint`: map `grid_fingerprint -> {family_id -> {count, avg_prior}}`

### ActionEfficacy

* `attempts`: int
* `no_effect_count`: int
* `effect_count`: int
* `avg_changed_cells`: float
* `avg_changed_bbox_area`: float
* `last_step_seen`: int
* `event_signature_counts`: map `signature -> count`
* `source_counts`: map `planner_source -> count`

### StateActionEfficacy

* `attempts`: int
* `no_effect_count`: int
* `last_effect_step`: int | null
* `last_step_seen`: int

### CoordStat

* `attempts`: int
* `no_effect_count`: int
* `avg_changed_cells`: float
* `last_step_seen`: int

### TemplateStats

* `times_considered`: int
* `times_triggered`: int
* `times_scored_positive`: int
* `last_step_triggered`: int | null
* `supporting_events`: map `event -> count`
