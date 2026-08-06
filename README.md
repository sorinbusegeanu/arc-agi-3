# ARC-AGI3 remaining report repairs

Extract into the repository root. This package preserves and loads the earlier
drop-ins.

## H02 compact metadata

When direct replay lift comes from the compact interaction join, the report now
fills:

```text
row_count_available
row_count_used
prediction_violation_row_count
non_prediction_violation_row_count
mean_replay_priority_for_prediction_violating_interactions
mean_replay_priority_for_non_prediction_violating_interactions
candidate_tables_used
prediction_violation_metric_source
replay_priority_metric_source
db_path
```

## H08 family-link selectivity

World-model family links are filtered using all of the following:

- verified provenance;
- support of at least two;
- at least two observed events;
- at least two linked roles or positive prediction gain.

Strong links are ranked by event support, role breadth, raw support and
prediction gain. An adaptive cap is then applied:

```text
min(configured_cap, max(8, round(sqrt(candidate_family_count))))
```

The fix does not raise the old 50-link cap. It prevents components from
expanding indefinitely as candidate families accumulate.

Diagnostics are stored under:

```text
memory_summary.world_model_family_selection_repair
```

## H09 graph-edge provenance

Future-option graph-edge events are backfilled from concrete interaction
metadata in `memory_nodes` or `graph_nodes`.

A graph-edge classification is verified when its source interaction has a
concrete game and context. Target scope is also populated when the target is a
resolvable interaction.

## Install

```bash
unzip -o arc_agi3_remaining_report_repairs.zip
```

## Verify

```bash
PYTHONPATH=src python -c \
'import v6.remaining_report_repairs as p; import v6.hypothesis_h02_report as h2; import v6.higher_order_substrate as h8; import v6.future_options as h9; print(p._PATCHED, h2._ARC_AGI3_COMPACT_METADATA_FIX, h8._ARC_AGI3_WORLD_MODEL_FAMILY_SELECTIVITY_FIX, h9._ARC_AGI3_EDGE_PROVENANCE_BACKFILL_FIX)'
```

Expected:

```text
True True True True
```

A new output directory is not required. The fixes apply from the next
hypothesis-suite execution.
