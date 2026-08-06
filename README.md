# ARC-AGI3 H07/H08 evidence repairs

Extract into the repository root. It preserves the earlier drop-ins and loads
automatically through the existing `PYTHONPATH=src` command.

## Fixes

### H07 longitudinal population continuity

Previously stored relevant event IDs remain relevant for the same persistent
concept while those events still exist and remain valid. Changing role links no
longer replaces almost the entire held-out population.

New diagnostics:

```text
population_retention_policy
historical_population_ids_available
historical_population_ids_carried_forward
relevant_by_historical_population_count
```

The first epoch using this patch can immediately reuse the full prior event-ID
population stored in `concept_incremental_coverage_state`.

### H07 compact prediction and contradiction evidence

The compact fallback now supports:

- `memory_nodes.attrs_json` and `payload_json`;
- interaction IDs such as `M0:interaction:g12345`;
- `updated_step`, local step and global-step columns;
- direct and two-hop interaction-to-contingency-to-family links;
- role matching through family, carrier, context or game links.

Prediction events are included only when a concrete compact structural link to
one of the concept's source roles is found.

### H08 current promotion state

H08 now uses the latest `current_validation_passed` result and the current
promotion-score gate. Historical retention is reported separately and cannot
satisfy the H08 promoted-concept gate.

World-model validation is also run against temporary current-validation flags,
then the historical `concept_candidates.is_promoted` values are restored.

## Install

```bash
unzip -o arc_agi3_h07_h08_evidence_repairs.zip
```

## Verify

```bash
PYTHONPATH=src python -c \
'import v6.h07_h08_evidence_repairs as p; import v6.higher_order_substrate as h; import v6.hypothesis_h08_report as r; print(p._PATCHED, h._ARC_AGI3_H07_EVIDENCE_CONTINUITY_FIX, r._ARC_AGI3_CURRENT_PROMOTION_FIX)'
```

Expected:

```text
True True True
```

The patch applies on the next hypothesis-suite execution. A new output
directory is not required.
