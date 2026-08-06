# ARC-AGI3 H07/H09 next repairs

Copy `src` into the repository root. The patch loads automatically because the
run command uses `PYTHONPATH=src`.

## H07

- Filters incremental validation diagnostics to signatures currently present in
  `concept_candidates`.
- Excludes stale historical diagnostics from current candidate counts.
- Separates persistent historical retention from current validated promotion.
- Uses `current_validation_passed` and the current promotion-score gate for the
  scientific `promoted_concept_count`.
- Adds explicit rejection reasons for zero coverage, insufficient samples,
  insufficient cross-scope evidence, absent lift, score failure, and
  incomparable held-out populations.
- Does not delete or rewrite historical promotion state.

New report fields include:

```text
persistent_retained_concept_count
current_validated_promoted_concept_count
stale_historical_promotion_diagnostic_count
concepts_retained_without_current_validation
```

## H09

- Detects cross-game and cross-context motif recurrence across distinct verified
  observations of the same motif.
- Does not require each observation to contain a source-target transfer pair.
- Excludes surrogate scopes from recurrence evidence.
- Retains the old pairwise counts under `pairwise_verified_*`.
- Uses recurrence evidence for qualifying emergent motifs and the H09 decision.

New report fields include:

```text
verified_cross_game_recurrence_motif_count
verified_cross_context_recurrence_motif_count
verified_cross_game_recurrence_observation_count
verified_cross_context_recurrence_observation_count
motif_scope_evidence_method
```

## Verify

```bash
PYTHONPATH=src python -c \
'import v6.h07_h09_next_repairs as p; import v6.hypothesis_h07_report as h7; import v6.hypothesis_h09_report as h9; print(p._PATCHED, h7._ARC_AGI3_CURRENT_CANDIDATE_REPORT_FIX, h9._ARC_AGI3_CROSS_OBSERVATION_SCOPE_FIX)'
```

Expected:

```text
True True True
```

No new output directory is required. The corrected logic applies from the next
hypothesis-suite execution.
