# ARC-AGI3 v6.1 canonical drop-in

Built against:

```text
repository: sorinbusegeanu/arc-agi-3
branch: main
commit: a55688a60a56762e9f582ceb8997b39df92c5025
```

## Install

Extract this archive in the repository root:

```bash
unzip -o arc_agi3_v61_canonical_dropin.zip
```

No separate migration command is required. The next hypothesis-suite run performs
the additive v6.1 migration before derivation.

## Canonical files replaced

```text
src/v6/hypothesis_suite_report.py
src/v6/provenance_validation.py
src/v6/memory/substrate.py
src/v6/graph/graph_manager.py
src/v6/context/contradiction_tracker.py
```

## New canonical modules

```text
src/v6/reporting/contracts.py
src/v6/reporting/evidence_snapshot.py
src/v6/reporting/framework.py
src/v6/memory/migrations/v61.py
src/v6/memory/controller.py
```

## Reporting changes

The existing `run_hypothesis_suite_report(...)` interface is preserved.

Execution is now separated into:

```text
WRITE   sampling/runtime interaction writes
DERIVE  family, role, transfer, concept, world-model and future-option derivation
REPORT  read-only evaluation against an immutable SQLite evidence snapshot
```

Each H01-H12 result contains:

```text
raw_decision
evidence_contract_gate
quality_gate
dependency_gate
final_decision
decision
```

`decision` remains as a compatibility alias for `final_decision`.

Strict behavior:

- missing required tables or fields becomes `INSUFFICIENT_EVIDENCE`;
- proxy-only evidence cannot validate;
- low required coverage fails the quality gate;
- invalid or missing required provenance fails the quality gate;
- failed dependencies cannot produce final `VALID`;
- evaluator exceptions or attempted writes fail closed;
- source-memory fingerprints are checked before and after REPORT.

## Memory changes

The additive migration extends universal memory identity with:

```text
schema_version
evidence_version
created_epoch
updated_epoch
status
```

Memory edges gain:

```text
edge_status
edge_confidence
edge_source
specificity_score
last_validated_epoch
```

Promotions gain:

```text
source_memory_ids_json
compression_gain
prediction_lift
transfer_score
explanatory_reach
epoch
global_step
```

New audit tables:

```text
memory_lifecycle_events
context_split_lineage
strategy_reuse_events
```

`MemoryController` provides one facade for:

```text
observe_interaction
predict
choose_action_candidates
score_replay
promote_candidates
retrieve_similar
query_replay_candidates
query_similar_contingencies
query_successful_trajectories
query_contradictions
query_future_option_effects
query_strategies
record_contradiction
record_context_split
record_success_trajectory
record_strategy_reuse
```

## Tests

The package adds:

```text
src/v6/tests/test_v61_reporting_framework.py
src/v6/tests/test_v61_memory_schema.py
src/v6/tests/test_v61_context_splitting.py
src/v6/tests/fixtures/reporting_v61/H01.json ... H12.json
```

Run:

```bash
PYTHONPATH=src pytest -q \
  src/v6/tests/test_v61_reporting_framework.py \
  src/v6/tests/test_v61_memory_schema.py \
  src/v6/tests/test_v61_context_splitting.py
```

Package validation result:

```text
21 passed
all Python files compiled successfully
hypothesis-suite import/run smoke test passed
schema-migration/read-only-snapshot smoke test passed
```

## Compatibility notes

This drop-in does not add new `sitecustomize.py` monkeypatches and does not remove
the repair modules already present in the repository. Existing compatibility
patch loading remains unchanged.

The existing runtime memory prediction and action-selection paths remain in
`V6System`; `MemoryController` consolidates their APIs without changing the
current CLI flags or forcing memory-guided action selection on by default.
