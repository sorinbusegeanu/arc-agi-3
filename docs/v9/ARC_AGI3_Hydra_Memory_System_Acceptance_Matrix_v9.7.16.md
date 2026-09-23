# v9.7.16 Acceptance Matrix

This is the authoritative migration-status index for acceptance criteria 1-128 in section 31 of the v9.7.16 design. A unit test proves coverage of every criterion number. An \`implemented\` row identifies focused evidence; \`pending\` is intentionally non-accepting and must not be interpreted as complete.

Current cutover state (2026-09-21): the complete expanded v9 suite passes 454
tests with two warnings after the WAL-evidence, deterministic-training, storage,
restart-identity, dashboard-concurrency, and matched derivation-cut integrations.
`ScientificConfig.design_version` remains `9.7.9`. Canonical WAL and
immutable-root integration are available only behind the explicit
`RuntimeConfig.enable_canonical_durability` migration gate; the gate remains
off by default because its fixed-benchmark RAM and throughput requirements have
not passed. Standalone unit evidence does not make a pending whole-system
criterion implemented.

| Criterion | Status | Evidence |
|---:|---|---|
| 1 | pending | Phase implementation and acceptance evidence pending |
| 2 | pending | Phase implementation and acceptance evidence pending |
| 3 | pending | Phase implementation and acceptance evidence pending |
| 4 | pending | Phase implementation and acceptance evidence pending |
| 5 | pending | Phase implementation and acceptance evidence pending |
| 6 | pending | Phase implementation and acceptance evidence pending |
| 7 | pending | Phase implementation and acceptance evidence pending |
| 8 | pending | Phase implementation and acceptance evidence pending |
| 9 | pending | Phase implementation and acceptance evidence pending |
| 10 | pending | Phase implementation and acceptance evidence pending |
| 11 | pending | Phase implementation and acceptance evidence pending |
| 12 | pending | Phase implementation and acceptance evidence pending |
| 13 | pending | Phase implementation and acceptance evidence pending |
| 14 | pending | Phase implementation and acceptance evidence pending |
| 15 | pending | Phase implementation and acceptance evidence pending |
| 16 | pending | Phase implementation and acceptance evidence pending |
| 17 | pending | Phase implementation and acceptance evidence pending |
| 18 | pending | Phase implementation and acceptance evidence pending |
| 19 | pending | Phase implementation and acceptance evidence pending |
| 20 | pending | Phase implementation and acceptance evidence pending |
| 21 | pending | Phase implementation and acceptance evidence pending |
| 22 | pending | Phase implementation and acceptance evidence pending |
| 23 | pending | Phase implementation and acceptance evidence pending |
| 24 | pending | Phase implementation and acceptance evidence pending |
| 25 | pending | Phase implementation and acceptance evidence pending |
| 26 | pending | Phase implementation and acceptance evidence pending |
| 27 | pending | Phase implementation and acceptance evidence pending |
| 28 | pending | Phase implementation and acceptance evidence pending |
| 29 | pending | Phase implementation and acceptance evidence pending |
| 30 | pending | Phase implementation and acceptance evidence pending |
| 31 | pending | Phase implementation and acceptance evidence pending |
| 32 | pending | Phase implementation and acceptance evidence pending |
| 33 | pending | Phase implementation and acceptance evidence pending |
| 34 | pending | Phase implementation and acceptance evidence pending |
| 35 | pending | Phase implementation and acceptance evidence pending |
| 36 | pending | Phase implementation and acceptance evidence pending |
| 37 | pending | Phase implementation and acceptance evidence pending |
| 38 | implemented | src/v9/tests/test_epoch_inference_view.py::test_matched_actors_report_the_exact_bound_epoch_view |
| 39 | pending | Phase implementation and acceptance evidence pending |
| 40 | pending | Phase implementation and acceptance evidence pending |
| 41 | pending | Phase implementation and acceptance evidence pending |
| 42 | pending | Phase implementation and acceptance evidence pending |
| 43 | pending | Phase implementation and acceptance evidence pending |
| 44 | implemented | src/v9/tests/test_scientific_identity_determinism.py |
| 45 | pending | Phase implementation and acceptance evidence pending |
| 46 | pending | Phase implementation and acceptance evidence pending |
| 47 | pending | Phase implementation and acceptance evidence pending |
| 48 | pending | Phase implementation and acceptance evidence pending |
| 49 | pending | Phase implementation and acceptance evidence pending |
| 50 | implemented | src/v9/tests/test_producer_affinity.py::test_fixed_descriptor_crosses_stage_and_shard_before_single_decode |
| 51 | implemented | src/v9/tests/test_shm_credit_ownership.py::test_live_transport_and_compiled_result_subpools_share_one_global_ceiling |
| 52 | implemented | src/v9/tests/test_shm_credit_ownership.py::test_compiled_result_recovery_preserves_coordinator_owned_credit |
| 53 | implemented | src/v9/tests/test_signature_index_store.py; src/v9/tests/test_signature_index_restart.py |
| 54 | implemented | src/v9/tests/test_batched_publication_intake.py::test_pending_batch_buffer_enforces_exact_carried_byte_high_water; src/v9/tests/test_publication_throughput.py::test_reducer_completion_releases_exact_admitted_input_bytes |
| 55 | implemented | src/v9/tests/test_canonical_wal.py; src/v9/tests/test_wal_group_commit.py; src/v9/tests/test_wal_torn_tail.py |
| 56 | implemented | src/v9/tests/test_persistence_frontiers.py; src/v9/tests/test_live_canonical_durability.py |
| 57 | implemented | src/v9/tests/test_wal_replay_idempotence.py::test_wal_is_durable_before_overlay_becomes_visible; src/v9/tests/test_live_canonical_durability.py::test_live_graph_visibility_follows_wal_and_immutable_root |
| 58 | implemented | src/v9/tests/test_canonical_fragment_visibility.py; src/v9/tests/test_canonical_continuation_fragments.py |
| 59 | implemented | src/v9/tests/test_canonical_handle_atomicity.py; src/v9/tests/test_live_canonical_durability.py |
| 60 | implemented | src/v9/tests/test_post_wal_pre_publish_recovery.py; src/v9/tests/test_wal_replay_idempotence.py |
| 61 | implemented | src/v9/tests/test_snapshot_handle_consistency.py; src/v9/tests/test_canonical_snapshot_crash_recovery.py |
| 62 | implemented | src/v9/tests/test_hgt_materialization_atomicity.py; src/v9/tests/test_training_evidence_segments.py |
| 63 | implemented | src/v9/tests/test_hgt_checkpoint_lsn.py; src/v9/tests/test_training_evidence_segments.py |
| 64 | implemented | src/v9/tests/test_wal_consumer_retention.py; src/v9/tests/test_live_canonical_durability.py::test_runtime_reclaims_only_snapshot_and_hgt_acknowledged_wal |
| 65 | pending | Phase implementation and acceptance evidence pending |
| 66 | pending | Phase implementation and acceptance evidence pending |
| 67 | implemented | src/v9/tests/test_canonical_work_budget.py; src/v9/tests/test_canonical_oversized_quarantine.py |
| 68 | pending | Phase implementation and acceptance evidence pending |
| 69 | implemented | src/v9/tests/test_derivation_leases.py; src/v9/tests/test_derivation_retry_idempotence.py |
| 70 | pending | Phase implementation and acceptance evidence pending |
| 71 | implemented | src/v9/tests/test_live_canonical_durability.py::test_derivation_publication_uses_wal_overlay_and_atomic_root |
| 72 | pending | Phase implementation and acceptance evidence pending |
| 73 | implemented | src/v9/tests/test_v9716_governor_accounting.py |
| 74 | pending | Phase implementation and acceptance evidence pending |
| 75 | pending | Phase implementation and acceptance evidence pending |
| 76 | pending | Phase implementation and acceptance evidence pending |
| 77 | pending | Phase implementation and acceptance evidence pending |
| 78 | pending | Phase implementation and acceptance evidence pending |
| 79 | pending | Phase implementation and acceptance evidence pending |
| 80 | pending | Phase implementation and acceptance evidence pending |
| 81 | pending | Phase implementation and acceptance evidence pending |
| 82 | pending | Phase implementation and acceptance evidence pending |
| 83 | pending | Phase implementation and acceptance evidence pending |
| 84 | pending | Phase implementation and acceptance evidence pending |
| 85 | pending | Phase implementation and acceptance evidence pending |
| 86 | pending | Phase implementation and acceptance evidence pending |
| 87 | pending | Phase implementation and acceptance evidence pending |
| 88 | implemented | src/v9/tests/test_developmental_cut.py; src/v9/tests/test_developmental_cut_operator_coverage.py; src/v9/tests/test_developmental_cut_schedule_independence.py; src/v9/tests/test_parallel_developmental_cut.py |
| 89 | implemented | src/v9/tests/test_training_cut.py; src/v9/tests/test_hgt_iterative_training.py |
| 90 | implemented | src/v9/tests/test_training_cut_replay_determinism.py |
| 91 | implemented | src/v9/tests/test_training_cut_optimizer_count.py; src/v9/tests/test_hgt_iterative_training.py |
| 92 | implemented | src/v9/tests/test_training_evidence_records.py; src/v9/tests/test_live_canonical_durability.py |
| 93 | implemented | src/v9/tests/test_training_evidence_restart.py; src/v9/tests/test_training_evidence_segments.py |
| 94 | implemented | src/v9/tests/test_scientific_identity_determinism.py |
| 95 | implemented | src/v9/tests/test_scientific_identity_determinism.py |
| 96 | implemented | src/v9/tests/test_storage_governor.py; src/v9/tests/test_snapshot_handle_consistency.py |
| 97 | implemented | src/v9/tests/test_storage_governor.py; src/v9/tests/test_hgt_iterative_training.py |
| 98 | pending | Phase implementation and acceptance evidence pending |
| 99 | pending | Phase implementation and acceptance evidence pending |
| 100 | pending | Phase implementation and acceptance evidence pending |
| 101 | pending | Phase implementation and acceptance evidence pending |
| 102 | pending | Phase implementation and acceptance evidence pending |
| 103 | implemented | src/v9/tests/test_h16_grounding_conditions.py; src/v9/tests/test_h16_bidirectional_transfer.py |
| 104 | implemented | src/v9/tests/test_h16_c3_alignment_destroyed.py; src/v9/tests/test_h16_c3_no_metadata_leak.py |
| 105 | implemented | src/v9/tests/test_h16_bidirectional_transfer.py |
| 106 | implemented | src/v9/tests/test_scientific_modes.py; src/v9/tests/test_h17_async_development.py |
| 107 | implemented | src/v9/tests/test_h17_async_development.py |
| 108 | pending | Phase implementation and acceptance evidence pending |
| 109 | pending | Phase implementation and acceptance evidence pending |
| 110 | implemented | src/v9/tests/test_trial_manifest.py; src/v9/tests/test_h19_unused_horizon_discard.py |
| 111 | implemented | src/v9/tests/test_h19_no_adaptive_budget_confounds.py; src/v9/tests/test_h19_unused_horizon_discard.py |
| 112 | pending | Model-only control is declared but lacks integrated execution evidence |
| 113 | implemented | src/v9/tests/test_developmental_milestone_ledger.py; src/v9/tests/test_training_evidence_records.py |
| 114 | implemented | src/v9/tests/test_prediction_registry.py |
| 115 | implemented | src/v9/tests/test_prediction_registry.py |
| 116 | pending | Phase implementation and acceptance evidence pending |
| 117 | pending | Phase implementation and acceptance evidence pending |
| 118 | pending | Phase implementation and acceptance evidence pending |
| 119 | implemented | src/v9/tests/test_h17_async_development.py; src/v9/tests/test_scientific_modes.py |
| 120 | implemented | src/v9/tests/test_training_evidence_records.py; src/v9/tests/test_hgt_iterative_training.py |
| 121 | implemented | src/v9/tests/test_canonical_store.py; src/v9/tests/test_canonical_handle_atomicity.py; src/v9/tests/test_snapshot_handle_consistency.py |
| 122 | implemented | src/v9/tests/test_canonical_continuation_fragments.py; src/v9/tests/test_canonical_fragment_visibility.py |
| 123 | pending | H18 transform is unit-tested but is not yet enforced at every adapter boundary |
| 124 | pending | H18 anti-leak transforms are unit-tested but are not yet enforced at every adapter boundary |
| 125 | implemented | src/v9/tests/test_trial_manifest.py; src/v9/tests/test_h19_unused_horizon_discard.py |
| 126 | implemented | src/v9/tests/test_h16_c3_alignment_destroyed.py; src/v9/tests/test_h16_c3_no_metadata_leak.py |
| 127 | implemented | src/v9/tests/test_prediction_registry.py |
| 128 | implemented | src/v9/tests/test_training_cut_kernel_fallback.py; src/v9/tests/test_hgt_iterative_training.py |
