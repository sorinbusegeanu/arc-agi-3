# v9.7.16 Acceptance Matrix

This is the authoritative migration-status index for acceptance criteria 1-128 in section 31 of the v9.7.16 design. A unit test proves coverage of every criterion number. An \`implemented\` row identifies focused evidence; \`pending\` is intentionally non-accepting and must not be interpreted as complete.

Current cutover state (2026-09-20): the historical baseline is 295 passed in
368.85 seconds. The last complete expanded-suite gate before the subsequent
transport, snapshot-crash, and matched-view changes was 390 passed with two
warnings in 559.15 seconds; those later changes currently have focused evidence
only. `ScientificConfig.design_version` remains `9.7.9`. Canonical WAL and
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
| 64 | pending | Phase implementation and acceptance evidence pending |
| 65 | pending | Phase implementation and acceptance evidence pending |
| 66 | pending | Phase implementation and acceptance evidence pending |
| 67 | pending | Phase implementation and acceptance evidence pending |
| 68 | pending | Phase implementation and acceptance evidence pending |
| 69 | pending | Phase implementation and acceptance evidence pending |
| 70 | pending | Phase implementation and acceptance evidence pending |
| 71 | pending | Phase implementation and acceptance evidence pending |
| 72 | pending | Phase implementation and acceptance evidence pending |
| 73 | pending | Phase implementation and acceptance evidence pending |
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
| 88 | pending | Phase implementation and acceptance evidence pending |
| 89 | pending | Phase implementation and acceptance evidence pending |
| 90 | pending | Phase implementation and acceptance evidence pending |
| 91 | pending | Phase implementation and acceptance evidence pending |
| 92 | pending | Phase implementation and acceptance evidence pending |
| 93 | pending | Phase implementation and acceptance evidence pending |
| 94 | implemented | src/v9/tests/test_scientific_identity_determinism.py |
| 95 | implemented | src/v9/tests/test_scientific_identity_determinism.py |
| 96 | pending | Phase implementation and acceptance evidence pending |
| 97 | pending | Phase implementation and acceptance evidence pending |
| 98 | pending | Phase implementation and acceptance evidence pending |
| 99 | pending | Phase implementation and acceptance evidence pending |
| 100 | pending | Phase implementation and acceptance evidence pending |
| 101 | pending | Phase implementation and acceptance evidence pending |
| 102 | pending | Phase implementation and acceptance evidence pending |
| 103 | pending | Phase implementation and acceptance evidence pending |
| 104 | pending | Phase implementation and acceptance evidence pending |
| 105 | pending | Phase implementation and acceptance evidence pending |
| 106 | pending | Phase implementation and acceptance evidence pending |
| 107 | pending | Phase implementation and acceptance evidence pending |
| 108 | pending | Phase implementation and acceptance evidence pending |
| 109 | pending | Phase implementation and acceptance evidence pending |
| 110 | pending | Phase implementation and acceptance evidence pending |
| 111 | pending | Phase implementation and acceptance evidence pending |
| 112 | pending | Phase implementation and acceptance evidence pending |
| 113 | pending | Phase implementation and acceptance evidence pending |
| 114 | pending | Phase implementation and acceptance evidence pending |
| 115 | pending | Phase implementation and acceptance evidence pending |
| 116 | pending | Phase implementation and acceptance evidence pending |
| 117 | pending | Phase implementation and acceptance evidence pending |
| 118 | pending | Phase implementation and acceptance evidence pending |
| 119 | pending | Phase implementation and acceptance evidence pending |
| 120 | pending | Phase implementation and acceptance evidence pending |
| 121 | pending | Phase implementation and acceptance evidence pending |
| 122 | pending | Phase implementation and acceptance evidence pending |
| 123 | pending | Phase implementation and acceptance evidence pending |
| 124 | pending | Phase implementation and acceptance evidence pending |
| 125 | pending | Phase implementation and acceptance evidence pending |
| 126 | pending | Phase implementation and acceptance evidence pending |
| 127 | pending | Phase implementation and acceptance evidence pending |
| 128 | pending | Phase implementation and acceptance evidence pending |
