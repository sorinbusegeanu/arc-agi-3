from __future__ import annotations

import unittest

import v8  # noqa: F401 - install current runtime stack
from v8.actor_read_view_v851 import ActorReadView
from v8.arena import NodeRecord, SharedNodeArena
from v8.model import MemoryLevel, MemoryUid
from v8.research.researcher_packet import build_packet


class ResearchPacketAndActorSnapshotV864Tests(unittest.TestCase):
    def test_research_packet_contains_decision_evidence_without_raw_noise(self) -> None:
        summary = {
            "actors": [
                {
                    "actor_id": 1,
                    "game_id": "g1",
                    "steps": 10,
                    "levels_completed": 2,
                    "wins": 1,
                    "failures": 0,
                    "resets": 1,
                    "planned_steps": 3,
                    "replans": 0,
                },
                {
                    "actor_id": 2,
                    "game_id": "g1",
                    "steps": 0,
                    "levels_completed": 0,
                    "wins": 0,
                    "failures": 0,
                    "resets": 0,
                    "planned_steps": 0,
                    "replans": 0,
                },
            ],
            "automatic_transfer_experiments": {"attempted": 0, "completed": 0, "passed": 0},
            "metrics": {
                "memories": 13,
                "edges": 7,
                "level_counts": {"0": 10, "1": 2, "2": 1, "3": 0},
                "memory_normalization": {
                    "m1g_nodes": 2,
                    "m1n_nodes": 1,
                    "m1n_cross_game_nodes": 1,
                    "m1n_per_grounded_support": 0.5,
                    "m2_from_m1n": 1,
                    "pipeline_packet_bytes": 999999,
                },
                "process_memory": {"processes": [{"pid": 12345, "rss_bytes": 999999}]},
                "actor_memory": [{"pid": 12345, "pss_bytes": 999999}],
            },
        }
        h_report = [
            {
                "hypothesis_id": "H01",
                "final_decision": "VALID",
                "evidence_count": 2,
                "paper_claim": "contingencies form",
                "blocker": "",
                "quality_gate": "PASS",
                "required_measurements": ["raw-noise-marker"],
            },
            {
                "hypothesis_id": "H05",
                "final_decision": "INSUFFICIENT_EVIDENCE",
                "evidence_count": 0,
                "paper_claim": "roles recur",
                "blocker": "missing role_candidate",
            },
        ]
        reporting_cut = {
            "paper_traceability": ["reporting-cut-noise-marker"],
            "observation_contract": {"forbidden_semantic_fields": ["noise"]},
            "graph_digest": "deadbeef",
        }
        evidence_digest = {
            "available": True,
            "record_count": 4,
            "evidence_kind_counts": {"contingency_recurrence": 4},
            "causal_intervention_counts": {},
            "effect_direction_counts": {"neutral": 4},
            "distinct_source_games": 2,
            "distinct_target_games": 0,
            "samples": [{"memory_uid_hi": 999, "evidence_id": "raw-ledger-noise-marker"}],
        }

        packet = build_packet(
            summary,
            revision="abc123",
            argv=("continuous-run", "--games", "mix"),
            h_report=h_report,
            reporting_cut=reporting_cut,
            evidence_digest=evidence_digest,
            log_tail="[00:00] effectiveness L=20% G=10%",
        )

        self.assertIn('"id": "H01"', packet)
        self.assertIn('"blocker": "missing role_candidate"', packet)
        self.assertIn('"game_id": "g1"', packet)
        self.assertIn('"actors": 2', packet)
        self.assertIn('"active_actors": 1', packet)
        self.assertIn('"M2": 1', packet)
        self.assertIn('"distinct_source_games": 2', packet)
        self.assertIn("effectiveness L=20% G=10%", packet)

        for noise in (
            "## Reporting cut",
            "## Full run summary",
            "process_memory",
            "actor_memory",
            "paper_traceability",
            "observation_contract",
            "graph_digest",
            "raw-noise-marker",
            "reporting-cut-noise-marker",
            "raw-ledger-noise-marker",
            "memory_uid_hi",
            "12345",
            "pipeline_packet_bytes",
        ):
            self.assertNotIn(noise, packet)

    def test_nonempty_actor_scan_uses_coherent_cut_not_long_live_seqlock(self) -> None:
        arena = SharedNodeArena(capacity=4)
        try:
            row = NodeRecord(
                uid=MemoryUid(1, 1),
                fingerprint=11,
                level=int(MemoryLevel.M1),
                memory_type=100,
                key_parts=(7, 8, 9),
                support_count=3,
                significance_sum=0.0,
                prediction_error_sum=0.0,
                learning_value_sum=0.0,
                transfer_prior_sum=0.0,
                explanatory_sum=0.0,
                future_option_sum=0.0,
                score_weight=1.0,
                updated_watermark=1,
            )
            arena.begin_write()
            arena.write(0, row)
            arena.end_write(count=1)

            original_read = arena.read

            def destabilizing_live_read(index: int):
                value = original_read(index)
                count = arena.count
                arena.begin_write()
                arena.end_write(count=count)
                return value

            arena.read = destabilizing_live_read
            version, _nodes, _lineage, _active, counts, totals = ActorReadView._scan_node_arena(arena)

            self.assertEqual(version & 1, 0)
            self.assertEqual(sum(totals.values()), 3)
            self.assertEqual(sum(sum(bucket.values()) for bucket in counts.values()), 3)
        finally:
            arena.dispose()


if __name__ == "__main__":
    unittest.main()
