from __future__ import annotations

import unittest

import v8  # noqa: F401 - installs production runtime stack
from v8.arena import EdgeRecord, NodeRecord
from v8.formation_telemetry_v870 import (
    V870GenerativeCompressionEstimator,
    V870RelationalRoleEstimator,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.research import researcher_packet
from v8.structural_events import NormalizedPrimitive, StructuralFact


def node(
    level,
    memory_type,
    key,
    *,
    support=4,
    future=0.0,
    game_mask=3,
):
    uid = MemoryUid.from_key(level, memory_type, key)
    return NodeRecord(
        uid,
        (uid.hi ^ uid.lo) & ((1 << 64) - 1),
        int(level),
        int(memory_type),
        tuple(key),
        int(support),
        1.0,
        0.0,
        1.0,
        0.5,
        1.0,
        float(future),
        1.0,
        10,
        int(game_mask),
        int(CognitiveState.ACTIVE),
        int(ValidationState.STRUCTURAL),
        0.0,
        0.0,
        0.0,
    )


def edge(source, relation, target):
    return EdgeRecord(
        source.uid,
        int(relation),
        target.uid,
        1,
        10,
        0.0,
        0.0,
        0,
        0,
    )


class FormationTelemetryV870Tests(unittest.TestCase):
    def test_m2_telemetry_counts_grounded_normalized_and_actual_gate_rejections(self):
        created_a = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 101).token
        created_b = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 202).token
        removed = StructuralFact(NormalizedPrimitive.COMPONENT_REMOVED, 303).token
        available = StructuralFact(NormalizedPrimitive.ACTION_BECAME_AVAILABLE, 404).token
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=7),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (created_a,), support=4),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (created_b,), support=5),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (removed,), support=2),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (available,), support=4),
        )
        estimator = V870GenerativeCompressionEstimator(
            min_support=3,
            min_members=2,
            min_benefit=1.0,
        )
        proposals = estimator.discover(rows, budget=8)
        telemetry = estimator._v870_formation_telemetry

        self.assertEqual(len(proposals), 1)
        self.assertEqual(telemetry["m1g_count"], 1)
        self.assertEqual(telemetry["m1n_count"], 4)
        self.assertEqual(telemetry["stable_m1n_support_ge_3"], 3)
        self.assertEqual(telemetry["m2_support_eligible_m1n"], 3)
        self.assertEqual(telemetry["m2_family_groups"], 2)
        self.assertEqual(telemetry["eligible_m2_groups"], 1)
        self.assertEqual(telemetry["m2_candidates_emitted"], 1)
        self.assertEqual(telemetry["m2_rejections"]["m1n_below_min_support"], 1)
        self.assertEqual(telemetry["m2_rejections"]["group_insufficient_members"], 1)

    def test_m2_telemetry_reports_budget_truncation_without_changing_eligibility(self):
        a = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 101).token
        b = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 202).token
        c = StructuralFact(NormalizedPrimitive.COMPONENT_REMOVED, 301).token
        d = StructuralFact(NormalizedPrimitive.COMPONENT_REMOVED, 302).token
        rows = tuple(
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (token,), support=4)
            for token in (a, b, c, d)
        )
        estimator = V870GenerativeCompressionEstimator()
        proposals = estimator.discover(rows, budget=1)
        telemetry = estimator._v870_formation_telemetry
        self.assertEqual(telemetry["eligible_m2_groups"], 2)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(telemetry["m2_rejections"]["budget_limited"], 1)

    def test_role_telemetry_separates_carrier_and_lower_support_failures(self):
        m1a = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (11, 1, 2, 12))
        m1b = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (21, 1, 2, 22))
        f1 = node(MemoryLevel.M2, MemoryType.FAMILY, (1001, 0))
        f2 = node(MemoryLevel.M2, MemoryType.FAMILY, (2002, 0))

        c1 = node(MemoryLevel.M3, MemoryType.CARRIER, (111, 501, 0))
        c2 = node(MemoryLevel.M3, MemoryType.CARRIER, (222, 502, 0))
        c3 = node(MemoryLevel.M3, MemoryType.CARRIER, (333, 503, 1))
        c4 = node(MemoryLevel.M3, MemoryType.CARRIER, (444, 504, -1))
        c5 = node(MemoryLevel.M3, MemoryType.CARRIER, (555, 505, -1))

        edges = (
            edge(c1, RelationType.EXPLAINS, f1),
            edge(c1, RelationType.EXPLAINS, m1a),
            edge(c2, RelationType.EXPLAINS, f2),
            edge(c2, RelationType.EXPLAINS, m1b),
            edge(c3, RelationType.EXPLAINS, m1a),
            edge(c4, RelationType.EXPLAINS, m1a),
            edge(c5, RelationType.EXPLAINS, m1a),
        )
        estimator = V870RelationalRoleEstimator(min_carriers=2)
        roles = estimator.propose_relational((m1a, m1b, f1, f2, c1, c2, c3, c4, c5), edges)
        telemetry = estimator._v870_formation_telemetry

        self.assertEqual(len(roles), 1)
        self.assertEqual(telemetry["m3_carrier_count"], 5)
        self.assertEqual(telemetry["m3_carrier_groups"], 3)
        self.assertEqual(telemetry["carrier_groups_ge_2"], 2)
        self.assertEqual(telemetry["role_candidates"], 1)
        self.assertEqual(telemetry["role_rejections"]["group_insufficient_distinct_carriers"], 1)
        self.assertEqual(telemetry["role_rejections"]["group_insufficient_lower_support"], 1)

    def test_research_packet_contains_formation_telemetry(self):
        summary = {
            "games": ["gp03"],
            "actors": [{"game_id": "gp03", "steps": 10}],
            "automatic_transfer_experiments": {"attempted": 0, "completed": 0, "passed": 0},
            "metrics": {
                "watermark": 10,
                "level_counts": {"1": 10, "2": 0, "3": 0, "4": 0, "7": 0},
                "formation_telemetry": {
                    "m1g_count": 7,
                    "m1n_count": 3,
                    "stable_m1n_support_ge_3": 1,
                    "eligible_m2_groups": 0,
                    "m3_carrier_count": 0,
                    "carrier_groups_ge_2": 0,
                    "role_candidates": 0,
                },
            },
        }
        packet = researcher_packet.build_packet(
            summary,
            revision="test",
            argv=["continuous-run", "--games", "gp03"],
            h_report=[],
            reporting_cut={},
            evidence_digest={"available": True, "record_count": 0},
            log_tail="",
        )
        self.assertIn('"formation_telemetry"', packet)
        self.assertIn('"m1g_count": 7', packet)
        self.assertIn('"stable_m1n_support_ge_3": 1', packet)

    def test_runtime_stack_installs_v870_estimators(self):
        from v8 import peers

        self.assertIs(peers.CompressionEstimator, V870GenerativeCompressionEstimator)
        self.assertIs(peers.FunctionalRoleEstimator, V870RelationalRoleEstimator)


if __name__ == "__main__":
    unittest.main()
