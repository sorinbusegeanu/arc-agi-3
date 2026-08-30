from __future__ import annotations

import unittest

from v9.consolidation import ConsolidationGate, RuntimeConsolidationAudit
from v9.environment_registry import EnvironmentIdentityRegistry
from v9.grounding import GroundingRegistry
from v9.lineage import LineageOverlayStore
from v9.memory import MultimodalMemoryStore
from v9.multimodal_events import BoundedMultimodalTimeline
from v9.progressive_similarity import ProgressiveSimilarityEngine, ScaleStatistics
from v9.residency import PayloadStore
from v9.runtime import V9ContinuousMemoryRuntime
from v9.scientific_config import ScientificConfig
from v9.snapshot_audit import (
    REQUIRED_V9_KEYS,
    audit_snapshot,
    migrate_legacy_v8_auxiliary,
)
from v9.transfer import EnvironmentNeutralTransferGate
from v9.versioning import VersionedMutationStore


class V9PersistenceAndConsolidationTests(unittest.TestCase):
    def test_legacy_migration_produces_complete_v9_auxiliary_state(self) -> None:
        migrated = migrate_legacy_v8_auxiliary(
            {"generation": 12}, scientific_config_id="cfg"
        )
        result = audit_snapshot(
            migrated,
            expected_scientific_config_id="cfg",
            allow_legacy_v8=False,
        )
        self.assertTrue(result.compatible)
        self.assertTrue(REQUIRED_V9_KEYS.issubset(migrated))
        object_versions = migrated["object_versions"]
        self.assertIsInstance(object_versions, dict)
        self.assertEqual(object_versions["graph_generation"], 12)

    def test_incomplete_v9_snapshot_is_rejected(self) -> None:
        payload = migrate_legacy_v8_auxiliary({}, scientific_config_id="cfg")
        payload.pop("grounding_state")
        result = audit_snapshot(
            payload,
            expected_scientific_config_id="cfg",
            allow_legacy_v8=False,
        )
        self.assertFalse(result.compatible)
        self.assertEqual(result.missing_keys, ("grounding_state",))

    def test_scientific_config_freezes_v9_research_parameters(self) -> None:
        config = ScientificConfig()
        self.assertEqual(config.progressive_radii, (1, 2, 4, 8))
        self.assertEqual(dict(config.beta_by_radius)[1], 1.0)
        self.assertEqual(
            config.symbol_behavior_gate, "G4_LOCAL_G5_CROSS_ENVIRONMENT"
        )
        self.assertTrue(config.runtime_consolidation_requires_empirical_gates)

    def test_runtime_auxiliary_payload_is_complete_and_auditable(self) -> None:
        runtime = V9ContinuousMemoryRuntime.__new__(V9ContinuousMemoryRuntime)
        runtime.scientific_config = ScientificConfig()
        runtime.environment_registry = EnvironmentIdentityRegistry()
        runtime.symbol_codecs = {}
        runtime.multimodal_timeline = BoundedMultimodalTimeline()
        runtime.multimodal_memory = MultimodalMemoryStore()
        runtime.versioned_mutations = VersionedMutationStore()
        runtime.lineage_state = LineageOverlayStore()
        runtime.normalization_state = ScaleStatistics()
        runtime.progressive_similarity = ProgressiveSimilarityEngine()
        runtime.grounding_state = GroundingRegistry()
        runtime.payload_state = PayloadStore()
        runtime.transfer_state = EnvironmentNeutralTransferGate()
        runtime._legacy_v8_auxiliary_migrated = False
        payload = runtime._v9_auxiliary_payload()
        result = audit_snapshot(
            payload,
            expected_scientific_config_id=runtime.scientific_config.config_id,
            allow_legacy_v8=False,
        )
        self.assertTrue(result.compatible)
        self.assertTrue(REQUIRED_V9_KEYS.issubset(payload))
        self.assertTrue(payload["v8_runtime_authority_retained"])

    def test_consolidation_refuses_historical_layer_removal_before_empirical_gates(self) -> None:
        audit = RuntimeConsolidationAudit(
            ("sampling_transfer_v833", "research_integrity_v878")
        )
        with self.assertRaises(RuntimeError):
            audit.assert_removal_allowed()

        ready = ConsolidationGate(True, True, True, True)
        unlocked = RuntimeConsolidationAudit(
            ("sampling_transfer_v833", "research_integrity_v878"), ready
        )
        unlocked.assert_removal_allowed()
        rows = unlocked.audit()
        self.assertTrue(rows)
        self.assertTrue(any(row.regression_required for row in rows))
        self.assertTrue(any(row.removal_allowed for row in rows))


if __name__ == "__main__":
    unittest.main()
