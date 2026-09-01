from __future__ import annotations

from types import SimpleNamespace
import unittest

from v8.environment_contract import BoundaryEvent, BoundaryScope
from v8.model import stable_u64
from v9.environments.alfred_adapter import AlfredAdapter
from v9.environments.babyai_adapter import BabyAIAdapter
from v9.memory import PayloadAvailabilityState, PayloadUid
from v9.research_grounding import GroundingCondition, GroundingControlRunner, H16Evaluator, H16Metrics
from v9.residency import PayloadProvenance, PayloadStore, ResidencyState
from v9.snapshot_audit import V9_AUXILIARY_SCHEMA_VERSION, audit_snapshot, migrate_legacy_v8_auxiliary
from v9.transfer import EnvironmentNeutralTransferGate, TransferKey


class FakeBabyAI:
    def __init__(self) -> None: self.action_space = SimpleNamespace(n=3); self.steps = 0
    def reset(self): self.steps = 0; return {"image": ((1,),), "mission": "go to red ball"}, {}
    def step(self, action: int): self.steps += 1; terminated = self.steps >= 2; return {"image": ((self.steps,),), "mission": "go to red ball"}, (1.0 if terminated else 0.0), terminated, False, {}


class FakeAlfredBackend:
    environment_name = "fake-alfred"
    def __init__(self) -> None: self.step_count = 0
    def reset(self): self.step_count = 0; return b"world-0", "pick object"
    def available_actions(self): return (2, 4)
    def step(self, action: int): self.step_count += 1; return b"world-1", "pick object", BoundaryEvent(BoundaryScope.EPISODE, +1, False)


class GroundingExperimentTests(unittest.TestCase):
    def test_matched_controls_use_same_seed_and_mechanic(self) -> None:
        trials = GroundingControlRunner().matched_trials((1, 2)); self.assertEqual(set(trials), set(GroundingCondition))
        for rows in trials.values(): self.assertEqual(tuple(row.seed for row in rows), (1, 2)); self.assertTrue(all(row.mechanic == "advance" for row in rows))
        self.assertTrue(all(row.symbol_observations == 0 for row in trials[GroundingCondition.C0_INTERACTION_ONLY])); self.assertTrue(all(row.world_observations == 0 for row in trials[GroundingCondition.C1_SYMBOLS_ONLY]))

    def test_h16_requires_all_controls_and_c2_separation(self) -> None:
        evaluator = H16Evaluator(); rows = [H16Metrics(GroundingCondition.C0_INTERACTION_ONLY, 5, 0.0, 0.0, 0.0, 0.1), H16Metrics(GroundingCondition.C1_SYMBOLS_ONLY, 5, 0.0, 0.0, 0.0, 0.1), H16Metrics(GroundingCondition.C2_ALIGNED, 5, 0.3, 0.2, 0.2, 0.0), H16Metrics(GroundingCondition.C3_SHUFFLED, 5, 0.0, 0.0, 0.0, 0.1)]
        decision = evaluator.evaluate(rows); self.assertTrue(decision.interpretable); self.assertTrue(decision.c2_separates_controls); self.assertFalse(evaluator.evaluate(rows[:-1]).interpretable)


class AdapterTests(unittest.TestCase):
    def test_babyai_exposes_raw_instruction_bytes_only(self) -> None:
        adapter = BabyAIAdapter(FakeBabyAI()); observation = adapter.reset(); self.assertEqual(observation.instruction_bytes, b"go to red ball"); self.assertEqual(len(adapter.instruction_symbols()), len(observation.instruction_bytes)); self.assertEqual(adapter.available_actions(), (0, 1, 2)); adapter.step(0); adapter.step(0); self.assertTrue(adapter.cognitive_boundary_event().positive)
    def test_alfred_uses_target_local_actions_and_external_payload(self) -> None:
        payloads = PayloadStore(max_hot_payload_bytes=1024, max_hot_payloads=4); adapter = AlfredAdapter(FakeAlfredBackend(), payload_store=payloads); observation = adapter.reset(); self.assertEqual(adapter.available_actions(), (2, 4)); self.assertEqual(len(adapter.instruction_symbols()), len(b"pick object")); self.assertIsNotNone(payloads.payload(observation.payload_uid))
        with self.assertRaises(ValueError): adapter.step(1)
        adapter.step(2); self.assertTrue(adapter.cognitive_boundary_event().positive)


class TransferResidencySnapshotTests(unittest.TestCase):
    def test_transfer_is_target_scoped_and_never_reuses_raw_source_action(self) -> None:
        gate = EnvironmentNeutralTransferGate(); key_a = TransferKey(1, 2, 99); key_b = TransferKey(1, 3, 99); gate.observe_result(key_a, held_out=True, success=True); admitted = gate.evaluate(key_a, structurally_admissible=True, target_available_actions=(7, 8), target_grounded_action=8); other = gate.evaluate(key_b, structurally_admissible=True, target_available_actions=(7, 8), target_grounded_action=8); self.assertTrue(admitted.admitted); self.assertEqual(admitted.target_action, 8); self.assertFalse(other.admitted); gate.observe_result(key_a, held_out=True, success=False); self.assertEqual(gate.trust(key_b), 0)
    def test_payload_pressure_is_bounded_and_provenance_survives_retirement(self) -> None:
        store = PayloadStore(max_hot_payload_bytes=3, max_hot_payloads=1)
        for index, raw in enumerate((b"aa", b"bb"), start=1):
            digest = stable_u64(raw, person=b"v9-payload-digest"); store.register(PayloadProvenance(index, 10, 20, index, 30, PayloadUid(digest), digest, PayloadAvailabilityState.EXTERNAL), raw)
        self.assertLessEqual(store.hot_bytes, 3); first = store.provenance(1); self.assertIn(first.residency, {ResidencyState.COMPACT, ResidencyState.RETIRED_PAYLOAD, ResidencyState.ARCHIVED}); self.assertEqual(first.environment_instance_id, 10); self.assertEqual(first.episode_id, 20); self.assertEqual(PayloadStore.from_state_dict(store.state_dict()).state_dict(), store.state_dict())
    def test_snapshot_audit_requires_complete_v9_state_and_explicit_legacy_migration(self) -> None:
        legacy = {"generation": 7}; audit = audit_snapshot(legacy, allow_legacy_v8=True); self.assertTrue(audit.compatible); self.assertTrue(audit.legacy_migration); migrated = migrate_legacy_v8_auxiliary(legacy, scientific_config_id="abc"); self.assertEqual(migrated["v9_auxiliary_state_version"], V9_AUXILIARY_SCHEMA_VERSION); self.assertTrue(audit_snapshot(migrated, expected_scientific_config_id="abc").compatible); self.assertFalse(audit_snapshot(migrated, expected_scientific_config_id="other").compatible)


if __name__ == "__main__": unittest.main()
