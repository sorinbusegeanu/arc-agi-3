from __future__ import annotations

import unittest

import v8
from v8.arena import EdgeRecord, NodeRecord
from v8.intelligence_loop_experiment import BehavioralMetrics, audit_intelligence_chain, run_intelligence_loop_experiment
from v8.intelligence_loop_v087 import V087GenerativeCompressionEstimator, V087RelationalRoleEstimator, process_replay_cognition
from v8.lifecycle import LifecycleController
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState
from v8.peers_v82 import V82DevelopmentalPeerSupervisor
from v8.promotion import EvidenceGatedPromotionEngine
from v8.structural_events import NormalizedPrimitive, StructuralFact


def node(level, memory_type, key, *, support=4, significance=1.0, learning=1.0, transfer=0.5, explanatory=1.0, future=0.0, cognitive=CognitiveState.ACTIVE, validation=ValidationState.STRUCTURAL, game_mask=3, success=0.0, cost=0.0, attempts=0.0):
    uid = MemoryUid.from_key(level, memory_type, key)
    return NodeRecord(uid, (uid.hi ^ uid.lo) & ((1 << 64) - 1), int(level), int(memory_type), tuple(key), int(support), float(significance), 0.0, float(learning), float(transfer), float(explanatory), float(future), 1.0, 10, int(game_mask), int(cognitive), int(validation), float(success), float(cost), float(attempts))


def edge(source, relation, target, *, score=0.0):
    return EdgeRecord(source.uid if isinstance(source, NodeRecord) else source, int(relation), target.uid if isinstance(target, NodeRecord) else target, 1, 10, float(score), 1.0 if score else 0.0, 0, 0)


class FakeReadView:
    def __init__(self, nodes, edges=()):
        self.nodes, self.edges = tuple(nodes), tuple(edges)

    def node_records(self, *, level=None):
        return self.nodes if level is None else tuple(row for row in self.nodes if int(row.level) == int(level))

    def edge_records(self):
        return self.edges

    def source_games(self, uid, *, max_depth=8):
        del max_depth
        row = next((item for item in self.nodes if item.uid == uid), None)
        return frozenset() if row is None else frozenset(index for index in range(64) if int(row.game_mask) & (1 << index))


class V087IntelligenceLoopTests(unittest.TestCase):
    def test_compression_discovers_new_m2_from_m1n(self):
        a = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 101).token
        b = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 202).token
        m1a = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (a,), support=4)
        m1b = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (b,), support=5)
        proposals = V087GenerativeCompressionEstimator().discover((m1a, m1b), budget=8)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(set(proposals[0].parents), {m1a.uid, m1b.uid})
        candidates = EvidenceGatedPromotionEngine().propose((m1a, m1b), (), budget=16)
        self.assertTrue(any(item.evidence_kind == "generative_compression" for item in candidates))

    def test_relational_roles_ignore_raw_family_identity(self):
        m1a = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (11, 1, 2, 12))
        m1b = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (21, 1, 2, 22))
        f1 = node(MemoryLevel.M2, MemoryType.FAMILY, (1001, 0))
        f2 = node(MemoryLevel.M2, MemoryType.FAMILY, (2002, 0))
        c1 = node(MemoryLevel.M3, MemoryType.CARRIER, (111, 501, 0))
        c2 = node(MemoryLevel.M3, MemoryType.CARRIER, (222, 502, 0))
        edges = (edge(c1, RelationType.EXPLAINS, f1), edge(c1, RelationType.EXPLAINS, m1a), edge(c2, RelationType.EXPLAINS, f2), edge(c2, RelationType.EXPLAINS, m1b))
        roles = V087RelationalRoleEstimator().propose_relational((m1a, m1b, f1, f2, c1, c2), edges)
        self.assertEqual(len(roles), 1)
        self.assertEqual(set(roles[0].carriers), {c1.uid, c2.uid})
        self.assertNotIn(111, roles[0].key_parts)
        self.assertNotIn(222, roles[0].key_parts)

    def test_structurally_ready_concept_can_form_probe_scaffold_but_failed_cannot(self):
        candidate = node(MemoryLevel.M4, MemoryType.CONCEPT, (7, 8), support=4, cognitive=CognitiveState.CANDIDATE, validation=ValidationState.STRUCTURAL)
        failed = node(MemoryLevel.M4, MemoryType.CONCEPT, (11, 12), support=4, cognitive=CognitiveState.QUARANTINED, validation=ValidationState.FAILED)
        validated = node(MemoryLevel.M4, MemoryType.CONCEPT, (9, 10), support=4, cognitive=CognitiveState.VALIDATED, validation=ValidationState.VALIDATED)
        engine = EvidenceGatedPromotionEngine()
        self.assertTrue(any(int(item.level) == int(MemoryLevel.M5) for item in engine.propose((candidate,), (), budget=16)))
        self.assertFalse(any(int(item.level) == int(MemoryLevel.M5) for item in engine.propose((failed,), (), budget=16)))
        self.assertTrue(any(int(item.level) == int(MemoryLevel.M5) for item in engine.propose((validated,), (), budget=16)))

    def test_failed_transfer_is_target_scoped_and_does_not_quarantine_source(self):
        concept = node(MemoryLevel.M4, MemoryType.CONCEPT, (31, 32), support=5, cognitive=CognitiveState.VALIDATED, validation=ValidationState.VALIDATED)
        consequence = node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (concept.uid.hi, concept.uid.lo, 44, 1), cognitive=CognitiveState.ACTIVE, validation=ValidationState.STRUCTURAL)
        view = FakeReadView((concept, consequence), (edge(consequence, RelationType.EXPLAINS, concept),))
        submitted = []
        supervisor = V82DevelopmentalPeerSupervisor(read_view=view, submit_proposal=submitted.append, watermark=lambda: 20, generation=lambda: 1, interval_seconds=100.0)
        for target in (101, 102, 103):
            supervisor.record_transfer_trial(concept.uid, target_game_hash=target, metric_on=0.0, metric_off=1.0, formation_games=(1, 2))
        relation_updates = [item for item in submitted if int(item.memory_type) == int(MemoryType.TRANSFER_EVIDENCE)]
        self.assertEqual(len(relation_updates), 3)
        self.assertTrue(all(item.parent_uid == concept.uid for item in relation_updates))
        self.assertTrue(all(item.transfer_prior_sum < 0.0 for item in relation_updates))
        self.assertTrue(all(item.validation_state == int(ValidationState.TESTED) for item in relation_updates))
        self.assertFalse(any(item.uid == concept.uid for item in submitted))
        self.assertFalse(any(item.uid == consequence.uid for item in submitted))
        state = supervisor.state_dict()
        restored = V82DevelopmentalPeerSupervisor(read_view=view, submit_proposal=lambda proposal: None, watermark=lambda: 20, generation=lambda: 1, interval_seconds=100.0)
        restored.load_state(state)
        self.assertEqual(len(restored.transfer.trials(concept.uid)), 3)

    def test_failed_validation_never_receives_lifecycle_promotion(self):
        failed = node(MemoryLevel.M4, MemoryType.CONCEPT, (41, 42), support=10, significance=1.0, learning=1.0, transfer=1.0, explanatory=1.0, cognitive=CognitiveState.ACTIVE, validation=ValidationState.FAILED)
        decision = LifecycleController().decide(failed)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.QUARANTINED))

    def test_replay_can_create_abstraction_without_new_experience(self):
        a = StructuralFact(NormalizedPrimitive.COMPONENT_REMOVED, 501).token
        b = StructuralFact(NormalizedPrimitive.COMPONENT_REMOVED, 502).token
        m1a = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (a,), support=5)
        m1b = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (b,), support=5)
        submitted = []
        supervisor = V82DevelopmentalPeerSupervisor(read_view=FakeReadView((m1a, m1b)), submit_proposal=submitted.append, watermark=lambda: 50, generation=lambda: 2, interval_seconds=100.0)
        metrics = process_replay_cognition(supervisor)
        self.assertGreater(metrics["processed"], 0)
        self.assertGreater(metrics["new_memories"], 0)
        self.assertTrue(any(int(item.level) == int(MemoryLevel.M2) for item in submitted))

    def test_complete_chain_audit_and_ablation_contract(self):
        m1 = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1 << 63 | 2,))
        m2 = node(MemoryLevel.M2, MemoryType.FAMILY, (1 << 63 | 2, 0))
        m3a = node(MemoryLevel.M3, MemoryType.ROLE, (1, 2, 0, 0))
        m3b = node(MemoryLevel.M3, MemoryType.ROLE, (3, 4, 0, 0))
        m4 = node(MemoryLevel.M4, MemoryType.CONCEPT, (5, 6), cognitive=CognitiveState.VALIDATED, validation=ValidationState.VALIDATED)
        m5 = node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (m4.uid.hi, m4.uid.lo, 7, 1))
        m6 = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 7, 8), cognitive=CognitiveState.ACTIVE)
        m7 = node(MemoryLevel.M7, MemoryType.STRATEGY, (9, m6.uid.hi, m6.uid.lo, 10), cognitive=CognitiveState.ACTIVE, success=3.0, cost=6.0, attempts=4.0)
        edges = (edge(m2, RelationType.EXPLAINS, m1), edge(m3a, RelationType.EXPLAINS, m2), edge(m3b, RelationType.SIMILAR_TO, m3a, score=0.8), edge(m3b, RelationType.TRANSFER_CORRESPONDENCE, m3a, score=0.8), edge(m4, RelationType.EXPLAINS, m3a), edge(m5, RelationType.EXPLAINS, m4), edge(m6, RelationType.EXPLAINS, m5), edge(m7, RelationType.LEADS_TO, m6))
        nodes = (m1, m2, m3a, m3b, m4, m5, m6, m7)
        self.assertTrue(audit_intelligence_chain(nodes, edges).complete)

        def runner(controls):
            utility = 0.0 if controls.fresh or not controls.memory_enabled else 1.0 + (0.25 if controls.transfer_enabled else 0.0) + (0.25 if controls.concepts_enabled else 0.0) + (0.10 if controls.replay_enabled else 0.0)
            return BehavioralMetrics(1.0 if utility else 10.0, 1.0 if utility else 0.0, utility, min(1.0, utility), 2.0 if utility else 20.0, 0.25 if controls.transfer_enabled and utility else 0.0)

        experiment = run_intelligence_loop_experiment(nodes=nodes, edges=edges, episode_runner=runner)
        self.assertTrue(experiment.audit.complete)
        self.assertGreater(experiment.memory_effect, 0.0)
        self.assertAlmostEqual(experiment.restored_delta, 0.0)
        self.assertGreater(experiment.results["normal"].utility, experiment.results["replay_off"].utility)


if __name__ == "__main__":
    unittest.main()
