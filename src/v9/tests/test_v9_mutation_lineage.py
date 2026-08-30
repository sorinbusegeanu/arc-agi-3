from __future__ import annotations

import random
import unittest

from v9.formation import M2Family, ProvenanceComposition, form_m2, form_m3
from v9.lineage import AuthorityState, LineageOverlayStore, RegimeState
from v9.memory import GroundedRelationKind, M1NRecord, M1NUid, NormalizedChannel
from v9.versioning import ObjectRef, ReadDependency, StateMutationProposal, StateWrite, VersionedMutationStore


class VersionedMutationTests(unittest.TestCase):
    def test_unrelated_global_generation_change_does_not_invalidate_read_set(self) -> None:
        store = VersionedMutationStore()
        dependency = ObjectRef("node", 1)
        store.apply_additive(ObjectRef("node", 2))
        proposal = StateMutationProposal.build("authority", base_graph_generation=0, target_partition_ids=(0,), read_set=(ReadDependency(dependency, 0),), evidence_refs=(10,), causal_watermark=5, writes=(StateWrite(ObjectRef("overlay", 3), 7),))
        self.assertTrue(store.apply_stateful((proposal,))[0].accepted)

    def test_changed_read_dependency_rejects_stateful_proposal(self) -> None:
        store = VersionedMutationStore()
        dependency = ObjectRef("node", 1)
        proposal = StateMutationProposal.build("authority", base_graph_generation=0, target_partition_ids=(0,), read_set=(ReadDependency(dependency, 0),), evidence_refs=(), causal_watermark=2, writes=(StateWrite(ObjectRef("overlay", 3), 7),))
        store.apply_additive(dependency)
        result = store.apply_stateful((proposal,))[0]
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "STALE_READ_SET")
        self.assertEqual(store.read(ObjectRef("overlay", 3))[0], 0)

    def test_random_schedule_has_same_deterministic_result(self) -> None:
        proposals = [StateMutationProposal.build("set", base_graph_generation=0, target_partition_ids=(0,), read_set=(), evidence_refs=(i,), causal_watermark=i, writes=(StateWrite(ObjectRef("x", 1), i),)) for i in (3, 1, 2)]
        first = VersionedMutationStore(); first.apply_stateful(proposals)
        shuffled = list(proposals); random.Random(7).shuffle(shuffled)
        second = VersionedMutationStore(); second.apply_stateful(shuffled)
        self.assertEqual(first.state_dict(), second.state_dict())
        self.assertEqual(first.read(ObjectRef("x", 1))[0], 3)

    def test_cross_partition_rejection_is_all_or_nothing(self) -> None:
        store = VersionedMutationStore(); dep = ObjectRef("node", 1); store.apply_additive(dep)
        proposal = StateMutationProposal.build("cross", base_graph_generation=0, target_partition_ids=(1, 2), read_set=(ReadDependency(dep, 0),), evidence_refs=(), causal_watermark=1, writes=(StateWrite(ObjectRef("a", 1), 1), StateWrite(ObjectRef("b", 2), 2)))
        self.assertFalse(store.apply_stateful((proposal,))[0].accepted)
        self.assertEqual(store.read(ObjectRef("a", 1))[0], 0)
        self.assertEqual(store.read(ObjectRef("b", 2))[0], 0)

    def test_snapshot_preserves_object_versions(self) -> None:
        store = VersionedMutationStore(); store.apply_additive(ObjectRef("node", 1), 2)
        self.assertEqual(VersionedMutationStore.from_state_dict(store.state_dict()).state_dict(), store.state_dict())


class MixedFormationTests(unittest.TestCase):
    def _fact(self, uid: int, channel: NormalizedChannel) -> M1NRecord:
        return M1NRecord(M1NUid(uid), GroundedRelationKind.SYMBOL_PRECEDES_ACTION, channel, ())

    def test_symbol_only_m2_has_no_direct_action_authority(self) -> None:
        family = form_m2((self._fact(1, NormalizedChannel.SYMBOL), self._fact(2, NormalizedChannel.SYMBOL)))[0]
        self.assertEqual(family.composition, ProvenanceComposition.SYMBOL_ONLY)
        self.assertFalse(family.action_authority)

    def test_mixed_recurrence_forms_mixed_family(self) -> None:
        family = form_m2((self._fact(1, NormalizedChannel.SYMBOL), self._fact(2, NormalizedChannel.CROSS_MODAL)))[0]
        self.assertEqual(family.composition, ProvenanceComposition.MIXED)

    def test_m3_convergence_uses_structure_not_raw_family_uid(self) -> None:
        family_a = M2Family(10, 99, (), ProvenanceComposition.WORLD_ONLY, 2)
        family_b = M2Family(20, 99, (), ProvenanceComposition.SYMBOL_ONLY, 2)
        role = form_m3((family_a, family_b))[0]
        self.assertEqual(role.families, (10, 20))
        self.assertEqual(role.composition, ProvenanceComposition.MIXED)


class LineageIsolationTests(unittest.TestCase):
    def test_lineage_mutation_cannot_change_other_lineage_view(self) -> None:
        store = LineageOverlayStore(); store.suspend(11, 100, 7, watermark=20)
        self.assertEqual(store.effective_state(11, 100, 7).regime_state, RegimeState.SUSPENDED)
        other = store.effective_state(11, 200, 7)
        self.assertEqual(other.regime_state, RegimeState.ACTIVE)
        self.assertEqual(other.authority_state, AuthorityState.ACTIVE)

    def test_independent_descendant_stays_active_and_unsupported_enters_probation(self) -> None:
        store = LineageOverlayStore()
        self.assertEqual(store.audit_descendant(12, 100, 7, independent_support=1, watermark=5).regime_state, RegimeState.ACTIVE)
        self.assertEqual(store.audit_descendant(13, 100, 7, independent_support=0, watermark=5).regime_state, RegimeState.PROBATION)

    def test_probation_uses_relevant_opportunities_and_can_restore(self) -> None:
        store = LineageOverlayStore(); store.audit_descendant(13, 100, 7, independent_support=0, watermark=5)
        self.assertEqual(store.retire_if_exhausted(13, 100, 7, required_opportunities=2).regime_state, RegimeState.PROBATION)
        store.observe_relevant_evidence(13, 100, 7)
        self.assertEqual(store.observe_relevant_evidence(13, 100, 7, positive_independent_support=True).regime_state, RegimeState.ACTIVE)

    def test_minimal_cow_only_when_structural_identity_diverges(self) -> None:
        self.assertFalse(LineageOverlayStore.canonical_identity_fork_required(10, 10))
        self.assertTrue(LineageOverlayStore.canonical_identity_fork_required(10, 11))

    def test_lineage_snapshot_round_trip(self) -> None:
        store = LineageOverlayStore(); store.suspend(11, 100, 7, watermark=20)
        self.assertEqual(LineageOverlayStore.from_state_dict(store.state_dict()).state_dict(), store.state_dict())


if __name__ == "__main__":
    unittest.main()
