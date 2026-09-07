from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.learning_fixes_v088_target_lineage_grounding_fix import (
    _canonical_role_rows,
    _normalized_role_reference,
    _role_lineage_rows,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.similarity import BoundedNeighborhoodSimilarity, NeighborhoodDescriptor
from v8.structural_correspondence import StructuralCorrespondenceEstimator


class V8TargetLineageGroundingTests(unittest.TestCase):
    def test_m4_grounding_uses_normalized_m3_role_lineage(self):
        concept_uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (11, 0))
        role_uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (22, 0))
        carrier_uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.CARRIER, (22, 33, 0))
        concept = SimpleNamespace(
            uid=concept_uid,
            level=MemoryLevel.M4,
            memory_type=MemoryType.CONCEPT,
            key_parts=(11, 0),
            updated_watermark=100,
        )
        role = SimpleNamespace(
            uid=role_uid,
            level=MemoryLevel.M3,
            memory_type=MemoryType.ROLE,
            key_parts=(22, 0),
            updated_watermark=90,
        )
        carrier = SimpleNamespace(
            uid=carrier_uid,
            level=MemoryLevel.M3,
            memory_type=MemoryType.CARRIER,
            key_parts=(22, 33, 0),
            updated_watermark=80,
        )
        by_uid = {concept_uid: concept, role_uid: role, carrier_uid: carrier}
        edges = (
            SimpleNamespace(
                source_uid=concept_uid,
                target_uid=role_uid,
                relation_type=RelationType.EXPLAINS,
            ),
            SimpleNamespace(
                source_uid=role_uid,
                target_uid=carrier_uid,
                relation_type=RelationType.EXPLAINS,
            ),
            SimpleNamespace(
                source_uid=concept_uid,
                target_uid=carrier_uid,
                relation_type=RelationType.DEPENDS_ON,
            ),
        )

        lineage = _role_lineage_rows(concept_uid, by_uid, edges)
        self.assertEqual(tuple(row.uid for row in lineage), (role_uid,))

        reference = _normalized_role_reference(lineage[0])
        descriptor = reference["descriptor"]
        self.assertEqual(int(descriptor.level), int(MemoryLevel.M3))
        self.assertEqual(
            descriptor.outgoing_relations,
            ((int(RelationType.EXPLAINS), 2),),
        )
        self.assertEqual(descriptor.incoming_relations, ())
        self.assertEqual(descriptor.dependency_signature, 0)

        target = NeighborhoodDescriptor(
            uid=MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (99, 0)),
            level=int(MemoryLevel.M3),
            memory_type=int(MemoryType.ROLE),
            incoming_relations=(),
            outgoing_relations=((int(RelationType.EXPLAINS), 2),),
            neighbor_levels=((int(MemoryLevel.M3), 2),),
            neighbor_types=((int(MemoryType.CARRIER), 2),),
            dependency_signature=0,
            enable_block_signature=0,
            future_option_bucket=0,
            consequence_bucket=0,
            context_bucket=0,
            descriptor_version=1,
        )
        similarity = BoundedNeighborhoodSimilarity.score(descriptor, target)
        self.assertGreaterEqual(similarity.score, 0.65)

        counter = reference["structural_counter"]
        preserved, mismatched, mapping_size, epsilon = StructuralCorrespondenceEstimator._error(
            counter, counter
        )
        self.assertEqual(preserved, 2)
        self.assertEqual(mismatched, 0)
        self.assertEqual(mapping_size, 1)
        self.assertEqual(epsilon, 0.0)

    def test_persisted_m4_without_parent_edge_recovers_role_from_canonical_key(self):
        concept_uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (44, 1))
        matching_role_uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (44, 1))
        other_role_uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (45, 1))
        concept = SimpleNamespace(
            uid=concept_uid,
            level=MemoryLevel.M4,
            memory_type=MemoryType.CONCEPT,
            key_parts=(44, 1),
            updated_watermark=100,
        )
        matching_role = SimpleNamespace(
            uid=matching_role_uid,
            level=MemoryLevel.M3,
            memory_type=MemoryType.ROLE,
            key_parts=(44, 1, 999),
            updated_watermark=90,
        )
        other_role = SimpleNamespace(
            uid=other_role_uid,
            level=MemoryLevel.M3,
            memory_type=MemoryType.ROLE,
            key_parts=(45, 1),
            updated_watermark=90,
        )
        by_uid = {
            concept_uid: concept,
            matching_role_uid: matching_role,
            other_role_uid: other_role,
        }

        self.assertEqual(_role_lineage_rows(concept_uid, by_uid, ()), ())
        recovered = _canonical_role_rows(concept, by_uid)
        self.assertEqual(tuple(row.uid for row in recovered), (matching_role_uid,))


if __name__ == "__main__":
    unittest.main()
