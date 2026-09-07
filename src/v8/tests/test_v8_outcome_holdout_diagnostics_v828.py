from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.outcome_holdout_diagnostics_v828 import _diagnose_h13_holdout
from v8.outcomes import OutcomeEquivalenceEstimator


class _ReadView:
    def __init__(self, nodes, edges):
        self._nodes = tuple(nodes)
        self._edges = tuple(edges)

    def node_records(self):
        return self._nodes

    def edge_records(self):
        return self._edges


def _m6(uid_lo: int):
    return SimpleNamespace(
        uid=MemoryUid(1, uid_lo),
        level=int(MemoryLevel.M6),
        memory_type=int(MemoryType.OUTCOME),
        key_parts=(1, 2, 1),
        cognitive_state=int(CognitiveState.ACTIVE),
        support_count=12,
    )


def _provenance_edges(root: MemoryUid, world: int, start: int, count: int):
    rows = []
    for offset in range(count):
        lineage = MemoryUid(2, start + offset)
        rows.append(
            SimpleNamespace(
                source_uid=root,
                target_uid=lineage,
                relation_type=int(RelationType.EXPLAINS),
            )
        )
        rows.append(
            SimpleNamespace(
                source_uid=lineage,
                target_uid=MemoryUid(0, world),
                relation_type=int(RelationType.GAME_PROVENANCE),
            )
        )
    return rows


class OutcomeHoldoutDiagnosticsV828Tests(unittest.TestCase):
    def test_diagnostics_find_holdout_without_mutating_live_estimator(self) -> None:
        left = _m6(1)
        right = _m6(2)
        edges = []
        edges += _provenance_edges(left.uid, 10, 100, 3)
        edges += _provenance_edges(left.uid, 30, 110, 3)
        edges += _provenance_edges(right.uid, 20, 200, 3)
        edges += _provenance_edges(right.uid, 30, 210, 3)

        live = OutcomeEquivalenceEstimator()
        before = live.state_dict().copy()
        supervisor = SimpleNamespace(
            read_view=_ReadView((left, right), edges),
            outcomes=live,
        )

        snapshot = _diagnose_h13_holdout(supervisor)

        self.assertEqual(snapshot["class_count"], 1)
        self.assertEqual(snapshot["roots_with_provenance"], 2)
        self.assertGreaterEqual(snapshot["candidate_worlds_total"], 1)
        self.assertGreaterEqual(snapshot["accepted_candidates"], 1)
        self.assertEqual(live.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
