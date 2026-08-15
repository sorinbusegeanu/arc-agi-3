from __future__ import annotations

import unittest

from v8.arena import NodeRecord
from v8.hypotheses import evaluate_live_hypothesis_statuses
from v8.model import MemoryLevel, MemoryType, MemoryUid


class FakeReadView:
    def __init__(self, rows: tuple[NodeRecord, ...]) -> None:
        self.rows = rows

    def node_records(self, *, level=None):
        if level is None:
            return self.rows
        wanted = int(level)
        return tuple(row for row in self.rows if int(row.level) == wanted)


def node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key_parts: tuple[int, ...],
    *,
    support: int = 2,
    future: float = 0.0,
    prediction_error: float = 0.0,
    transfer: float = 0.0,
    validation: int = 0,
) -> NodeRecord:
    uid = MemoryUid.from_key(level, memory_type, key_parts)
    return NodeRecord(
        uid=uid,
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key_parts,
        support_count=support,
        significance_sum=1.0,
        prediction_error_sum=prediction_error,
        learning_value_sum=1.0,
        transfer_prior_sum=transfer,
        explanatory_sum=0.0,
        future_option_sum=future,
        score_weight=1.0,
        updated_watermark=1,
        cognitive_state=0,
        validation_state=validation,
    )


class LiveHypothesisTests(unittest.TestCase):
    def test_empty_graph_has_insufficient_evidence(self) -> None:
        statuses = evaluate_live_hypothesis_statuses(FakeReadView(()))
        self.assertEqual(set(statuses.values()), {"INSUFFICIENT_EVIDENCE"})

    def test_structural_graph_produces_partial_evidence_instead_of_placeholder_all_insufficient(self) -> None:
        outcome_parent = (901, 902)
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 1, 20), future=1.0),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (11, 2, 21), future=0.0),
            node(MemoryLevel.M2, MemoryType.FAMILY, (30,)),
            node(MemoryLevel.M3, MemoryType.ROLE, (30, 40, 1), future=1.0),
            node(MemoryLevel.M4, MemoryType.CONCEPT, (30, 1), future=1.0),
            node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (50, 51, 20, 1), future=1.0),
            node(MemoryLevel.M6, MemoryType.OUTCOME, (20, 1, 2), future=1.0),
            node(MemoryLevel.M7, MemoryType.STRATEGY, (1001, *outcome_parent, 7)),
            node(MemoryLevel.M7, MemoryType.STRATEGY, (1002, *outcome_parent, 8)),
        )
        statuses = evaluate_live_hypothesis_statuses(FakeReadView(rows))

        for hypothesis_id in ("H01", "H03", "H04", "H05", "H07", "H08", "H09", "H10", "H12", "H13", "H14"):
            self.assertEqual(statuses[hypothesis_id], "PARTIALLY_VALID", hypothesis_id)

        for hypothesis_id in ("H02", "H06", "H11", "H15"):
            self.assertEqual(statuses[hypothesis_id], "INSUFFICIENT_EVIDENCE", hypothesis_id)

    def test_transfer_and_prediction_proxy_evidence_can_be_reported_partial(self) -> None:
        rows = (
            node(
                MemoryLevel.M1,
                MemoryType.CONTINGENCY,
                (10, 1, 20),
                prediction_error=1.0,
            ),
            node(
                MemoryLevel.M3,
                MemoryType.ROLE,
                (30, 40, 1),
                transfer=1.0,
            ),
            node(
                MemoryLevel.M4,
                MemoryType.CONCEPT,
                (30, 1),
                transfer=1.0,
                validation=1,
            ),
        )
        statuses = evaluate_live_hypothesis_statuses(FakeReadView(rows))
        self.assertEqual(statuses["H02"], "PARTIALLY_VALID")
        self.assertEqual(statuses["H06"], "PARTIALLY_VALID")
        self.assertEqual(statuses["H11"], "PARTIALLY_VALID")
        self.assertEqual(statuses["H15"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
