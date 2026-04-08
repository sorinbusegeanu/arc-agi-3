from __future__ import annotations

from collections import OrderedDict

from .hypothesisTypes import HypothesisV4


_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


class HypothesisPrunerV4:
    def __init__(self, max_total_hypotheses: int = 8, max_per_kind: int = 3) -> None:
        self.max_total_hypotheses = int(max_total_hypotheses)
        self.max_per_kind = int(max_per_kind)

    def _sort_key(self, item: tuple[int, HypothesisV4]) -> tuple[int, int, int, int]:
        index, hypothesis = item
        return (
            -_CONFIDENCE_RANK[hypothesis.confidence_band],
            -len(hypothesis.supporting_evidence),
            len(hypothesis.contradicting_evidence),
            index,
        )

    def prune(self, revision: int, hypotheses: tuple[HypothesisV4, ...]) -> tuple[HypothesisV4, ...]:
        filtered = [
            (index, hypothesis)
            for index, hypothesis in enumerate(hypotheses)
            if hypothesis.expiry_revision is None or hypothesis.expiry_revision >= revision
        ]
        by_kind: OrderedDict[str, list[tuple[int, HypothesisV4]]] = OrderedDict()
        for item in filtered:
            by_kind.setdefault(item[1].kind, []).append(item)
        kept: list[tuple[int, HypothesisV4]] = []
        for items in by_kind.values():
            kept.extend(sorted(items, key=self._sort_key)[: self.max_per_kind])
        kept = sorted(kept, key=self._sort_key)[: self.max_total_hypotheses]
        kept.sort(key=lambda item: item[0])
        return tuple(hypothesis for _, hypothesis in kept)
