from __future__ import annotations

from .hypothesisTypes import HypothesisV4


class HypothesisContradictionCheckerV4:
    def is_contradictory(self, left: HypothesisV4, right: HypothesisV4) -> bool:
        if left.hypothesis_id in right.incompatible_with:
            return True
        if right.hypothesis_id in left.incompatible_with:
            return True
        if left.kind == right.kind and "candidate_value" in left.payload and "candidate_value" in right.payload:
            return left.payload.get("candidate_value") != right.payload.get("candidate_value")
        return False

    def pairwise_contradictions(self, hypotheses: tuple[HypothesisV4, ...]) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for left_index, left in enumerate(hypotheses):
            for right in hypotheses[left_index + 1 :]:
                if self.is_contradictory(left, right):
                    pair = tuple(sorted((left.hypothesis_id, right.hypothesis_id)))
                    if pair not in pairs:
                        pairs.append(pair)
        return tuple(sorted(pairs))
