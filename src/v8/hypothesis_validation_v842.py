from __future__ import annotations

from collections import defaultdict
from typing import Iterable


_INSTALLED = False
_OLD_PROXY_TRANSFER_INTERVENTION = "leave_one_memory_out_correspondence_ablation"
_BASE_V82_EVALUATE = None


def _legacy_proxy_transfer(row: object) -> bool:
    return bool(
        str(getattr(row, "evidence_kind", ""))
        in {"transfer_trial_pass", "concept_transfer_pass"}
        and str(getattr(row, "causal_intervention", ""))
        == _OLD_PROXY_TRANSFER_INTERVENTION
    )


def _evaluate_v842(self, evidence: Iterable[object]):
    """Ignore historical structural-score proxy rows in scientific decisions."""
    rows = tuple(row for row in evidence if not _legacy_proxy_transfer(row))
    return _BASE_V82_EVALUATE(self, rows)


def _similarity_evaluate_v842(self, nodes, edges):
    """Keep similarity bounded while reserving comparisons/results for cross-game reuse."""
    nodes = tuple(nodes)
    edges = tuple(edges)
    descriptors = self.descriptors(nodes, edges)
    by_uid = {row.uid: row for row in nodes}
    index = defaultdict(list)
    fallback = defaultdict(list)
    for descriptor in sorted(descriptors.values(), key=lambda item: item.uid):
        index[
            (
                descriptor.level,
                descriptor.memory_type,
                self._relation_bucket(descriptor),
                descriptor.future_option_bucket,
            )
        ].append(descriptor)
        fallback[(descriptor.level, descriptor.memory_type)].append(descriptor)

    dirty = [
        descriptor
        for descriptor in sorted(descriptors.values(), key=lambda item: item.uid)
        if descriptor.descriptor_version
        > self._processed_versions.get(descriptor.uid, -1)
    ]
    results = {}
    for descriptor in dirty:
        candidates = []
        relation_bucket = self._relation_bucket(descriptor)
        for future_delta in (0, -1, 1):
            candidates.extend(
                index.get(
                    (
                        descriptor.level,
                        descriptor.memory_type,
                        relation_bucket,
                        descriptor.future_option_bucket + future_delta,
                    ),
                    (),
                )
            )
        if len(candidates) < self.max_candidates:
            candidates.extend(
                fallback.get((descriptor.level, descriptor.memory_type), ())
            )
        unique_candidates = {
            candidate.uid: candidate
            for candidate in candidates
            if candidate.uid != descriptor.uid
        }
        source_row = by_uid.get(descriptor.uid)
        source_mask = 0 if source_row is None else int(source_row.game_mask)

        def candidate_rank(candidate):
            target_row = by_uid.get(candidate.uid)
            target_mask = 0 if target_row is None else int(target_row.game_mask)
            cross_game = bool(
                source_mask and target_mask and source_mask != target_mask
            )
            overlap = (
                (source_mask & target_mask).bit_count() if cross_game else 64
            )
            return (0 if cross_game else 1, overlap, candidate.uid)

        scored = []
        for candidate in sorted(
            unique_candidates.values(), key=candidate_rank
        )[: self.max_candidates]:
            self.candidate_comparisons += 1
            evidence = self.score(descriptor, candidate)
            if evidence.score >= self.threshold:
                scored.append(evidence)

        ranked = sorted(
            scored,
            key=lambda item: (-item.score, item.source_uid, item.target_uid),
        )

        def is_cross_game(evidence) -> bool:
            left = by_uid.get(evidence.source_uid)
            right = by_uid.get(evidence.target_uid)
            if left is None or right is None:
                return False
            left_mask = int(left.game_mask)
            right_mask = int(right.game_mask)
            return bool(left_mask and right_mask and left_mask != right_mask)

        cross = [row for row in ranked if is_cross_game(row)]
        reserve = max(1, self.top_results // 2)
        chosen = cross[:reserve]
        chosen_keys = {(row.source_uid, row.target_uid) for row in chosen}
        for row in ranked:
            key = (row.source_uid, row.target_uid)
            if key in chosen_keys:
                continue
            chosen.append(row)
            chosen_keys.add(key)
            if len(chosen) >= self.top_results:
                break
        for row in chosen[: self.top_results]:
            results[(row.source_uid, row.target_uid)] = row

        self._processed_versions[descriptor.uid] = descriptor.descriptor_version
        self.processed_descriptors += 1

    return tuple(
        results[key]
        for key in sorted(results, key=lambda pair: (pair[0], pair[1]))
    )


def _hypothesis_status_line_v842(evidence_rows, watermark_value: int) -> str:
    from v8.diagnostics import format_hypothesis_line
    from v8.evaluation_v82 import V82ScientificHypothesisEvaluator

    watermark = int(watermark_value)
    cut = tuple(
        row
        for row in evidence_rows
        if int(getattr(row, "evidence_available_watermark", 0)) <= watermark
        and int(
            getattr(
                row,
                "decision_watermark",
                getattr(row, "evidence_available_watermark", 0),
            )
        )
        <= watermark
    )
    evaluator = V82ScientificHypothesisEvaluator()
    statuses = evaluator.status_map(evaluator.evaluate(cut))
    return format_hypothesis_line(statuses)


def install_hypothesis_validation_v842() -> None:
    global _INSTALLED, _BASE_V82_EVALUATE
    if _INSTALLED:
        return

    from v8 import runtime_observability_v836 as observability
    from v8.evaluation_v82 import V82ScientificHypothesisEvaluator
    from v8.similarity import BoundedNeighborhoodSimilarity

    _BASE_V82_EVALUATE = V82ScientificHypothesisEvaluator.evaluate
    V82ScientificHypothesisEvaluator.evaluate = _evaluate_v842

    BoundedNeighborhoodSimilarity.evaluate = _similarity_evaluate_v842
    observability._hypothesis_status_line = _hypothesis_status_line_v842

    _INSTALLED = True
