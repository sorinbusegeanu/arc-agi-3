from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha1
from typing import Any


@dataclass(frozen=True)
class ContextContradictionEvent:
    interaction_id: str
    context_signature: str
    action_signature: str | None
    predicted_family_id: str | None
    actual_family_id: str | None
    prediction_confidence: float | None
    context_depth: int
    contradiction_key: str
    suggested_context_depth: int
    reason: str
    split_proposal_id: str | None = None
    differentiating_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextSplitProposal:
    split_proposal_id: str
    parent_context_signature: str
    action_signature: str | None
    suggested_context_depth: int
    contradiction_keys: tuple[str, ...]
    conflicting_actual_families: tuple[str, ...]
    differentiating_features: tuple[str, ...]
    support_count: int
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextContradictionTracker:
    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        min_repeats_for_expansion: int = 2,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_repeats_for_expansion = int(
            min_repeats_for_expansion
        )
        self.events: list[ContextContradictionEvent] = []
        self.by_context: Counter[str] = Counter()
        self.by_context_action: Counter[str] = Counter()
        self.by_contradiction_key: Counter[str] = Counter()
        self.actual_families_by_context_action: dict[
            str, set[str]
        ] = defaultdict(set)
        self.context_features_by_context_action: dict[
            str, list[tuple[str, ...]]
        ] = defaultdict(list)
        self._split_proposals: dict[str, ContextSplitProposal] = {}
        self.prediction_result_count = 0
        self.prediction_error_positive_count = 0
        self.predicted_family_available_count = 0
        self.actual_family_available_count = 0
        self.wrong_prediction_count = 0
        self.confident_wrong_prediction_count = 0
        self.contradiction_suppressed_missing_prediction_count = 0
        self.contradiction_suppressed_missing_actual_count = 0
        self.contradiction_suppressed_low_confidence_count = 0
        self.contradiction_suppressed_correct_or_unknown_count = 0

    def record_prediction_result(
        self,
        *,
        interaction_id: str,
        context_signature: str | None,
        action_signature: str | None,
        predicted_family_id: str | None,
        actual_family_id: str | None,
        prediction_correct: bool | None,
        prediction_confidence: float | None,
        context_depth: int,
        max_context_depth: int | None = None,
    ) -> ContextContradictionEvent | None:
        self.prediction_result_count += 1
        if prediction_correct is False:
            self.prediction_error_positive_count += 1
        if predicted_family_id is not None:
            self.predicted_family_available_count += 1
        if actual_family_id is not None:
            self.actual_family_available_count += 1
        if prediction_correct is not False:
            self.contradiction_suppressed_correct_or_unknown_count += 1
            return None
        if predicted_family_id is None:
            self.contradiction_suppressed_missing_prediction_count += 1
            return None
        if actual_family_id is None:
            self.contradiction_suppressed_missing_actual_count += 1
            return None
        if not context_signature:
            self.contradiction_suppressed_correct_or_unknown_count += 1
            return None
        if str(predicted_family_id) == str(actual_family_id):
            self.contradiction_suppressed_correct_or_unknown_count += 1
            return None
        self.wrong_prediction_count += 1
        if (
            prediction_confidence is not None
            and float(prediction_confidence) < self.min_confidence
        ):
            self.contradiction_suppressed_low_confidence_count += 1
            return None

        self.confident_wrong_prediction_count += 1
        context_action_key = (
            f"{context_signature}|{action_signature}"
        )
        contradiction_key = (
            f"{context_action_key}|"
            f"{predicted_family_id}->{actual_family_id}"
        )
        suggested_context_depth = min(
            int(context_depth) + 1,
            int(max_context_depth or (int(context_depth) + 1)),
        )
        self.by_context[str(context_signature)] += 1
        self.by_context_action[context_action_key] += 1
        self.by_contradiction_key[contradiction_key] += 1
        self.actual_families_by_context_action[
            context_action_key
        ].add(str(actual_family_id))
        features = _context_features(str(context_signature))
        self.context_features_by_context_action[
            context_action_key
        ].append(features)

        proposal = self._maybe_build_split_proposal(
            context_signature=str(context_signature),
            action_signature=action_signature,
            context_action_key=context_action_key,
            suggested_context_depth=suggested_context_depth,
        )
        event = ContextContradictionEvent(
            interaction_id=str(interaction_id),
            context_signature=str(context_signature),
            action_signature=(
                None if action_signature is None
                else str(action_signature)
            ),
            predicted_family_id=str(predicted_family_id),
            actual_family_id=str(actual_family_id),
            prediction_confidence=(
                None
                if prediction_confidence is None
                else float(prediction_confidence)
            ),
            context_depth=int(context_depth),
            contradiction_key=contradiction_key,
            suggested_context_depth=suggested_context_depth,
            reason="confident_wrong_prediction_same_context",
            split_proposal_id=(
                None if proposal is None
                else proposal.split_proposal_id
            ),
            differentiating_features=(
                () if proposal is None
                else proposal.differentiating_features
            ),
        )
        self.events.append(event)
        return event

    def should_expand_context(
        self,
        context_signature: str,
        action_signature: str | None = None,
    ) -> bool:
        if action_signature is not None:
            key = f"{context_signature}|{action_signature}"
            return int(
                self.by_context_action.get(key, 0)
            ) >= self.min_repeats_for_expansion
        return int(
            self.by_context.get(str(context_signature), 0)
        ) >= self.min_repeats_for_expansion

    def split_proposals(self) -> list[ContextSplitProposal]:
        return [
            self._split_proposals[key]
            for key in sorted(self._split_proposals)
        ]

    def mark_split_validated(
        self,
        split_proposal_id: str,
        *,
        improved: bool,
    ) -> None:
        proposal = self._split_proposals.get(
            str(split_proposal_id)
        )
        if proposal is None:
            return
        self._split_proposals[str(split_proposal_id)] = (
            ContextSplitProposal(
                **{
                    **proposal.to_dict(),
                    "status": "validated" if improved else "rejected",
                }
            )
        )

    def summary(self) -> dict[str, Any]:
        top_contradictions = [
            {
                "contradiction_key": key,
                "count": int(count),
            }
            for key, count in self.by_contradiction_key.most_common(20)
        ]
        return {
            "context_contradiction_count": len(self.events),
            "prediction_result_count": int(
                self.prediction_result_count
            ),
            "prediction_error_positive_count": int(
                self.prediction_error_positive_count
            ),
            "predicted_family_available_count": int(
                self.predicted_family_available_count
            ),
            "actual_family_available_count": int(
                self.actual_family_available_count
            ),
            "wrong_prediction_count": int(
                self.wrong_prediction_count
            ),
            "confident_wrong_prediction_count": int(
                self.confident_wrong_prediction_count
            ),
            "contradiction_event_count": len(self.events),
            "contradiction_suppressed_missing_prediction_count": int(
                self.contradiction_suppressed_missing_prediction_count
            ),
            "contradiction_suppressed_missing_actual_count": int(
                self.contradiction_suppressed_missing_actual_count
            ),
            "contradiction_suppressed_low_confidence_count": int(
                self.contradiction_suppressed_low_confidence_count
            ),
            "contradiction_suppressed_correct_or_unknown_count": int(
                self.contradiction_suppressed_correct_or_unknown_count
            ),
            "contradicted_context_count": len(self.by_context),
            "contradicted_context_action_count": len(
                self.by_context_action
            ),
            "repeated_contradiction_count": sum(
                1
                for count in self.by_contradiction_key.values()
                if int(count) >= self.min_repeats_for_expansion
            ),
            "context_split_proposal_count": len(
                self._split_proposals
            ),
            "context_split_proposals": [
                proposal.to_dict()
                for proposal in self.split_proposals()[:20]
            ],
            "top_contradictions": top_contradictions,
        }

    def _maybe_build_split_proposal(
        self,
        *,
        context_signature: str,
        action_signature: str | None,
        context_action_key: str,
        suggested_context_depth: int,
    ) -> ContextSplitProposal | None:
        support = int(
            self.by_context_action.get(context_action_key, 0)
        )
        if support < self.min_repeats_for_expansion:
            return None
        families = sorted(
            self.actual_families_by_context_action[
                context_action_key
            ]
        )
        if len(families) < 2:
            return None

        feature_rows = self.context_features_by_context_action[
            context_action_key
        ]
        differentiating = _differentiating_features(feature_rows)
        proposal_id = (
            "context_split:"
            + sha1(
                f"{context_action_key}|{suggested_context_depth}".encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
        )
        contradiction_keys = tuple(
            sorted(
                key
                for key in self.by_contradiction_key
                if key.startswith(context_action_key + "|")
            )
        )
        proposal = ContextSplitProposal(
            split_proposal_id=proposal_id,
            parent_context_signature=context_signature,
            action_signature=action_signature,
            suggested_context_depth=suggested_context_depth,
            contradiction_keys=contradiction_keys,
            conflicting_actual_families=tuple(families),
            differentiating_features=differentiating,
            support_count=support,
        )
        self._split_proposals[proposal_id] = proposal
        return proposal


def _context_features(context_signature: str) -> tuple[str, ...]:
    try:
        value = json.loads(context_signature)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = context_signature.split("|")
    if isinstance(value, dict):
        return tuple(
            f"{key}={value[key]}"
            for key in sorted(value)
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            f"f{index}={item}"
            for index, item in enumerate(value)
        )
    return (str(value),)


def _differentiating_features(
    rows: list[tuple[str, ...]],
) -> tuple[str, ...]:
    if len(rows) < 2:
        return ()
    width = max(len(row) for row in rows)
    output: list[str] = []
    for index in range(width):
        values = {
            row[index]
            for row in rows
            if index < len(row)
        }
        if len(values) > 1:
            output.append(f"feature_index:{index}")
    return tuple(output)
