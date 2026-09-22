from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConcreteAdmissionDecision:
    retain: bool
    reason: str


def _power_of_two(value: int) -> bool:
    selected = int(value)
    return selected > 0 and (selected & (selected - 1)) == 0


def should_retain_concrete(
    scientific: Any,
    *,
    prior_support: int,
    context: Any | None = None,
    isf_static: tuple[float, float, float, float, float] | None = None,
    retained_representatives: int = 0,
    novel_context: bool = False,
    force_novel: bool = False,
) -> ConcreteAdmissionDecision:
    if not bool(getattr(scientific, "concrete_admission_enabled", True)):
        return ConcreteAdmissionDecision(True, "disabled")

    representatives = max(
        1, int(getattr(scientific, "concrete_admission_representatives_per_signature", 4))
    )
    support = max(0, int(prior_support))
    retained = max(0, int(retained_representatives))

    if context is not None:
        if (
            bool(getattr(context, "task_success", False))
            or bool(getattr(context, "task_failure", False))
            or bool(getattr(context, "task_truncated", False))
        ):
            return ConcreteAdmissionDecision(True, "boundary")
        if int(getattr(context, "primary_valence", 0) or 0) != 0:
            return ConcreteAdmissionDecision(True, "valence")

    if isf_static is not None:
        _pvi, future_options, prediction_error, _transfer, _explanatory = isf_static
        if abs(float(prediction_error)) >= float(
            getattr(scientific, "concrete_admission_prediction_error_threshold", 0.5)
        ):
            return ConcreteAdmissionDecision(True, "prediction_error")
        if abs(float(future_options)) >= float(
            getattr(scientific, "concrete_admission_future_option_threshold", 1.0)
        ):
            return ConcreteAdmissionDecision(True, "future_option_change")

    # A one-off structural novelty is not enough to justify permanent concrete
    # storage. Its normalized support is still counted, and a recurrence can
    # promote it into the representative reservoir.
    if support == 0:
        return ConcreteAdmissionDecision(False, "unconfirmed_novelty")

    if retained < representatives:
        return ConcreteAdmissionDecision(
            True,
            "recurrent_novel_context" if novel_context else "recurrent_representative",
        )

    next_support = support + 1
    milestone = bool(
        getattr(scientific, "concrete_admission_support_milestones", True)
    ) and _power_of_two(next_support)
    if milestone and force_novel:
        return ConcreteAdmissionDecision(True, "novel_dependent_milestone")
    if milestone and novel_context:
        return ConcreteAdmissionDecision(True, "novel_context_milestone")
    if milestone:
        return ConcreteAdmissionDecision(True, "support_milestone")

    if force_novel:
        return ConcreteAdmissionDecision(False, "novel_dependent_redundant")
    if novel_context:
        return ConcreteAdmissionDecision(False, "novel_context_redundant")
    return ConcreteAdmissionDecision(False, "redundant_support")


__all__ = ["ConcreteAdmissionDecision", "should_retain_concrete"]
