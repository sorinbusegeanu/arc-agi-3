from __future__ import annotations

import v7.developmental_v707 as d

_INSTALLED = False


def _is_online_evidence(evidence) -> bool:
    """Identify evidence emitted by the real online sampling lane.

    Direct fixtures/tools historically construct ContextEpisodeEvidence without
    trajectory/context metadata. They keep v7.0.6 compatibility semantics, while
    production sampling uses v7.0.7 learned future-option significance.
    """
    return bool(
        str(getattr(evidence, "trajectory_segment_id", "") or "")
        or tuple(getattr(evidence, "context_signatures", ()) or ())
        or str(getattr(evidence, "selection_mode", "") or "")
    )


def install_v707_compatibility() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from v7.derivation.pipeline import MemoryLearningPipeline
    from v7.runtime import V7Runtime

    v707_utility = V7Runtime._observed_decision_utility
    legacy_utility = getattr(
        V7Runtime,
        "_v707_original_observed_decision_utility",
        None,
    )
    if legacy_utility is not None:
        @staticmethod
        def _compatible_utility(evidence):
            if not _is_online_evidence(evidence):
                return legacy_utility(evidence)
            return v707_utility(evidence)

        V7Runtime._observed_decision_utility = _compatible_utility

    v707_candidate = MemoryLearningPipeline._m1_candidate
    legacy_candidate = getattr(
        MemoryLearningPipeline,
        "_v707_original_m1_candidate",
        None,
    )
    if legacy_candidate is not None:
        @staticmethod
        def _compatible_candidate(evidence, context_signature):
            if not _is_online_evidence(evidence):
                return legacy_candidate(evidence, context_signature)
            return v707_candidate(evidence, context_signature)

        MemoryLearningPipeline._m1_candidate = _compatible_candidate


__all__ = ["install_v707_compatibility"]
