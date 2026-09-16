from __future__ import annotations

from typing import Any

from v9.cognition.action_selection import adaptive_epsilon


_STATE_FLOORS = {
    "PROBING": 0.25,
    "LOW_EVIDENCE": 0.20,
    "VIABLE": 0.0,
    "VIABILITY_ANOMALY": 0.35,
    "RECOVERING": 0.20,
}


def viability_adjusted_epsilon(
    base_epsilon: float,
    branching_factor: int,
    *,
    state: str,
    coverage: float,
    uncertainty: float,
    learning_progress: float,
    profile_rate: float = 0.0,
) -> float:
    stagnation = max(0.0, min(1.0, 1.0 - max(0.0, float(learning_progress))))
    generic = adaptive_epsilon(
        float(base_epsilon),
        max(1, int(branching_factor)),
        coverage=max(0.0, min(1.0, float(coverage))),
        uncertainty=max(0.0, min(1.0, float(uncertainty))),
        stagnation=stagnation,
    )
    floor = float(_STATE_FLOORS.get(str(state), 0.0))
    return max(generic, min(0.50, max(floor, float(profile_rate))))


def install_adaptive_exploration(multiprocess_module: Any) -> None:
    """Route actor action selection through persisted viability state."""
    if getattr(multiprocess_module, "_viability_exploration_installed", False):
        return
    original_choose_action = multiprocess_module.choose_action

    def choose_action(view: Any, actions: tuple[int, ...], **kwargs: Any) -> int:
        environment_id = kwargs.get("target_environment_id")
        profile = None
        viability = getattr(view, "viability", None)
        if environment_id is not None and callable(viability):
            profile = viability(int(environment_id))
        if profile is not None:
            counts = kwargs.get("context_action_counts") or {}
            tried = sum(int(counts.get(int(action), 0)) > 0 for action in actions)
            local_coverage = tried / max(1, len(actions))
            coverage = min(float(profile.action_coverage), local_coverage) if counts else float(profile.action_coverage)
            kwargs["epsilon"] = viability_adjusted_epsilon(
                float(kwargs.get("epsilon", 0.0)),
                len(actions),
                state=str(profile.state),
                coverage=coverage,
                uncertainty=float(profile.policy_uncertainty),
                learning_progress=float(profile.learning_progress),
                profile_rate=float(profile.effective_exploration_rate),
            )
            kwargs["stagnation"] = max(
                float(kwargs.get("stagnation", 0.0)),
                max(0.0, 1.0 - max(0.0, float(profile.learning_progress))),
            )
        return int(original_choose_action(view, actions, **kwargs))

    multiprocess_module.choose_action = choose_action
    multiprocess_module._viability_exploration_installed = True
