from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ValidationResult:
    success: bool
    actions_executed: int
    reason: str = ""
    terminal_state: str = ""
    levels_completed: int = 0


class ReplayAdapter(Protocol):
    def validate(self, candidate) -> ValidationResult:
        ...


def _target_reached(env, target) -> bool:
    state = str(getattr(env, "last_outcome_state", ""))
    levels = int(getattr(env, "last_levels_completed", 0))
    if str(target.terminal_state) == "WIN":
        return state == "WIN"
    return levels >= int(target.levels_completed)


class ArcReplayAdapter:
    """ARC-specific replay adapter behind the generic trajectory optimizer API."""

    def validate(self, candidate) -> ValidationResult:
        from v7.environment.arc_adapter import ArcGridEnvironment

        anchor = candidate.source.anchor
        target = candidate.source.target
        env = ArcGridEnvironment(
            game_id=str(anchor.source_id),
            seed=int(anchor.seed),
            env_root=anchor.env_root,
        )
        env.game_wait_seconds = 0.0
        executed = 0
        try:
            for action in anchor.prefix_actions:
                env.step(int(action))
                executed += 1
                if str(getattr(env, "last_outcome_state", "")) == "GAME_OVER":
                    return ValidationResult(
                        False,
                        executed,
                        "anchor_failed",
                        "GAME_OVER",
                        int(getattr(env, "last_levels_completed", 0)),
                    )

            if _target_reached(env, target):
                return ValidationResult(
                    False,
                    executed,
                    "anchor_already_reaches_target",
                    str(getattr(env, "last_outcome_state", "")),
                    int(getattr(env, "last_levels_completed", 0)),
                )

            candidate_steps = 0
            for action in candidate.actions:
                env.step(int(action))
                candidate_steps += 1
                if _target_reached(env, target):
                    return ValidationResult(
                        True,
                        candidate_steps,
                        "target_preserved",
                        str(getattr(env, "last_outcome_state", "")),
                        int(getattr(env, "last_levels_completed", 0)),
                    )
                if str(getattr(env, "last_outcome_state", "")) == "GAME_OVER":
                    return ValidationResult(
                        False,
                        candidate_steps,
                        "candidate_failed",
                        "GAME_OVER",
                        int(getattr(env, "last_levels_completed", 0)),
                    )

            return ValidationResult(
                False,
                candidate_steps,
                "target_not_reached",
                str(getattr(env, "last_outcome_state", "")),
                int(getattr(env, "last_levels_completed", 0)),
            )
        except BaseException as exc:
            return ValidationResult(
                False,
                executed,
                f"{type(exc).__name__}: {exc}",
                str(getattr(env, "last_outcome_state", "")),
                int(getattr(env, "last_levels_completed", 0)),
            )


def validate_candidate(candidate, adapter: ReplayAdapter) -> ValidationResult:
    return adapter.validate(candidate)


def validate_arc_candidate(candidate) -> ValidationResult:
    return validate_candidate(candidate, ArcReplayAdapter())
