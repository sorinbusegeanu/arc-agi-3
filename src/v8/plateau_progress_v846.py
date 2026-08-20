from __future__ import annotations

"""v8.46 progress fidelity and optimization-stagnation repair.

Two independent runtime repairs live here:

* adaptive lease progress retains the deepest level actually reached after a lease
  completes, while preserving v8.23 as the public progress-synthesis authority;
* Pareto-frontier churn only restarts solution optimization when the best action
  cost for that scope improves meaningfully, while preserving v8.19's level-wide
  frontier-version coordination.
"""

from dataclasses import dataclass


_INSTALLED = False
_BASE_DEEPEST_LEVEL = None
_BASE_OBSERVE_FRONTIER_CANDIDATE = None
_MAX_LEVEL_REACHED: dict[str, int] = {}


def _record_progress_depth(row) -> None:
    game_id = getattr(row, "game_id", None)
    depth = getattr(row, "max_level_reached", None)
    if game_id is None or depth is None:
        return
    try:
        value = int(depth)
    except (TypeError, ValueError):
        return
    if value < 0:
        return
    game = str(game_id)
    prior = int(_MAX_LEVEL_REACHED.get(game, -1))
    if value > prior:
        _MAX_LEVEL_REACHED[game] = value


@dataclass(frozen=True, slots=True)
class LeaseProgressV846:
    """Lease progress event that retains episode depth in the parent scheduler."""

    worker_id: int
    lease_id: int
    row: object

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        if name == "row":
            _record_progress_depth(value)
        return value


def _deepest_level_v846(row) -> int:
    """Use retained adaptive-lease depth when synthesized rows lost that field."""

    direct = getattr(row, "max_level_reached", None)
    try:
        if direct is not None and int(direct) >= 0:
            return int(direct)
    except (TypeError, ValueError):
        pass

    game_id = getattr(row, "game_id", None)
    if game_id is not None:
        retained = int(_MAX_LEVEL_REACHED.get(str(game_id), -1))
        if retained >= 0:
            return retained
    return int(_BASE_DEEPEST_LEVEL(row))


def _meaningful_cost_improvement(self, prior_winner, current_winner) -> bool:
    if current_winner is None:
        return False
    if prior_winner is None:
        return True
    threshold = max(1, int(self.config.min_meaningful_improvement))
    return int(current_winner.cost) <= int(prior_winner.cost) - threshold


def _observe_frontier_candidate_v846(
    self,
    scope,
    candidate,
    *,
    terminal_state: str,
    generation: int,
) -> bool:
    """Retain Pareto evidence without mistaking same-cost churn for improvement."""

    from v8 import adaptive_learning_allocation_v819 as v819

    with self._lock:
        prior_winner = self.frontier.winner(scope)
        record = self._record(scope.game_id, scope.level)
        previous_state = record.state
        previous_frontier_version = int(record.frontier_version)
        previous_exhausted_version = int(record.optimizer_exhausted_version)
        previous_validations = int(record.validations_since_improvement)
        previous_improvement_generation = int(
            record.last_frontier_improvement_generation
        )
        signals = self._signals.setdefault(
            str(scope.game_id), v819.GamePrioritySignals()
        )
        previous_competence = float(signals.competence_improvement)
        previous_novelty = float(signals.novelty)

    # Preserve the existing v8.19 wrapper: it owns monotonic level-wide frontier
    # versions and transfer-opportunity evidence across distinct Pareto scopes.
    changed = bool(
        _BASE_OBSERVE_FRONTIER_CANDIDATE(
            self,
            scope,
            candidate,
            terminal_state=terminal_state,
            generation=generation,
        )
    )
    if not changed:
        return False

    with self._lock:
        current_winner = self.frontier.winner(scope)
        meaningful = _meaningful_cost_improvement(
            self, prior_winner, current_winner
        )
        if meaningful:
            return True

        record = self._record(scope.game_id, scope.level)
        current_frontier_version = int(record.frontier_version)

        # The frontier itself changed and remains valid evidence, but no better
        # action cost was discovered. Undo only optimization-progress side effects.
        if previous_state == v819.GameLearningState.SOLVED_STABLE:
            record.state = previous_state
        record.validations_since_improvement = previous_validations
        record.last_frontier_improvement_generation = previous_improvement_generation
        if previous_exhausted_version == previous_frontier_version:
            record.optimizer_exhausted_version = current_frontier_version
        else:
            record.optimizer_exhausted_version = previous_exhausted_version

        signals = self._signals.setdefault(
            str(scope.game_id), v819.GamePrioritySignals()
        )
        signals.competence_improvement = previous_competence
        signals.novelty = previous_novelty

    return True


def _reset_progress_depth_v846() -> None:
    _MAX_LEVEL_REACHED.clear()


def install_plateau_progress_v846() -> None:
    global _INSTALLED, _BASE_DEEPEST_LEVEL
    global _BASE_OBSERVE_FRONTIER_CANDIDATE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import diagnostics

    _BASE_DEEPEST_LEVEL = diagnostics._deepest_level
    _BASE_OBSERVE_FRONTIER_CANDIDATE = (
        v819.AdaptiveLearningCoordinator.observe_frontier_candidate
    )

    # _ProgressAdapter resolves _LeaseProgress dynamically. Recording the row when
    # the parent scheduler consumes it preserves the real actor-side episode depth
    # after active lease state is discarded. v8.23 remains the public row builder.
    v819._LeaseProgress = LeaseProgressV846
    diagnostics._deepest_level = _deepest_level_v846

    # Wrap, rather than replace internally, the already-composed v8.19 authority.
    v819.AdaptiveLearningCoordinator.observe_frontier_candidate = (
        _observe_frontier_candidate_v846
    )

    _INSTALLED = True
