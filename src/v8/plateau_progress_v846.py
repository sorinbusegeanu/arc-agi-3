from __future__ import annotations

"""v8.46 progress fidelity and optimization-stagnation repair.

Two independent runtime repairs live here:

* adaptive lease aggregation preserves the deepest level actually reached during
  the current run instead of falling back to cumulative level-transition counts;
* Pareto-frontier churn only restarts solution optimization when the best action
  cost for that scope improves meaningfully.
"""

from dataclasses import dataclass, replace


_INSTALLED = False
_BASE_ADAPTIVE_PROGRESS_ROWS = None
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
    """Lease progress event that retains episode depth after the lease completes."""

    worker_id: int
    lease_id: int
    row: object

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        if name == "row":
            _record_progress_depth(value)
        return value


def _adaptive_progress_rows_v846(
    actor_module,
    jobs,
    completed_by_game,
    active_progress,
    active_leases,
):
    for row in active_progress.values():
        if row is not None:
            _record_progress_depth(row)

    rows = tuple(
        _BASE_ADAPTIVE_PROGRESS_ROWS(
            actor_module,
            jobs,
            completed_by_game,
            active_progress,
            active_leases,
        )
    )
    repaired = []
    for row in rows:
        game = str(getattr(row, "game_id", ""))
        retained = int(_MAX_LEVEL_REACHED.get(game, -1))
        current = int(getattr(row, "max_level_reached", -1) or -1)
        depth = max(current, retained)
        if depth >= 0 and hasattr(row, "max_level_reached"):
            try:
                row = replace(row, max_level_reached=depth)
            except (TypeError, ValueError):
                pass
        repaired.append(row)
    return tuple(repaired)


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
    """Keep Pareto evidence without treating same-cost churn as optimizer progress."""

    from v8 import adaptive_learning_allocation_v819 as v819

    with self._lock:
        self.register_games((scope.game_id,))
        prior_winner = self.frontier.winner(scope)
        record = self._record(scope.game_id, scope.level)
        previous_state = record.state
        previous_frontier_version = int(record.frontier_version)
        previous_exhausted_version = int(record.optimizer_exhausted_version)

        changed, version = self.frontier.add(scope, candidate)
        current_winner = self.frontier.winner(scope)

        if record.first_success_generation <= 0:
            record.first_success_generation = max(1, int(generation))
        record.last_success_generation = max(1, int(generation))
        if record.state == v819.GameLearningState.UNSOLVED:
            record.state = v819.GameLearningState.SOLVED_OPTIMIZING
        if str(terminal_state) == "WIN":
            self._game_won[scope.game_id] = True

        meaningful = bool(
            changed
            and _meaningful_cost_improvement(self, prior_winner, current_winner)
        )
        if changed:
            record.frontier_version = int(version)
            if meaningful:
                record.optimizer_exhausted_version = -1
                record.validations_since_improvement = 0
                record.last_frontier_improvement_generation = max(1, int(generation))
                if previous_state == v819.GameLearningState.SOLVED_STABLE:
                    record.state = v819.GameLearningState.SOLVED_OPTIMIZING
                signals = self._signals.setdefault(
                    scope.game_id,
                    v819.GamePrioritySignals(),
                )
                signals.competence_improvement = min(
                    1.5, signals.competence_improvement + 0.10
                )
                signals.novelty = min(1.35, signals.novelty + 0.05)
            elif previous_exhausted_version == previous_frontier_version:
                # Frontier bookkeeping advanced, but the optimizer had already
                # exhausted the prior best-cost frontier. Preserve that exhaustion
                # against the new evidence-only version instead of re-opening it.
                record.optimizer_exhausted_version = int(version)

            self._emit(
                f"frontier game={scope.game_id} level={scope.level} "
                f"source={candidate.source.value} cost={candidate.cost} "
                f"reliability={candidate.reliability:.2f} version={version}"
            )

        if previous_state != record.state:
            self._emit(
                f"learning state game={scope.game_id} level={scope.level} "
                f"{previous_state.value}->{record.state.value}"
            )
        return bool(changed)


def _reset_progress_depth_v846() -> None:
    _MAX_LEVEL_REACHED.clear()


def install_plateau_progress_v846() -> None:
    global _INSTALLED, _BASE_ADAPTIVE_PROGRESS_ROWS
    global _BASE_OBSERVE_FRONTIER_CANDIDATE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819

    _BASE_ADAPTIVE_PROGRESS_ROWS = v819._adaptive_progress_rows
    _BASE_OBSERVE_FRONTIER_CANDIDATE = (
        v819.AdaptiveLearningCoordinator.observe_frontier_candidate
    )

    # _ProgressAdapter resolves this module global at call time. Replacing the
    # event class therefore records every actor-published depth in the parent
    # scheduler before completed lease state is discarded.
    v819._LeaseProgress = LeaseProgressV846
    v819._adaptive_progress_rows = _adaptive_progress_rows_v846
    v819.AdaptiveLearningCoordinator.observe_frontier_candidate = (
        _observe_frontier_candidate_v846
    )

    _INSTALLED = True
