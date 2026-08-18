from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

import v8  # noqa: F401 - install chronological runtime stack
from v8 import adaptive_allocator_occupancy_v840 as occupancy
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import lease_dispatch_lifecycle_v843 as v843
from v8.model import CognitiveState, MemoryUid, ValidationState


class _RestoredReadView:
    def __init__(self, rows, *, delay: float = 0.0, forbid_refresh: bool = False) -> None:
        self.rows = dict(rows)
        self.delay = float(delay)
        self.forbid_refresh = bool(forbid_refresh)
        self.refreshes = 0
        self._node_by_uid = {}

    def _refresh_strategy_cache(self) -> None:
        self.refreshes += 1
        if self.forbid_refresh:
            raise AssertionError("unsolved dispatch must not inspect lifecycle graph")
        if self.delay > 0:
            time.sleep(self.delay)
        self._node_by_uid = dict(self.rows)


def _solved_coordinator() -> tuple[v819.AdaptiveLearningCoordinator, _RestoredReadView]:
    game = "restored"
    strategy_uid = MemoryUid(11, 101)
    outcome_uid = MemoryUid(22, 202)
    view = _RestoredReadView(
        {
            strategy_uid: SimpleNamespace(cognitive_state=int(CognitiveState.ACTIVE)),
            outcome_uid: SimpleNamespace(cognitive_state=int(CognitiveState.ACTIVE)),
        },
        delay=0.05,
    )
    coordinator = v819.AdaptiveLearningCoordinator()
    coordinator.register_games((game,))
    coordinator._game_won[game] = True
    coordinator._records[(game, 1)] = v819.GameLevelLearningRecord(
        state=v819.GameLearningState.SOLVED_OPTIMIZING,
        first_success_generation=1,
        last_success_generation=1,
        last_frontier_improvement_generation=1,
    )
    scope = v819.FrontierScope(game, 1, 7, outcome_uid.hi, outcome_uid.lo)
    coordinator.frontier.add(
        scope,
        v819.FrontierCandidate(
            strategy_uid=strategy_uid,
            trajectory_id="restored-trajectory",
            action_hash=123,
            cost=4,
            attempts=1,
            successes=1,
            validation_state=int(ValidationState.VALIDATED),
            source=v819.FrontierSource.SAMPLER,
            generation=1,
        ),
    )
    coordinator._v827_lifecycle_authority = True
    coordinator._v827_read_view = view
    return coordinator, view


class LeaseDispatchLifecycleV843Tests(unittest.TestCase):
    def test_genuinely_unsolved_choose_mode_never_refreshes_lifecycle_graph(self) -> None:
        coordinator = v819.AdaptiveLearningCoordinator()
        coordinator.register_games(("cold",))
        view = _RestoredReadView({}, forbid_refresh=True)
        coordinator._v827_lifecycle_authority = True
        coordinator._v827_read_view = view

        assigned = []

        def assign(worker_id: int) -> bool:
            self.assertEqual(
                coordinator.choose_mode("cold"),
                v819.SamplingMode.DISCOVERY,
            )
            self.assertEqual(
                coordinator.game_state("cold"),
                v819.GameLearningState.UNSOLVED,
            )
            assigned.append(worker_id)
            return True

        workers = set(range(1, 31))
        result = occupancy._refill_idle_workers(workers, assign)

        self.assertEqual(result, tuple(range(1, 31)))
        self.assertEqual(assigned, list(range(1, 31)))
        self.assertEqual(view.refreshes, 0)

    def test_restored_graph_initial_fill_never_rebuilds_lifecycle_index(self) -> None:
        coordinator, view = _solved_coordinator()
        assigned = []

        def assign(worker_id: int) -> bool:
            mode = coordinator.choose_mode("restored")
            self.assertEqual(
                coordinator.game_state("restored"),
                v819.GameLearningState.SOLVED_OPTIMIZING,
            )
            if mode == v819.SamplingMode.ALTERNATIVE:
                self.assertFalse(coordinator.alternative_exclusion("restored").is_zero)
            assigned.append(worker_id)
            return True

        workers = set(range(1, 31))
        started = time.monotonic()
        result = occupancy._refill_idle_workers(workers, assign)
        elapsed = time.monotonic() - started

        self.assertEqual(result, tuple(range(1, 31)))
        self.assertEqual(assigned, list(range(1, 31)))
        self.assertEqual(view.refreshes, 0)
        self.assertLess(elapsed, 0.25)

    def test_non_dispatch_refresh_publishes_index_for_later_refills(self) -> None:
        coordinator, view = _solved_coordinator()

        self.assertIsNotNone(v843._dispatch_lifecycle_index(view))
        self.assertEqual(view.refreshes, 1)

        workers = {1, 2}
        occupancy._refill_idle_workers(
            workers,
            lambda _worker_id: coordinator.game_state("restored")
            == v819.GameLearningState.SOLVED_OPTIMIZING,
        )

        self.assertEqual(workers, set())
        self.assertEqual(view.refreshes, 1)

    def test_allocation_telemetry_uses_cached_lifecycle_index(self) -> None:
        coordinator, view = _solved_coordinator()

        with v843._dispatch_lifecycle_cache():
            rows = coordinator.telemetry()

        self.assertEqual(len(rows), 1)
        self.assertEqual(view.refreshes, 0)

    def test_v843_is_final_refill_and_choose_mode_authority(self) -> None:
        self.assertIs(occupancy._refill_idle_workers, v843._refill_idle_workers_v843)
        self.assertIs(v819.AdaptiveLearningCoordinator.choose_mode, v843._choose_mode_v843)
        self.assertIs(perf._write_allocation_log_live, v843._write_allocation_log_v843)
        self.assertIs(perf._allocation_stdout_live, v843._allocation_stdout_v843)


if __name__ == "__main__":
    unittest.main()
