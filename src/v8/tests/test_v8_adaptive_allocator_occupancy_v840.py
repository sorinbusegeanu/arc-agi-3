from __future__ import annotations

import os
import unittest

import v8  # noqa: F401 - install chronological runtime stack
from v8 import adaptive_allocator_breadth_v840 as breadth
from v8 import adaptive_allocator_occupancy_v840 as v840
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import cli_v819
from v8 import sampling_control_repair_v823 as v823


class AdaptiveAllocatorOccupancyV840Tests(unittest.TestCase):
    def test_normal_completion_releases_full_reservation(self) -> None:
        ledger = v840._BudgetLedger(100)
        ledger.reserve(1, 10)
        self.assertEqual((ledger.consumed, ledger.reserved, ledger.available), (0, 10, 90))

        released = ledger.complete(1, 10)

        self.assertEqual(released, 0)
        self.assertEqual((ledger.consumed, ledger.reserved, ledger.available), (10, 0, 90))

    def test_short_completion_returns_unused_credits(self) -> None:
        ledger = v840._BudgetLedger(100)
        ledger.reserve(1, 10)

        released = ledger.complete(1, 3)

        self.assertEqual(released, 7)
        self.assertEqual((ledger.consumed, ledger.reserved, ledger.available), (3, 0, 97))

    def test_refill_fills_every_idle_slot_with_dispatchable_budget(self) -> None:
        ledger = v840._BudgetLedger(30)
        idle = {1, 2, 3}

        def assign(worker_id: int) -> bool:
            if ledger.available < 10:
                return False
            ledger.reserve(worker_id, 10)
            return True

        assigned = v840._refill_idle_workers(idle, assign)

        self.assertEqual(assigned, (1, 2, 3))
        self.assertEqual(idle, set())
        self.assertEqual(ledger.reserved, 30)

    def test_occupancy_stays_full_while_uncommitted_budget_exists(self) -> None:
        ledger = v840._BudgetLedger(100)
        idle = set(range(1, 9))
        active: set[int] = set()

        def assign(worker_id: int) -> bool:
            if ledger.available < 10:
                return False
            ledger.reserve(worker_id, 10)
            active.add(worker_id)
            return True

        v840._refill_idle_workers(idle, assign)
        self.assertEqual(len(active), 8)

        for worker_id in (1, 2):
            active.remove(worker_id)
            ledger.complete(worker_id, 10)
            idle.add(worker_id)
            v840._refill_idle_workers(idle, assign)
            self.assertEqual(len(active), 8)

        self.assertEqual(ledger.consumed, 20)
        self.assertEqual(ledger.reserved, 80)
        self.assertEqual(ledger.available, 0)

    def test_actor_option_is_a_cap_and_all_job_descriptors_survive(self) -> None:
        self.assertIs(cli_v819._requested_actor_pool, v840._requested_actor_pool_v840)
        self.assertEqual(cli_v819._requested_actor_pool(["continuous-run", "--games", "learning"]), 8)
        self.assertEqual(
            cli_v819._requested_actor_pool(
                ["continuous-run", "--games", "learning", "--actors", "6"]
            ),
            6,
        )
        batch = cli_v819._ActorJobBatch(tuple(range(36)), 8)
        self.assertEqual(len(batch), 8)
        self.assertEqual(tuple(batch), tuple(range(36)))

    def test_pool_resolution_uses_explicit_cap_even_after_tuple_conversion(self) -> None:
        prior = os.environ.get(v823._ACTOR_POOL_ENV)
        try:
            os.environ[v823._ACTOR_POOL_ENV] = "6"
            jobs = tuple(range(10))
            self.assertEqual(v823.requested_actor_pool(len(jobs)), 6)
        finally:
            if prior is None:
                os.environ.pop(v823._ACTOR_POOL_ENV, None)
            else:
                os.environ[v823._ACTOR_POOL_ENV] = prior

    def test_first_breadth_pass_does_not_precommit_full_game_budgets(self) -> None:
        self.assertIs(
            perf._v823_initial_unsolved_lease_steps,
            breadth._initial_breadth_lease_steps_v840,
        )
        prior = os.environ.get(breadth._ALLOCATION_LEASE_ENV)
        try:
            os.environ.pop(breadth._ALLOCATION_LEASE_ENV, None)
            first = perf._v823_initial_unsolved_lease_steps(
                available=100_000,
                base_steps=10_000,
                initial_probe=True,
                worker_count=8,
                game_count=10,
            )
            later = perf._v823_initial_unsolved_lease_steps(
                available=100_000,
                base_steps=10_000,
                initial_probe=False,
                worker_count=8,
                game_count=10,
            )
        finally:
            if prior is None:
                os.environ.pop(breadth._ALLOCATION_LEASE_ENV, None)
            else:
                os.environ[breadth._ALLOCATION_LEASE_ENV] = prior
        self.assertEqual(first, 4096)
        self.assertEqual(later, 10_000)
        self.assertEqual(first * 8, 32_768)

    def test_initial_breadth_quantum_is_configurable(self) -> None:
        prior = os.environ.get(breadth._ALLOCATION_LEASE_ENV)
        try:
            os.environ[breadth._ALLOCATION_LEASE_ENV] = "2500"
            self.assertEqual(
                perf._v823_initial_unsolved_lease_steps(
                    available=10_000,
                    base_steps=10_000,
                    initial_probe=True,
                    worker_count=6,
                    game_count=10,
                ),
                2500,
            )
            self.assertEqual(
                perf._v823_initial_unsolved_lease_steps(
                    available=10_000,
                    base_steps=10_000,
                    initial_probe=False,
                    worker_count=6,
                    game_count=10,
                ),
                10_000,
            )
        finally:
            if prior is None:
                os.environ.pop(breadth._ALLOCATION_LEASE_ENV, None)
            else:
                os.environ[breadth._ALLOCATION_LEASE_ENV] = prior

    def test_final_scheduler_is_v840(self) -> None:
        self.assertIs(perf._adaptive_run_actor_jobs_perf, v840._adaptive_run_actor_jobs_v840)

    def test_budget_cannot_overcommit_or_double_complete(self) -> None:
        ledger = v840._BudgetLedger(10)
        ledger.reserve(1, 10)
        with self.assertRaises(RuntimeError):
            ledger.reserve(2, 1)
        ledger.complete(1, 4)
        with self.assertRaises(RuntimeError):
            ledger.complete(1, 1)
        self.assertEqual(ledger.consumed + ledger.reserved + ledger.available, 10)


if __name__ == "__main__":
    unittest.main()
