from __future__ import annotations

import argparse
import os
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path


_INSTALLED = False
_ACTOR_POOL_ENV = "ARC_AGI3_V8_ACTOR_POOL_SIZE"


@dataclass(slots=True)
class _BudgetLedger:
    """Track consumed credits separately from currently in-flight reservations."""

    total: int
    consumed: int = 0
    reservations: dict[int, int] = field(default_factory=dict)

    @property
    def reserved(self) -> int:
        return sum(max(0, int(value)) for value in self.reservations.values())

    @property
    def available(self) -> int:
        return max(0, int(self.total) - int(self.consumed) - int(self.reserved))

    def reserve(self, worker_id: int, steps: int) -> int:
        worker = int(worker_id)
        value = max(0, int(steps))
        if worker in self.reservations:
            raise RuntimeError(f"worker {worker} already has an in-flight reservation")
        if value <= 0 or value > self.available:
            raise RuntimeError(
                f"invalid reservation worker={worker} steps={value} available={self.available}"
            )
        self.reservations[worker] = value
        self._check()
        return value

    def complete(self, worker_id: int, actual_steps: int) -> int:
        worker = int(worker_id)
        if worker not in self.reservations:
            raise RuntimeError(f"worker {worker} completed without an in-flight reservation")
        reserved = int(self.reservations.pop(worker))
        actual = max(0, int(actual_steps))
        if actual > reserved:
            raise RuntimeError(
                f"worker {worker} consumed {actual} steps from a {reserved}-step lease"
            )
        self.consumed += actual
        self._check()
        return reserved - actual

    def _check(self) -> None:
        committed = int(self.consumed) + int(self.reserved)
        if committed < 0 or committed > int(self.total):
            raise RuntimeError(
                f"adaptive allocator budget invariant violated: "
                f"consumed={self.consumed} reserved={self.reserved} total={self.total}"
            )


def _requested_actor_pool_v840(values: list[str]) -> int:
    """Treat --actors as a process cap, never as a minimum lane count."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--actors", type=int, default=8)
    parsed, _unknown = parser.parse_known_args(values)
    requested = int(parsed.actors)
    if requested <= 0:
        raise ValueError("--actors must be positive")
    return requested


def _refill_idle_workers(idle_workers: set[int], assign) -> tuple[int, ...]:
    """Fill every idle process for which interaction budget is dispatchable."""

    assigned: list[int] = []
    for worker_id in sorted(tuple(idle_workers)):
        if not bool(assign(int(worker_id))):
            break
        idle_workers.discard(int(worker_id))
        assigned.append(int(worker_id))
    return tuple(assigned)


def _adaptive_run_actor_jobs_v840(
    runtime,
    jobs,
    *,
    timeout: float | None = None,
    progress_interval_seconds: float = 60.0,
    progress_callback=None,
    reporting_queue=None,
):
    """Adaptive scheduler with reusable in-flight credits and stable occupancy.

    A lease reserves credits only while it is executing. Completion always releases
    that reservation and records only the steps actually consumed. All idle worker
    slots are refilled after every scheduling opportunity. Unsolved lease duration
    remains episode-aligned through the v8.23/v8.26 helper so this repair does not
    reintroduce the former arbitrary 2048-step episode cutoff.
    """

    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf

    jobs = tuple(jobs)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None or not jobs:
        return perf._BASE_RUN_ACTOR_JOBS(
            runtime,
            jobs,
            timeout=timeout,
            progress_interval_seconds=progress_interval_seconds,
            progress_callback=progress_callback,
            reporting_queue=reporting_queue,
        )
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    games = tuple(dict.fromkeys(str(job.game_id) for job in jobs))
    coordinator.register_games(games)
    total_budget = sum(max(0, int(job.steps)) for job in jobs)
    if total_budget <= 0:
        return ()

    base_budget_by_game: dict[str, int] = {game: 0 for game in games}
    first_job_by_game: dict[str, object] = {}
    for job in jobs:
        game = str(job.game_id)
        base_budget_by_game[game] = base_budget_by_game.get(game, 0) + max(0, int(job.steps))
        first_job_by_game.setdefault(game, job)

    worker_count = perf._v823_requested_actor_pool(len(jobs))
    template = jobs[0]
    trajectory_root = str(Path(runtime.root) / "trajectory_optimizer")
    os.environ[perf._TRAJECTORY_ROOT_ENV] = trajectory_root

    peers = getattr(runtime, "peers", None)
    if peers is not None:
        peers.pause()
    runtime.start()
    ctx = runtime._mp_ctx
    if peers is not None:
        startup_timeout = 300.0 if timeout is None else max(0.01, min(float(timeout), 300.0))
        if not peers.wait_idle(startup_timeout):
            raise TimeoutError("v8 peers did not pause before adaptive actor startup")
        runtime.wait_quiescent(
            timeout=startup_timeout,
            resume_peers=False,
            settle_peers=False,
        )

    assignments = [ctx.Queue(maxsize=2) for _ in range(worker_count)]
    events = ctx.Queue(maxsize=max(64, worker_count * 16))
    ready = [ctx.Event() for _ in range(worker_count)]
    processes = [
        ctx.Process(
            target=perf._worker_until_win,
            kwargs={
                "worker_id": index + 1,
                "assignment_queue": assignments[index],
                "event_queue": events,
                "ready_event": ready[index],
                "experience_ring_args": runtime._stage_rings[0].attachment_args(),
                "read_descriptors": runtime.shard_descriptors,
                "watermark": runtime._watermark,
                "stop_event": runtime._stop,
                "actor_throttle": runtime._actor_throttle,
                "snapshot_freeze": runtime._snapshot_freeze,
                "trajectory_root": trajectory_root,
            },
            name=f"v8-adaptive-actor-{index + 1:03d}",
            daemon=True,
        )
        for index in range(worker_count)
    ]

    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    next_progress = started + float(progress_interval_seconds)
    next_log = started + float(v819._ALLOCATION_LOG_SECONDS)
    next_stdout = started + float(v819._ALLOCATION_STDOUT_SECONDS)
    budget = _BudgetLedger(int(total_budget))
    lease_id = 0
    active_leases: dict[int, object] = {}
    active_progress: dict[int, object] = {}
    idle_workers: set[int] = set(range(1, worker_count + 1))
    completed_by_game: dict[str, dict[str, int]] = {}
    provisional_wins: dict[str, float] = {}
    leases_by_game: dict[str, int] = {}
    initial_games = list(games)
    no_progress_retries = 0

    def bucket(game: str) -> dict[str, int]:
        return completed_by_game.setdefault(
            game,
            {
                "steps": 0,
                "wins": 0,
                "failures": 0,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 0,
                "best_win_steps": 0,
                "last_win_steps": 0,
                "resets": 0,
            },
        )

    def choose_game() -> str:
        now = time.monotonic()
        for game, expiry in tuple(provisional_wins.items()):
            if coordinator.game_state(game) != v819.GameLearningState.UNSOLVED or now >= expiry:
                provisional_wins.pop(game, None)
        candidates = tuple(game for game in games if game not in provisional_wins)
        if not candidates:
            candidates = games
        active_reserved: dict[str, int] = {}
        active_counts: dict[str, int] = {}
        for lease in active_leases.values():
            game = str(lease.game_id)
            active_reserved[game] = active_reserved.get(game, 0) + int(lease.steps)
            active_counts[game] = active_counts.get(game, 0) + 1
        with coordinator._lock:
            return min(
                candidates,
                key=lambda game: (
                    (
                        float(coordinator._run[game].sample_steps)
                        + float(active_reserved.get(game, 0))
                    )
                    / max(1e-9, float(coordinator.sampling_weight(game))),
                    int(coordinator._run[game].leases) + int(active_counts.get(game, 0)),
                    game,
                ),
            )

    def assign(worker_id: int) -> bool:
        nonlocal lease_id
        available = int(budget.available)
        if available <= 0:
            return False
        initial_probe = bool(initial_games)
        if initial_probe:
            game = initial_games.pop(0)
        else:
            game = choose_game()
        mode = coordinator.choose_mode(game)
        if coordinator.game_state(game) == v819.GameLearningState.UNSOLVED:
            steps = perf._v823_initial_unsolved_lease_steps(
                available=int(available),
                base_steps=max(1, int(base_budget_by_game.get(game, 1))),
                initial_probe=initial_probe,
                worker_count=worker_count,
                game_count=len(games),
            )
        else:
            steps = coordinator.recommended_lease_steps(game, available)
        steps = min(int(available), max(1, int(steps)))
        lease_id += 1
        excluded = (
            coordinator.alternative_exclusion(game)
            if mode == v819.SamplingMode.ALTERNATIVE
            else v819.MemoryUid.zero()
        )
        base_job = first_job_by_game.get(game, template)
        game_lease_index = int(leases_by_game.get(game, 0))
        leases_by_game[game] = game_lease_index + 1
        lease = v819.ActorLease(
            lease_id,
            int(worker_id),
            game,
            int(steps),
            int(getattr(base_job, "seed", 0)) + game_lease_index * 7919,
            getattr(base_job, "env_root", None),
            float(getattr(base_job, "epsilon", 0.10)),
            int(getattr(base_job, "graph_check_steps", 1000)),
            mode,
            excluded,
        )
        budget.reserve(int(worker_id), int(steps))
        active_leases[int(worker_id)] = lease
        active_progress[int(worker_id)] = None
        assignments[int(worker_id) - 1].put(lease)
        return True

    def refill() -> tuple[int, ...]:
        return _refill_idle_workers(idle_workers, assign)

    try:
        for process in processes:
            process.start()
        while not all(event.is_set() for event in ready):
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                detail = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                raise RuntimeError(f"adaptive actor failed during startup: {detail}")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("adaptive actor startup timed out")
            time.sleep(0.01)
        if peers is not None:
            peers.resume()
        refill()

        while active_leases or budget.available > 0:
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                event = None

            if isinstance(event, v819._LeaseProgress):
                row = event.row
                if isinstance(row, actor_module.ActorLearningBatch):
                    runtime.record_actor_results((row,))
                elif isinstance(row, actor_module.ActorProgress):
                    current = active_leases.get(int(event.worker_id))
                    if current is not None and int(current.lease_id) == int(event.lease_id):
                        active_progress[int(event.worker_id)] = row
            elif isinstance(event, v819._LeaseResult):
                worker_id = int(event.worker_id)
                lease = event.lease
                current = active_leases.get(worker_id)
                if current is None or int(current.lease_id) != int(lease.lease_id):
                    continue
                result = event.result
                active_leases.pop(worker_id, None)
                active_progress.pop(worker_id, None)
                actual = max(0, int(getattr(result, "steps", 0)))
                budget.complete(worker_id, actual)
                idle_workers.add(worker_id)
                coordinator.record_lease(lease.game_id, lease.mode, actual)
                values = bucket(str(lease.game_id))
                prior_steps = int(values["steps"])
                values["steps"] += actual
                values["wins"] += int(getattr(result, "wins", 0))
                values["failures"] += int(getattr(result, "failures", 0))
                values["levels_completed"] += int(getattr(result, "levels_completed", 0))
                values["replans"] += int(getattr(result, "replans", 0))
                values["planned_steps"] += int(getattr(result, "planned_steps", 0))
                values["resets"] += int(getattr(result, "resets", 0))
                result_best = int(getattr(result, "best_win_steps", 0) or 0)
                if result_best > 0:
                    prior_best = int(values.get("best_win_steps", 0) or 0)
                    values["best_win_steps"] = (
                        result_best if prior_best <= 0 else min(prior_best, result_best)
                    )
                result_last = int(getattr(result, "last_win_steps", 0) or 0)
                if result_last > 0:
                    values["last_win_steps"] = result_last
                pending = getattr(result, "pending_learning", None)
                if pending is not None:
                    runtime.record_actor_results((pending,))
                if int(getattr(result, "wins", 0)) > 0:
                    provisional_wins[str(lease.game_id)] = (
                        time.monotonic() + perf._PROVISIONAL_WIN_SECONDS
                    )
                    if values["first_win_step"] <= 0:
                        values["first_win_step"] = prior_steps + max(1, actual)
                if actual <= 0:
                    no_progress_retries += 1
                else:
                    no_progress_retries = 0
                if no_progress_retries > max(32, worker_count * 4):
                    raise RuntimeError("adaptive allocator could not consume interaction credits")

            # Refilling is global rather than tied to the worker that produced the
            # most recent completion. This keeps all process slots occupied until
            # the true global interaction budget is nearly exhausted.
            refill()

            retry = getattr(runtime, "_v819_retry_deferred", None)
            if retry is not None:
                retry()
            service = getattr(runtime, "_v814_trajectory_optimizer", None)
            if v819._optimizer_idle(service):
                coordinator.mark_optimizer_idle(generation=int(runtime.generation))
            else:
                coordinator.stabilize(generation=int(runtime.generation))

            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                detail = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                raise RuntimeError(f"adaptive actor failed: {detail}")

            now = time.monotonic()
            if now >= next_progress:
                rows = v819._adaptive_progress_rows(
                    actor_module,
                    jobs,
                    completed_by_game,
                    active_progress,
                    active_leases,
                )
                if reporting_queue is not None:
                    for row in rows:
                        try:
                            reporting_queue.put_nowait(row)
                        except queue.Full:
                            break
                if progress_callback is not None:
                    progress_callback(rows)
                while next_progress <= now:
                    next_progress += float(progress_interval_seconds)
            if now >= next_log:
                perf._write_allocation_log_live(
                    runtime,
                    coordinator,
                    completed_by_game,
                    active_progress,
                    active_leases,
                )
                next_log = now + float(v819._ALLOCATION_LOG_SECONDS)
            if now >= next_stdout:
                perf._allocation_stdout_live(
                    coordinator,
                    completed_by_game,
                    active_progress,
                    active_leases,
                )
                next_stdout = now + float(v819._ALLOCATION_STDOUT_SECONDS)
            if deadline is not None and now >= deadline:
                raise TimeoutError("adaptive actor jobs timed out")

        final_rows = v819._adaptive_progress_rows(
            actor_module,
            jobs,
            completed_by_game,
            {},
            {},
        )
        if reporting_queue is not None:
            for row in final_rows:
                try:
                    reporting_queue.put_nowait(row)
                except queue.Full:
                    break
        if progress_callback is not None:
            progress_callback(final_rows)
        perf._write_allocation_log_live(runtime, coordinator, completed_by_game, {}, {})

        first_job: dict[str, object] = {}
        for job in jobs:
            first_job.setdefault(str(job.game_id), job)
        results = []
        for job in jobs:
            game = str(job.game_id)
            values = completed_by_game.get(game, {})
            if first_job[game] is not job:
                values = {}
            results.append(
                actor_module.ActorResult(
                    int(job.actor_id),
                    game,
                    int(values.get("steps", 0)),
                    int(values.get("wins", 0)),
                    int(values.get("failures", 0)),
                    int(values.get("levels_completed", 0)),
                    int(values.get("resets", 0)),
                    int(values.get("replans", 0)),
                    int(values.get("planned_steps", 0)),
                    (),
                    (),
                    (),
                    None,
                )
            )
        return tuple(results)
    except BaseException:
        for q in assignments:
            try:
                q.put_nowait(None)
            except BaseException:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2.0)
        raise
    finally:
        for q in assignments:
            try:
                q.put_nowait(None)
            except BaseException:
                pass
        for process in processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for q in assignments:
            q.cancel_join_thread()
            q.close()
        events.cancel_join_thread()
        events.close()


def install_adaptive_allocator_occupancy_v840() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import cli_v819

    # The progressive/v8.39 outer wrappers resolve this symbol dynamically, so
    # replacing the scheduler here preserves their batching and maintenance work.
    perf._adaptive_run_actor_jobs_perf = _adaptive_run_actor_jobs_v840

    # v8.28 made --actors a minimum and expanded it to game count. v8.40 restores
    # the CLI meaning to an explicit/default process cap while retaining all job
    # descriptors for adaptive scheduling and reporting.
    cli_v819._requested_actor_pool = _requested_actor_pool_v840
    _INSTALLED = True
