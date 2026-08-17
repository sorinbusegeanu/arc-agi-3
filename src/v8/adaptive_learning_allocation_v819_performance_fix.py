from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path


_INSTALLED = False
_BASE_RUN_ACTOR_JOBS = None
_BASE_VIEW_INIT = None
_BASE_PLAN_CANDIDATES = None
_BASE_RUNTIME_START = None
_TRAJECTORY_ROOT_ENV = "ARC_AGI3_V8_TRAJECTORY_ROOT"
_PROVISIONAL_WIN_SECONDS = 60.0


class _LeaseStopProxy:
    def __init__(self, global_stop, local_stop: threading.Event) -> None:
        self._global_stop = global_stop
        self._local_stop = local_stop

    def is_set(self) -> bool:
        return bool(self._local_stop.is_set() or self._global_stop.is_set())

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        if timeout is None:
            while not self.is_set():
                time.sleep(0.01)
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self.is_set():
                return True
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return self.is_set()


def _worker_until_win(
    *,
    worker_id: int,
    assignment_queue,
    event_queue,
    ready_event,
    experience_ring_args,
    read_descriptors,
    watermark,
    stop_event,
    actor_throttle,
    snapshot_freeze,
    trajectory_root: str,
) -> None:
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_inspection_v819 as inspection

    ready_event.set()
    while not stop_event.is_set():
        try:
            lease = assignment_queue.get(timeout=0.10)
        except queue.Empty:
            continue
        if lease is None:
            return
        if not isinstance(lease, v819.ActorLease):
            continue

        prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        prior_excluded = os.environ.get(v819._ALTERNATIVE_EXCLUDE_ENV)
        prior_root = os.environ.get(_TRAJECTORY_ROOT_ENV)
        os.environ[v819._SAMPLING_MODE_ENV] = lease.mode.value
        os.environ[v819._ALTERNATIVE_EXCLUDE_ENV] = v819._uid_env(lease.excluded_strategy_uid)
        os.environ[_TRAJECTORY_ROOT_ENV] = str(trajectory_root)

        epsilon = float(lease.epsilon)
        if lease.mode == v819.SamplingMode.VERIFY:
            epsilon = 0.0
        elif lease.mode == v819.SamplingMode.ALTERNATIVE:
            epsilon = max(0.15, epsilon)
        elif lease.mode == v819.SamplingMode.TRANSFER:
            epsilon = max(0.20, epsilon)

        job = actor_module.ActorJob(
            actor_id=int(worker_id),
            game_id=str(lease.game_id),
            steps=int(lease.steps),
            seed=int(lease.seed),
            env_root=lease.env_root,
            epsilon=epsilon,
            graph_check_steps=int(lease.graph_check_steps),
        )

        local_stop = threading.Event()
        stop_proxy = _LeaseStopProxy(stop_event, local_stop)
        base_step = ArcGridEnvironment.step

        def step_until_win(env, action):
            result = base_step(env, action)
            if (
                str(getattr(env, "last_outcome_state", "")) == "WIN"
                and not bool(getattr(env, "last_step_was_reset_boundary", False))
            ):
                local_stop.set()
            return result

        ArcGridEnvironment.step = step_until_win
        inspection._reset_observed_capture()
        try:
            actor_module.actor_worker(
                job=job,
                experience_ring_args=experience_ring_args,
                read_descriptors=read_descriptors,
                watermark=watermark,
                stop_event=stop_proxy,
                result_queue=v819._ResultAdapter(event_queue, worker_id, lease),
                progress_queue=v819._ProgressAdapter(event_queue, worker_id, lease.lease_id),
                reporting_queue=None,
                actor_throttle=actor_throttle,
                snapshot_freeze=snapshot_freeze,
                startup_ready=None,
                startup_gate=None,
            )
        finally:
            ArcGridEnvironment.step = base_step
            inspection._reset_observed_capture()
            if prior_mode is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior_mode
            if prior_excluded is None:
                os.environ.pop(v819._ALTERNATIVE_EXCLUDE_ENV, None)
            else:
                os.environ[v819._ALTERNATIVE_EXCLUDE_ENV] = prior_excluded
            if prior_root is None:
                os.environ.pop(_TRAJECTORY_ROOT_ENV, None)
            else:
                os.environ[_TRAJECTORY_ROOT_ENV] = prior_root


def _live_steps_by_game(completed_by_game, active_progress, active_leases) -> dict[str, int]:
    totals = {
        str(game): max(0, int(values.get("steps", 0)))
        for game, values in completed_by_game.items()
    }
    for worker_id, progress in active_progress.items():
        lease = active_leases.get(worker_id)
        if lease is None or progress is None:
            continue
        game = str(lease.game_id)
        totals[game] = totals.get(game, 0) + max(0, int(getattr(progress, "steps", 0)))
    return totals


def _allocation_stdout_live(coordinator, completed_by_game, active_progress, active_leases) -> None:
    from v8 import adaptive_learning_allocation_v819 as v819

    live = _live_steps_by_game(completed_by_game, active_progress, active_leases)
    total = sum(live.values())
    states = {state.value: 0 for state in v819.GameLearningState}
    for game in sorted(coordinator._games):
        state = coordinator.game_state(game).value
        states[state] = states.get(state, 0) + 1
    leaders = sorted(live.items(), key=lambda item: (-item[1], item[0]))[:5]
    shares = ",".join(
        f"{game}:{(100.0 * steps / max(1, total)):.0f}%" for game, steps in leaders
    )
    print(
        f'[{time.strftime("%H:%M")}] sampling allocation '
        f"unsolved={states.get(v819.GameLearningState.UNSOLVED.value, 0)} "
        f"optimizing={states.get(v819.GameLearningState.SOLVED_OPTIMIZING.value, 0)} "
        f"stable={states.get(v819.GameLearningState.SOLVED_STABLE.value, 0)} "
        f"steps={total} top={shares}",
        flush=True,
    )


def _write_allocation_log_live(runtime, coordinator, completed_by_game, active_progress, active_leases) -> None:
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    rows = [asdict(row) for row in coordinator.telemetry(optimizer_service=service)]
    live = _live_steps_by_game(completed_by_game, active_progress, active_leases)
    total = max(1, sum(live.values()))
    for row in rows:
        game = str(row.get("game_id", ""))
        steps = int(live.get(game, 0))
        row["sample_steps"] = steps
        row["sample_share"] = float(steps) / float(total)
    payload = {
        "time": time.time(),
        "generation": int(getattr(runtime, "generation", 0)),
        "watermark": int(getattr(runtime, "watermark", 0)),
        "sample_steps": sum(live.values()),
        "games": rows,
    }
    target = Path(runtime.root) / "sampling_allocation.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _adaptive_run_actor_jobs_perf(
    runtime,
    jobs,
    *,
    timeout: float | None = None,
    progress_interval_seconds: float = 60.0,
    progress_callback=None,
    reporting_queue=None,
):
    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819

    jobs = tuple(jobs)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None or not jobs:
        return _BASE_RUN_ACTOR_JOBS(
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

    worker_count = max(1, len(jobs))
    template = jobs[0]
    trajectory_root = str(Path(runtime.root) / "trajectory_optimizer")
    os.environ[_TRAJECTORY_ROOT_ENV] = trajectory_root

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
            target=_worker_until_win,
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
    reserved = 0
    consumed = 0
    lease_id = 0
    active_leases: dict[int, object] = {}
    active_progress: dict[int, object] = {}
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
        nonlocal reserved, lease_id
        available = total_budget - reserved
        if available <= 0:
            return False
        if initial_games:
            game = initial_games.pop(0)
        else:
            game = choose_game()
        mode = coordinator.choose_mode(game)
        if coordinator.game_state(game) == v819.GameLearningState.UNSOLVED:
            steps = min(int(available), max(1, int(base_budget_by_game.get(game, 1))))
        else:
            steps = coordinator.recommended_lease_steps(game, available)
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
            worker_id,
            game,
            int(steps),
            int(getattr(base_job, "seed", 0)) + game_lease_index * 7919,
            getattr(base_job, "env_root", None),
            float(getattr(base_job, "epsilon", 0.10)),
            int(getattr(base_job, "graph_check_steps", 1000)),
            mode,
            excluded,
        )
        reserved += int(steps)
        active_leases[worker_id] = lease
        active_progress[worker_id] = None
        assignments[worker_id - 1].put(lease)
        return True

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
        for worker_id in range(1, worker_count + 1):
            if not assign(worker_id):
                break

        while consumed < total_budget or active_leases:
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
                result = event.result
                active_leases.pop(worker_id, None)
                active_progress.pop(worker_id, None)
                actual = max(0, int(getattr(result, "steps", 0)))
                shortfall = max(0, int(lease.steps) - actual)
                if shortfall:
                    reserved -= shortfall
                consumed += actual
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
                pending = getattr(result, "pending_learning", None)
                if pending is not None:
                    runtime.record_actor_results((pending,))
                if int(getattr(result, "wins", 0)) > 0:
                    provisional_wins[str(lease.game_id)] = time.monotonic() + _PROVISIONAL_WIN_SECONDS
                    if values["first_win_step"] <= 0:
                        values["first_win_step"] = prior_steps + max(1, actual)
                if actual <= 0:
                    no_progress_retries += 1
                else:
                    no_progress_retries = 0
                if no_progress_retries > max(32, worker_count * 4):
                    raise RuntimeError("adaptive allocator could not consume interaction credits")
                assign(worker_id)

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
                _write_allocation_log_live(
                    runtime,
                    coordinator,
                    completed_by_game,
                    active_progress,
                    active_leases,
                )
                next_log = now + float(v819._ALLOCATION_LOG_SECONDS)
            if now >= next_stdout:
                _allocation_stdout_live(
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
        _write_allocation_log_live(runtime, coordinator, completed_by_game, {}, {})

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


def _view_init_perf(self, *args, **kwargs) -> None:
    _BASE_VIEW_INIT(self, *args, **kwargs)
    from v8 import adaptive_learning_allocation_v819 as v819

    self._v819_cached_sampling_mode = str(
        os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value)
    )


def _plan_candidates_perf(self, context_signature, action_ids, **kwargs):
    from v8 import adaptive_learning_allocation_v819 as v819

    if str(getattr(self, "_v819_cached_sampling_mode", "DISCOVERY")) == v819.SamplingMode.DISCOVERY.value:
        return v819._BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)
    return _BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)


def _runtime_start_perf(self) -> None:
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        os.environ[_TRAJECTORY_ROOT_ENV] = str(service.root)
    _BASE_RUNTIME_START(self)


def install_adaptive_learning_allocation_v819_performance_fix() -> None:
    global _INSTALLED
    global _BASE_RUN_ACTOR_JOBS, _BASE_VIEW_INIT, _BASE_PLAN_CANDIDATES, _BASE_RUNTIME_START
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8.publication import LiveReadView
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    _BASE_VIEW_INIT = LiveReadView.__init__
    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    _BASE_RUNTIME_START = V82ContinuousMemoryRuntime.start

    actor_module.run_actor_jobs = _adaptive_run_actor_jobs_perf
    LiveReadView.__init__ = _view_init_perf
    LiveReadView.plan_candidates = _plan_candidates_perf
    V82ContinuousMemoryRuntime.start = _runtime_start_perf
    _INSTALLED = True
