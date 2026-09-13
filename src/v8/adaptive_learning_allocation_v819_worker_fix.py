from __future__ import annotations

import os
import queue
import threading


_INSTALLED = False
_TRAJECTORY_ROOT_ENV = "ARC_AGI3_V8_TRAJECTORY_ROOT"


class _LeaseStopProxy:
    def __init__(self, global_stop, local_stop: threading.Event) -> None:
        self._global_stop = global_stop
        self._local_stop = local_stop

    def is_set(self) -> bool:
        return bool(self._global_stop.is_set() or self._local_stop.is_set())

    def wait(self, timeout=None) -> bool:
        if self.is_set():
            return True
        if timeout is None:
            return bool(self._global_stop.wait())
        if self._global_stop.wait(timeout):
            return True
        return self._local_stop.is_set()


def _worker_until_completed_win(
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
    record_cut_path: str | None = None,
    verified_success_root: str | None = None,
) -> None:
    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_inspection_v819 as inspection

    if verified_success_root:
        from v8.verified_success_metrics_v866 import SUCCESS_ROOT_ENV

        # A persistent worker may be forked by a forkserver created before the
        # run scope existed.  Carry the authoritative scope as process input
        # instead of relying on the forkserver's stale environment snapshot.
        os.environ[SUCCESS_ROOT_ENV] = str(verified_success_root)

    # The CLI normally prepares this cut once while every canonical writer is
    # stopped. Keep the local-build fallback for direct/library callers.
    shared_path = record_cut_path or os.environ.get(actor_module.ACTOR_RECORD_CUT_ENV)
    expected_versions = None
    if shared_path:
        record_cuts, expected_versions = actor_module.load_actor_record_cut_file(
            shared_path
        )
    else:
        record_cuts = {}
    warm_view = actor_module.open_actor_read_view(
        read_descriptors,
        refresh_interval_seconds=None,
        record_cuts=record_cuts,
    )
    if expected_versions is not None:
        actor_module._validate_actor_record_cut(warm_view, expected_versions)
    warm_compact_cut = getattr(warm_view, "_warm_compact_cut", None)
    if callable(warm_compact_cut):
        warm_compact_cut()
    warm_view.close()
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
        base_reset_after_terminal = actor_module._reset_after_terminal_game

        def reset_after_terminal(env, wait_seconds: float) -> None:
            terminal_state = str(getattr(env, "last_outcome_state", ""))
            base_reset_after_terminal(env, wait_seconds)
            # actor_worker has already written the winning ExperienceEvent,
            # updated terminal counters, captured the trajectory and published
            # pending learning before this reset helper is called.
            if terminal_state == "WIN":
                local_stop.set()

        actor_module._reset_after_terminal_game = reset_after_terminal
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
                record_cuts=record_cuts,
            )
        finally:
            actor_module._reset_after_terminal_game = base_reset_after_terminal
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


def install_adaptive_learning_allocation_v819_worker_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8.adaptive_learning_allocation_v819_solve_fix import (
        install_adaptive_learning_allocation_v819_solve_fix,
    )

    perf._worker_until_win = _worker_until_completed_win
    install_adaptive_learning_allocation_v819_solve_fix()
    _INSTALLED = True
