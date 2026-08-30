from __future__ import annotations

"""v8.77 generic worker result delivery integrity.

``multiprocessing.Queue`` delivery and process-exit observation are not one atomic
operation.  A parent can observe all generic workers as exited immediately after a
queue timeout while the last result is still becoming readable.  v8.75 treated that
instant as authoritative and could therefore raise a false
``generic actor result missing`` error.

This layer preserves the mixed-run scheduling authority while adding two delivery
barriers:

* each child explicitly flushes its result queue feeder before it exits;
* after all children stop, the parent joins them and performs one bounded terminal
  result drain before deciding that an actor result is genuinely missing.
"""

import queue
import time


_INSTALLED = False
_BASE_GENERIC_PROCESS_WORKER = None
_TERMINAL_DRAIN_SECONDS = 1.0


def _generic_process_worker_v877(**kwargs) -> None:
    result_queue = kwargs.get("result_queue")
    try:
        return _BASE_GENERIC_PROCESS_WORKER(**kwargs)
    finally:
        if result_queue is not None:
            try:
                result_queue.close()
            except (AttributeError, ValueError, OSError):
                pass
            try:
                result_queue.join_thread()
            except (AttributeError, RuntimeError, ValueError, OSError):
                pass


def _record_generic_message_v877(message, by_actor: dict, errors: list[str]) -> None:
    if not (isinstance(message, tuple) and message):
        return
    if message[0] == "result":
        row = message[1]
        by_actor[int(row.actor_id)] = row
    elif message[0] == "error":
        errors.append(str(message[2]))


def _terminal_result_drain_v877(results, by_actor: dict, errors: list[str], expected: int) -> None:
    """Drain results that become visible immediately after worker process exit."""
    deadline = time.monotonic() + _TERMINAL_DRAIN_SECONDS
    while len(by_actor) < int(expected) and not errors:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        try:
            message = results.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        _record_generic_message_v877(message, by_actor, errors)


def _run_generic_jobs_v877(runtime, jobs, *, reporting_queue=None):
    from v8 import mixed_research_runtime_integrity_v875 as v875

    jobs = tuple(jobs)
    if not jobs:
        return ()
    if not v875._supports_generic_processes(runtime):
        return v875._BASE_GENERIC_JOBS(
            runtime, jobs, reporting_queue=reporting_queue
        )

    runtime.start()
    v875._emit_phase(runtime, "generic_sampling")
    runtime._v875_generic_active = True
    ctx = runtime._mp_ctx
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=v875._generic_process_worker_v875,
            kwargs={
                "job": job,
                "experience_ring_args": runtime._stage_rings[0].attachment_args(),
                "read_descriptors": runtime.shard_descriptors,
                "watermark": runtime._watermark,
                "stop_event": runtime._stop,
                "snapshot_freeze": runtime._snapshot_freeze,
                "result_queue": results,
                "reporting_queue": reporting_queue,
            },
            name=f"v8-generic-{int(job.actor_id):03d}-{job.game_id}",
            daemon=True,
        )
        for job in jobs
    ]
    by_actor: dict[int, object] = {}
    errors: list[str] = []
    try:
        for process in processes:
            process.start()
        while len(by_actor) < len(jobs):
            try:
                message = results.get(timeout=0.05)
            except queue.Empty:
                message = None
            _record_generic_message_v877(message, by_actor, errors)

            failed = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if errors or failed:
                detail = "; ".join(errors) or ", ".join(
                    f"{process.name}={process.exitcode}" for process in failed
                )
                raise RuntimeError(f"generic actor failed: {detail}")

            if all(not process.is_alive() for process in processes) and len(by_actor) < len(jobs):
                # Process liveness and Queue visibility are not atomic. Join the
                # exited workers and make a final bounded read before declaring a
                # successful actor's result missing.
                for process in processes:
                    process.join(timeout=0.5)
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    detail = ", ".join(
                        f"{process.name}={process.exitcode}" for process in failed
                    )
                    raise RuntimeError(f"generic actor failed: {detail}")
                _terminal_result_drain_v877(
                    results, by_actor, errors, len(jobs)
                )
                if errors:
                    raise RuntimeError(f"generic actor failed: {'; '.join(errors)}")
                if len(by_actor) < len(jobs):
                    missing = sorted(
                        int(job.actor_id)
                        for job in jobs
                        if int(job.actor_id) not in by_actor
                    )
                    raise RuntimeError(
                        f"generic actor result missing for ids: {missing}"
                    )

        for process in processes:
            process.join(timeout=2.0)
        return tuple(by_actor[key] for key in sorted(by_actor))
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2.0)
        raise
    finally:
        runtime._v875_generic_active = False
        if not bool(getattr(runtime, "_v875_arc_done", False)):
            v875._emit_phase(runtime, "arc_tail")
        try:
            results.cancel_join_thread()
        except (AttributeError, ValueError):
            pass
        try:
            results.close()
        except (AttributeError, ValueError):
            pass


def install_generic_result_flush_v877() -> None:
    global _INSTALLED, _BASE_GENERIC_PROCESS_WORKER
    if _INSTALLED:
        return

    from v8 import mixed_environment_v859 as mixed
    from v8 import mixed_research_runtime_integrity_v875 as v875

    _BASE_GENERIC_PROCESS_WORKER = v875._generic_process_worker_v875
    v875._generic_process_worker_v875 = _generic_process_worker_v877
    v875._run_generic_jobs_v875 = _run_generic_jobs_v877
    mixed._run_generic_jobs = v875._run_generic_jobs_v875
    _INSTALLED = True
