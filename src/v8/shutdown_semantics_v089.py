from __future__ import annotations

import json
import time


_INSTALLED = False


def _mark_sampling_started(runtime) -> None:
    runtime._sampling_complete = False


def _mark_sampling_complete(runtime) -> None:
    """Stop autonomous peer scheduling as soon as actor sampling has ended."""
    runtime._sampling_complete = True
    if runtime.peers is not None:
        runtime.peers.pause()


def _remaining(deadline: float) -> float:
    return max(0.0, float(deadline) - time.monotonic())


def _wait_quiescent_v089(
    self,
    *,
    timeout: float = 60.0,
    stable_checks: int = 5,
    resume_peers: bool = True,
    settle_peers: bool = True,
) -> None:
    """Drain queues without requiring an unbounded semantic fixed point at shutdown.

    During normal mid-run maintenance, explicit peer settling remains available. Once
    actor sampling is complete, no new peer cycle is scheduled: an already-running
    cycle is allowed to finish, then only stage/shard mutations are drained.

    Convergence for explicit semantic settling is based on accepted graph generation
    changes, not the number of attempted peer submissions.
    """
    deadline = time.monotonic() + float(timeout)
    stable = 0
    final_drain = bool(getattr(self, "_sampling_complete", False))
    if final_drain:
        settle_peers = False
        resume_peers = False

    if self.peers is not None:
        self.peers.pause()
        if not self.peers.wait_idle(_remaining(deadline)):
            raise TimeoutError("v8 peers did not finish the active cycle before drain")

    observed_generation = int(self.generation)
    try:
        while time.monotonic() < deadline:
            self.raise_worker_errors()
            if not self._is_quiescent():
                stable = 0
                observed_generation = int(self.generation)
                time.sleep(0.01)
                continue

            current_generation = int(self.generation)
            if current_generation != observed_generation:
                observed_generation = current_generation
                stable = 0

            if self.peers is not None and settle_peers:
                before_generation = int(self.generation)
                self.peers.run_once()
                if not self._is_quiescent():
                    stable = 0
                    time.sleep(0.01)
                    continue
                after_generation = int(self.generation)
                if after_generation != before_generation:
                    observed_generation = after_generation
                    stable = 0
                    time.sleep(0.01)
                    continue

            stable += 1
            if stable >= max(1, int(stable_checks)):
                return
            time.sleep(0.01)

        mode = "canonical shutdown drain" if final_drain else "quiescence"
        raise TimeoutError(
            f"v8 did not reach {mode}; " + json.dumps(self.metrics(), sort_keys=True)
        )
    finally:
        if (
            self.peers is not None
            and resume_peers
            and not final_drain
            and not self._snapshot_freeze.is_set()
        ):
            self.peers.resume()


def install_shutdown_semantics_v089() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import runtime as runtime_module

    base_run_actor_jobs = actor_module.run_actor_jobs

    def run_actor_jobs(runtime, jobs, *args, **kwargs):
        _mark_sampling_started(runtime)
        results = base_run_actor_jobs(runtime, jobs, *args, **kwargs)
        _mark_sampling_complete(runtime)
        return results

    actor_module.run_actor_jobs = run_actor_jobs
    runtime_module.ContinuousMemoryRuntime.wait_quiescent = _wait_quiescent_v089
    _INSTALLED = True
