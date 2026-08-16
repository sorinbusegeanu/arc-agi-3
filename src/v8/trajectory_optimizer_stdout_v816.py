from __future__ import annotations

import threading
import time


_REPORT_INTERVAL_SECONDS = 300.0
_INSTALLED = False
_BASE_INIT = None
_BASE_START = None
_BASE_STOP = None
_BASE_ACCEPT = None


def _emit_success_report_if_due(self, *, force: bool = False) -> bool:
    now = time.monotonic()
    with self._lock:
        rounds = int(getattr(self, "_v816_report_rounds", 0))
        if rounds <= 0:
            return False
        interval = float(getattr(self, "_v816_report_interval_seconds", _REPORT_INTERVAL_SECONDS))
        last = float(getattr(self, "_v816_last_report_monotonic", 0.0))
        if not force and now - last < interval:
            return False

        parent_cost = int(getattr(self, "_v816_report_parent_cost", 0))
        candidate_cost = int(getattr(self, "_v816_report_candidate_cost", 0))
        saved = int(getattr(self, "_v816_report_saved", 0))
        best_parent = int(getattr(self, "_v816_report_best_parent", 0))
        best_candidate = int(getattr(self, "_v816_report_best_candidate", 0))
        validations = max(
            0,
            int(getattr(self, "_validations", 0))
            - int(getattr(self, "_v816_report_validation_baseline", 0)),
        )

        self._v816_report_rounds = 0
        self._v816_report_parent_cost = 0
        self._v816_report_candidate_cost = 0
        self._v816_report_saved = 0
        self._v816_report_best_parent = 0
        self._v816_report_best_candidate = 0
        self._v816_last_report_monotonic = now
        self._v816_report_validation_baseline = int(getattr(self, "_validations", 0))

    print(
        f'[{time.strftime("%H:%M")}] trajectory optimization complete '
        f"rounds={rounds} validations={validations} "
        f"cost={parent_cost}->{candidate_cost} saved={saved} "
        f"best={best_parent}->{best_candidate}",
        flush=True,
    )
    return True


def _record_successful_round(self, candidate) -> None:
    parent_cost = int(candidate.source.cost)
    candidate_cost = int(candidate.cost)
    saved = max(0, parent_cost - candidate_cost)
    if saved <= 0:
        return

    with self._lock:
        self._v816_report_rounds = int(getattr(self, "_v816_report_rounds", 0)) + 1
        self._v816_report_parent_cost = int(
            getattr(self, "_v816_report_parent_cost", 0)
        ) + parent_cost
        self._v816_report_candidate_cost = int(
            getattr(self, "_v816_report_candidate_cost", 0)
        ) + candidate_cost
        self._v816_report_saved = int(getattr(self, "_v816_report_saved", 0)) + saved

        prior_best_parent = int(getattr(self, "_v816_report_best_parent", 0))
        prior_best_candidate = int(getattr(self, "_v816_report_best_candidate", 0))
        prior_best_saved = max(0, prior_best_parent - prior_best_candidate)
        if (
            saved > prior_best_saved
            or (saved == prior_best_saved and candidate_cost < prior_best_candidate)
            or prior_best_parent <= 0
        ):
            self._v816_report_best_parent = parent_cost
            self._v816_report_best_candidate = candidate_cost

    _emit_success_report_if_due(self)


def _reporter_loop(self) -> None:
    while not self._stop.wait(1.0):
        _emit_success_report_if_due(self)


def install_trajectory_optimizer_stdout_v816() -> None:
    global _INSTALLED, _BASE_INIT, _BASE_START, _BASE_STOP, _BASE_ACCEPT
    if _INSTALLED:
        return

    from v8.trajectory_optimizer_v814 import TrajectoryOptimizationService

    _BASE_INIT = TrajectoryOptimizationService.__init__
    _BASE_START = TrajectoryOptimizationService.start
    _BASE_STOP = TrajectoryOptimizationService.stop
    _BASE_ACCEPT = TrajectoryOptimizationService._accept

    def init(self, *args, **kwargs):
        _BASE_INIT(self, *args, **kwargs)
        self._v816_report_interval_seconds = _REPORT_INTERVAL_SECONDS
        self._v816_last_report_monotonic = time.monotonic() - _REPORT_INTERVAL_SECONDS
        self._v816_report_validation_baseline = int(getattr(self, "_validations", 0))
        self._v816_report_rounds = 0
        self._v816_report_parent_cost = 0
        self._v816_report_candidate_cost = 0
        self._v816_report_saved = 0
        self._v816_report_best_parent = 0
        self._v816_report_best_candidate = 0
        self._v816_reporter_thread = None

    def start(self) -> None:
        _BASE_START(self)
        thread = getattr(self, "_v816_reporter_thread", None)
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=_reporter_loop,
            args=(self,),
            name="v8-trajectory-optimizer-reporter",
            daemon=True,
        )
        self._v816_reporter_thread = thread
        thread.start()

    def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        _BASE_STOP(self, drain=drain, timeout=timeout)
        thread = getattr(self, "_v816_reporter_thread", None)
        if thread is not None:
            thread.join(timeout=min(1.0, max(0.1, float(timeout))))

    def accept(self, candidate):
        row = _BASE_ACCEPT(self, candidate)
        if getattr(row, "variant_id", None) == getattr(candidate, "candidate_id", None):
            _record_successful_round(self, candidate)
        return row

    TrajectoryOptimizationService.__init__ = init
    TrajectoryOptimizationService.start = start
    TrajectoryOptimizationService.stop = stop
    TrajectoryOptimizationService._accept = accept
    TrajectoryOptimizationService._v816_emit_success_report_if_due = _emit_success_report_if_due
    _INSTALLED = True
