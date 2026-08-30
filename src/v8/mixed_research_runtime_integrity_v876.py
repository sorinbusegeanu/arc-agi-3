from __future__ import annotations

"""v8.76 composition repairs for the v8.75 mixed-run integrity layer.

The v8.75 behavior is retained, but this layer restores historical public hook
identities and cached-read contracts that are required for a safe composed runtime.
It also keeps persisted ``generative_compression`` evidence admissible for H03 while
newer ``family_compression`` evidence remains canonical.
"""

import queue
import time
from dataclasses import replace


_INSTALLED = False


def _reporter_emit_line_v850(message: str, output_queue) -> None:
    """Keep the v8.50 public hook while no longer suppressing dedicated reports."""
    from v8 import learning_effectiveness_report_v850 as effectiveness

    effectiveness._BASE_REPORTER_EMIT_LINE(message, output_queue)


def _reporting_worker_v851_integrity(
    *,
    event_queue,
    stop_event,
    watermark,
    actors,
    interval_seconds: float,
    output_queue=None,
    hypothesis_interval_seconds: float = 300.0,
    total_steps: int | None = None,
    baseline=None,
) -> None:
    """Disk-authoritative reporter with one terminal all-job 100% flush."""
    from v8 import memory_efficiency_v851_integrity as memory_integrity
    from v8 import reporter
    from v8.actor import ActorProgress
    from v8.evidence import EvidenceRecord
    from v8.evidence_memory_v851 import _ROOT_ENV
    from v8.runtime_observability_v836 import _hypothesis_status_line
    import os

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    root = str(os.environ.get(_ROOT_ENV, "."))
    now = time.monotonic()
    next_report = now + float(interval_seconds)
    next_hypotheses = now + max(0.001, float(hypothesis_interval_seconds))
    saw_progress = False

    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(0.0, min(0.25, next_report - now, next_hypotheses - now))
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None

        if isinstance(row, ActorProgress):
            latest[int(row.actor_id)] = row
            saw_progress = True
        elif isinstance(row, EvidenceRecord):
            # The authoritative evidence source is the disk ledger read below.
            pass
        elif row == reporter.SAMPLING_COMPLETE:
            if saw_progress:
                rows = tuple(latest[key] for key in sorted(latest))
                line = reporter.format_periodic_progress_line(
                    rows, total_steps, baseline
                )
                from v8 import mixed_research_runtime_integrity_v875 as v875

                reporter._emit_line(v875._force_complete_percentage(line), output_queue)
            reporter._emit_sampling_complete(output_queue)
            return

        now = time.monotonic()
        if now >= next_report:
            rows = tuple(latest[key] for key in sorted(latest))
            reporter._emit_line(
                reporter.format_periodic_progress_line(rows, total_steps, baseline),
                output_queue,
            )
            while next_report <= now:
                next_report += float(interval_seconds)

        if now >= next_hypotheses:
            current = int(getattr(watermark, "value", 0))
            evidence = memory_integrity._read_evidence_for_report(root, current)
            reporter._emit_line(
                _hypothesis_status_line(evidence, current),
                output_queue,
            )
            del evidence
            while next_hypotheses <= now:
                next_hypotheses += max(0.001, float(hypothesis_interval_seconds))


def _authoritative_telemetry_game_state_v875(self, game_id: str):
    """Use cached lifecycle state while still consuming a durable runtime WIN marker."""
    from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
    from v8 import runtime_win_optimization_v834 as runtime_win

    game = str(game_id)
    if all(hasattr(self, name) for name in ("_lock", "_game_won", "_records")):
        runtime_win._promote_runtime_win_if_present(self, game)
    return solve_fix._cached_game_state(self, game)


def _request_final_peer_drain_v875(runtime) -> None:
    """Only the outer mixed runner may transition the runtime into final drain."""
    from v8 import mixed_research_runtime_integrity_v875 as v875

    if bool(getattr(runtime, "_v839_defer_sampling_finish", False)):
        return
    v875._emit_phase(runtime, "optimizer_drain")
    return v875._BASE_FINAL_PEER_DRAIN(runtime)


def _run_mixed_actor_jobs_v875(runtime, jobs, **kwargs):
    """Outer mixed completion authority without an unnecessary eager runtime start."""
    from v8 import mixed_environment_v859 as mixed
    from v8 import mixed_research_runtime_integrity_v875 as v875

    jobs = tuple(jobs)
    reporting_queue = kwargs.get("reporting_queue")
    prior_defer = bool(getattr(runtime, "_v839_defer_sampling_finish", False))
    runtime._v839_defer_sampling_finish = True
    runtime._v839_sampling_done_reported = False
    runtime._v875_reported_phases = set()
    runtime._v875_arc_done = not any(
        not mixed.is_generic_game(job.game_id) for job in jobs
    )
    runtime._v875_generic_active = False

    try:
        result = tuple(v875._BASE_MIXED_RUN(runtime, jobs, **kwargs))
    finally:
        runtime._v839_defer_sampling_finish = prior_defer

    if prior_defer:
        return result

    v875._publish_final_job_progress(reporting_queue, jobs, result)
    _request_final_peer_drain_v875(runtime)
    if reporting_queue is not None:
        from v8.reporter import SAMPLING_COMPLETE

        reporting_queue.put_nowait(SAMPLING_COMPLETE)
    runtime._v839_sampling_done_reported = True
    return result


def _install_h03_compression_compatibility() -> None:
    """Accept persisted v8.7 evidence without changing the canonical H03 meaning."""
    from v8 import evaluation

    rows = []
    for contract in evaluation.CONTRACTS:
        if contract.hypothesis_id == "H03":
            required = tuple(dict.fromkeys((*contract.required_kinds, "generative_compression")))
            contract = replace(contract, required_kinds=required)
        rows.append(contract)
    evaluation.CONTRACTS = tuple(rows)


def install_mixed_research_runtime_integrity_v876() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as allocation
    from v8 import intelligence_loop_v087 as intelligence
    from v8 import learning_effectiveness_report_v850 as effectiveness
    from v8 import lease_dispatch_continuity_v839 as lease
    from v8 import memory_efficiency_v851_integrity as memory_integrity
    from v8 import mixed_environment_v859 as mixed
    from v8 import mixed_research_runtime_integrity_v875 as v875
    from v8 import reporter
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    # Preserve the original M2 formation evidence label for persisted/runtime
    # compatibility; H03 explicitly accepts it as the legacy spelling below.
    intelligence._compression_to_candidate = v875._BASE_COMPRESSION_TO_CANDIDATE
    _install_h03_compression_compatibility()

    allocation.AdaptiveLearningCoordinator._v819_telemetry_game_state = (
        _authoritative_telemetry_game_state_v875
    )
    v875._authoritative_telemetry_game_state_v875 = (
        _authoritative_telemetry_game_state_v875
    )

    # Preserve v8.50's public hook identity while removing its suppression rule.
    effectiveness._reporter_emit_line_v850 = _reporter_emit_line_v850
    reporter._emit_line = effectiveness._reporter_emit_line_v850

    # Preserve the final v8.51 reporter authority/name while adding terminal flush.
    memory_integrity._reporting_worker_v851_integrity = _reporting_worker_v851_integrity
    reporter.reporting_worker = memory_integrity._reporting_worker_v851_integrity
    v875._reporting_worker_v875 = memory_integrity._reporting_worker_v851_integrity

    # v8.62 remains the public wait_quiescent authority. The optimizer phase is
    # emitted at the actual final-drain transition instead of wrapping this method.
    V82ContinuousMemoryRuntime.wait_quiescent = v875._BASE_WAIT_QUIESCENT

    v875._request_final_peer_drain_v875 = _request_final_peer_drain_v875
    lease._request_final_peer_drain = v875._request_final_peer_drain_v875

    v875._run_mixed_actor_jobs_v875 = _run_mixed_actor_jobs_v875
    mixed.run_mixed_actor_jobs = v875._run_mixed_actor_jobs_v875

    _INSTALLED = True
