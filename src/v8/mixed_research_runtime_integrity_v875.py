from __future__ import annotations

"""v8.75 mixed-run and research-integrity repairs.

This late layer keeps historical runtime layers intact while making the composed
production behavior authoritative in five places:

* normalized M2 formation emits the canonical family-compression evidence kind;
* adaptive telemetry reads the live lifecycle/win authority instead of a stale cache;
* mixed runs alone own sampling completion and final peer drain;
* generic mixed-environment actors run in child processes against the shared ring;
* the dedicated reporter remains visible and emits an explicit terminal 100% line.

It also emits coarse run phases and serializes the generic replay store across
processes so process-based generic sampling cannot lose a concurrent promotion.
"""

import os
import queue
import re
import time
from dataclasses import replace
from pathlib import Path


_INSTALLED = False
_BASE_COMPRESSION_TO_CANDIDATE = None
_BASE_TELEMETRY_GAME_STATE = None
_BASE_REPORTER_EMIT_LINE = None
_BASE_REPORTING_WORKER = None
_BASE_MIXED_RUN = None
_BASE_MIXED_ARC_RUN = None
_BASE_GENERIC_JOBS = None
_BASE_FINAL_PEER_DRAIN = None
_BASE_WAIT_QUIESCENT = None
_BASE_FINAL_SNAPSHOT = None
_BASE_PERSIST_GENERIC_WIN = None

_FINAL_PERCENT_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?% - ")


def _compression_to_candidate_v875(proposal):
    candidate = _BASE_COMPRESSION_TO_CANDIDATE(proposal)
    if str(getattr(candidate, "evidence_kind", "")) != "generative_compression":
        return candidate
    return replace(candidate, evidence_kind="family_compression")


def _authoritative_telemetry_game_state_v875(self, game_id: str):
    """Report the same live state authority used by allocation/verification."""
    return self.game_state(str(game_id))


def _emit_phase(runtime, phase: str) -> None:
    phase = str(phase).strip().lower().replace(" ", "_")
    if not phase:
        return
    emitted = getattr(runtime, "_v875_reported_phases", None)
    if not isinstance(emitted, set):
        emitted = set()
        runtime._v875_reported_phases = emitted
    if phase in emitted:
        return
    emitted.add(phase)
    from v8 import reporter

    reporter._emit_line(f"phase={phase}", None)


def _force_complete_percentage(line: str) -> str:
    text = str(line)
    if _FINAL_PERCENT_PATTERN.match(text):
        return _FINAL_PERCENT_PATTERN.sub("100% - ", text, count=1)
    return f"100% - {text}"


def _reporting_worker_v875(
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
    """v8.36 reporter with an authoritative final all-job progress emission."""
    from v8 import reporter
    from v8.actor import ActorProgress
    from v8.evidence import EvidenceRecord
    from v8.runtime_observability_v836 import _AsyncHypothesisReporter

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    evidence_by_id = {}
    now = time.monotonic()
    next_report = now + float(interval_seconds)
    next_hypotheses = now + max(0.001, float(hypothesis_interval_seconds))
    hypotheses = _AsyncHypothesisReporter()

    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(0.0, min(0.25, next_report - now, next_hypotheses - now))
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None

        if isinstance(row, ActorProgress):
            latest[int(row.actor_id)] = row
        elif isinstance(row, EvidenceRecord):
            evidence_by_id[str(row.evidence_id)] = row
        elif row == reporter.SAMPLING_COMPLETE:
            rows = tuple(latest[key] for key in sorted(latest))
            final_line = reporter.format_periodic_progress_line(
                rows, total_steps, baseline
            )
            reporter._emit_line(_force_complete_percentage(final_line), output_queue)
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

        hypotheses.emit_ready(output_queue)
        if now >= next_hypotheses:
            hypotheses.start(
                tuple(evidence_by_id.values()),
                int(getattr(watermark, "value", 0)),
            )
            while next_hypotheses <= now:
                next_hypotheses += max(0.001, float(hypothesis_interval_seconds))


class _SamplingCompletionFilter:
    """Forward actor progress while keeping SAMPLING_COMPLETE at mixed-run scope."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def put_nowait(self, value) -> None:
        from v8.reporter import SAMPLING_COMPLETE

        if value == SAMPLING_COMPLETE:
            return
        self._inner.put_nowait(value)

    def put(self, value, *args, **kwargs) -> None:
        from v8.reporter import SAMPLING_COMPLETE

        if value == SAMPLING_COMPLETE:
            return
        self._inner.put(value, *args, **kwargs)


def _request_final_peer_drain_v875(runtime) -> None:
    if bool(getattr(runtime, "_v839_defer_sampling_finish", False)):
        return
    return _BASE_FINAL_PEER_DRAIN(runtime)


def _run_arc_jobs_v875(runtime, jobs, **kwargs):
    reporting_queue = kwargs.get("reporting_queue")
    deferred = bool(getattr(runtime, "_v839_defer_sampling_finish", False))
    if deferred and reporting_queue is not None:
        kwargs["reporting_queue"] = _SamplingCompletionFilter(reporting_queue)
    try:
        return _BASE_MIXED_ARC_RUN(runtime, jobs, **kwargs)
    finally:
        if deferred:
            # v8.39 records this flag when it attempts its filtered inner sentinel.
            # Only the outer mixed runner may make it authoritative.
            runtime._v839_sampling_done_reported = False
        runtime._v875_arc_done = True
        if bool(getattr(runtime, "_v875_generic_active", False)):
            _emit_phase(runtime, "generic_sampling")


class _GenericProcessRuntime:
    """Minimal child-side runtime facade over the production shared experience ring."""

    def __init__(
        self,
        *,
        experience_ring_args,
        read_descriptors,
        watermark,
        stop_event,
        snapshot_freeze,
    ) -> None:
        from v8.ring import SharedRingBuffer

        self.shard_descriptors = tuple(read_descriptors)
        self._watermark = watermark
        self._stop = stop_event
        self._snapshot_freeze = snapshot_freeze
        self._ring = SharedRingBuffer(**experience_ring_args)

    @property
    def watermark(self) -> int:
        with self._watermark.get_lock():
            return int(self._watermark.value)

    def make_experience(self, **kwargs):
        from v8.model import EventId, ExperienceEvent

        producer_id = int(kwargs["producer_id"])
        producer_sequence = int(kwargs["producer_sequence"])
        return ExperienceEvent(
            event_id=EventId.from_producer(producer_id, producer_sequence),
            watermark=0,
            producer_id=producer_id,
            producer_sequence=producer_sequence,
            source_game_hash=int(kwargs["source_game_hash"]),
            global_step=max(0, int(kwargs.get("global_step", 0))),
            context_signature=int(kwargs["context_signature"]),
            action_id=int(kwargs["action_id"]),
            outcome_signature=int(kwargs["outcome_signature"]),
            family_signature=int(kwargs["family_signature"]),
            carrier_signature=int(kwargs.get("carrier_signature", 0)),
            future_option_delta=float(kwargs.get("future_option_delta", 0.0)),
            changed_cells=max(0, int(kwargs.get("changed_cells", 0))),
            terminal_polarity=int(kwargs.get("terminal_polarity", 0)),
            trajectory_signature=int(kwargs.get("trajectory_signature", 0)),
            next_context_signature=int(kwargs.get("next_context_signature", 0)),
            prediction_error=max(0.0, float(kwargs.get("prediction_error", 0.0))),
        )

    def submit(self, event) -> None:
        from v8.actor import _publish_actor_packet
        from v8.model import PipelineEvent, encode_pipeline

        def packet_for_watermark(current_watermark: int) -> bytes:
            accepted = replace(
                event,
                watermark=int(current_watermark),
                global_step=int(current_watermark),
            )
            return encode_pipeline(PipelineEvent(accepted))

        published = _publish_actor_packet(
            self._ring,
            self._watermark,
            packet_for_watermark,
            stop_event=self._stop,
            snapshot_freeze=self._snapshot_freeze,
        )
        if published is None and not self._stop.is_set():
            raise RuntimeError("generic actor failed to publish experience")

    def close(self) -> None:
        self._ring.close()


def _generic_process_worker_v875(
    *,
    job,
    experience_ring_args,
    read_descriptors,
    watermark,
    stop_event,
    snapshot_freeze,
    result_queue,
    reporting_queue,
) -> None:
    runtime = _GenericProcessRuntime(
        experience_ring_args=experience_ring_args,
        read_descriptors=read_descriptors,
        watermark=watermark,
        stop_event=stop_event,
        snapshot_freeze=snapshot_freeze,
    )
    try:
        from v8 import mixed_environment_v859 as mixed

        result = mixed.run_generic_actor_job(
            runtime, job, reporting_queue=reporting_queue
        )
        result_queue.put(("result", result))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                int(getattr(job, "actor_id", -1)),
                f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    finally:
        runtime.close()


def _supports_generic_processes(runtime) -> bool:
    return bool(
        getattr(runtime, "_mp_ctx", None) is not None
        and getattr(runtime, "_stage_rings", None)
        and getattr(runtime, "_watermark", None) is not None
        and getattr(runtime, "_stop", None) is not None
        and getattr(runtime, "_snapshot_freeze", None) is not None
        and getattr(runtime, "shard_descriptors", None)
    )


def _run_generic_jobs_v875(runtime, jobs, *, reporting_queue=None):
    jobs = tuple(jobs)
    if not jobs:
        return ()
    if not _supports_generic_processes(runtime):
        return _BASE_GENERIC_JOBS(
            runtime, jobs, reporting_queue=reporting_queue
        )

    runtime.start()
    _emit_phase(runtime, "generic_sampling")
    runtime._v875_generic_active = True
    ctx = runtime._mp_ctx
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_generic_process_worker_v875,
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
    by_actor = {}
    errors: list[str] = []
    try:
        for process in processes:
            process.start()
        while len(by_actor) < len(jobs):
            try:
                message = results.get(timeout=0.05)
            except queue.Empty:
                message = None
            if isinstance(message, tuple) and message:
                if message[0] == "result":
                    row = message[1]
                    by_actor[int(row.actor_id)] = row
                elif message[0] == "error":
                    errors.append(str(message[2]))
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
                missing = sorted(
                    int(job.actor_id) for job in jobs if int(job.actor_id) not in by_actor
                )
                raise RuntimeError(f"generic actor result missing for ids: {missing}")
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
            _emit_phase(runtime, "arc_tail")
        try:
            results.cancel_join_thread()
        except (AttributeError, ValueError):
            pass
        try:
            results.close()
        except (AttributeError, ValueError):
            pass


def _publish_final_job_progress(reporting_queue, jobs, results) -> None:
    if reporting_queue is None:
        return
    from v8.actor import ActorProgress

    job_by_actor = {int(job.actor_id): job for job in jobs}
    for row in sorted(results, key=lambda value: int(value.actor_id)):
        # Keep actual interaction counters intact. The terminal reporter explicitly
        # marks the completed job budget as 100% without fabricating interactions.
        _job = job_by_actor.get(int(row.actor_id))
        progress = ActorProgress(
            int(row.actor_id),
            str(row.game_id),
            max(0, int(row.steps)),
            max(0, int(row.wins)),
            max(0, int(row.failures)),
            max(0, int(row.levels_completed)),
            max(0, int(getattr(row, "replans", 0))),
            max(0, int(getattr(row, "planned_steps", 0))),
        )
        reporting_queue.put_nowait(progress)


def _run_mixed_actor_jobs_v875(runtime, jobs, **kwargs):
    jobs = tuple(jobs)
    reporting_queue = kwargs.get("reporting_queue")
    prior_defer = bool(getattr(runtime, "_v839_defer_sampling_finish", False))
    runtime._v839_defer_sampling_finish = True
    runtime._v839_sampling_done_reported = False
    runtime._v875_reported_phases = set()

    from v8 import mixed_environment_v859 as mixed

    runtime._v875_arc_done = not any(
        not mixed.is_generic_game(job.game_id) for job in jobs
    )
    runtime._v875_generic_active = False
    runtime.start()
    try:
        result = tuple(_BASE_MIXED_RUN(runtime, jobs, **kwargs))
    finally:
        runtime._v839_defer_sampling_finish = prior_defer

    if prior_defer:
        return result

    _publish_final_job_progress(reporting_queue, jobs, result)
    _BASE_FINAL_PEER_DRAIN(runtime)
    if reporting_queue is not None:
        from v8.reporter import SAMPLING_COMPLETE

        reporting_queue.put_nowait(SAMPLING_COMPLETE)
    runtime._v839_sampling_done_reported = True
    return result


def _wait_quiescent_v875(self, *args, **kwargs):
    if bool(getattr(self, "_sampling_complete", False)):
        _emit_phase(self, "optimizer_drain")
    return _BASE_WAIT_QUIESCENT(self, *args, **kwargs)


def _final_snapshot_v875(self, *args, **kwargs):
    _emit_phase(self, "final_snapshot")
    return _BASE_FINAL_SNAPSHOT(self, *args, **kwargs)


def _persist_generic_win_v875(success_root, event) -> bool:
    """Serialize the read-modify-write generic replay store across worker processes."""
    try:
        import fcntl
    except ImportError:
        return bool(_BASE_PERSIST_GENERIC_WIN(success_root, event))

    from v8 import restored_competence_v872 as restored

    runtime_root = restored._runtime_root_from_success_root(success_root)
    if runtime_root is None:
        return bool(_BASE_PERSIST_GENERIC_WIN(success_root, event))
    target = restored._generic_store_path(runtime_root)
    lock_path = Path(str(target) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            return bool(_BASE_PERSIST_GENERIC_WIN(success_root, event))
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def install_mixed_research_runtime_integrity_v875() -> None:
    global _INSTALLED
    global _BASE_COMPRESSION_TO_CANDIDATE, _BASE_TELEMETRY_GAME_STATE
    global _BASE_REPORTER_EMIT_LINE, _BASE_REPORTING_WORKER
    global _BASE_MIXED_RUN, _BASE_MIXED_ARC_RUN, _BASE_GENERIC_JOBS
    global _BASE_FINAL_PEER_DRAIN, _BASE_WAIT_QUIESCENT, _BASE_FINAL_SNAPSHOT
    global _BASE_PERSIST_GENERIC_WIN
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as allocation
    from v8 import intelligence_loop_v087 as intelligence
    from v8 import learning_effectiveness_report_v850 as effectiveness
    from v8 import lease_dispatch_continuity_v839 as lease
    from v8 import mixed_environment_v859 as mixed
    from v8 import reporter
    from v8 import restored_competence_v872 as restored
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_COMPRESSION_TO_CANDIDATE = intelligence._compression_to_candidate
    intelligence._compression_to_candidate = _compression_to_candidate_v875

    _BASE_TELEMETRY_GAME_STATE = getattr(
        allocation.AdaptiveLearningCoordinator,
        "_v819_telemetry_game_state",
        None,
    )
    allocation.AdaptiveLearningCoordinator._v819_telemetry_game_state = (
        _authoritative_telemetry_game_state_v875
    )

    # v8.50 suppressed the dedicated reporter's real effectiveness line on stdout.
    # Restore the pre-v8.50 emitter; v8.66 already owns the authoritative values.
    _BASE_REPORTER_EMIT_LINE = reporter._emit_line
    if reporter._emit_line is effectiveness._reporter_emit_line_v850:
        reporter._emit_line = effectiveness._BASE_REPORTER_EMIT_LINE

    _BASE_REPORTING_WORKER = reporter.reporting_worker
    reporter.reporting_worker = _reporting_worker_v875

    _BASE_FINAL_PEER_DRAIN = lease._request_final_peer_drain
    lease._request_final_peer_drain = _request_final_peer_drain_v875

    _BASE_GENERIC_JOBS = mixed._run_generic_jobs
    mixed._run_generic_jobs = _run_generic_jobs_v875
    _BASE_MIXED_ARC_RUN = mixed.run_arc_actor_jobs
    mixed.run_arc_actor_jobs = _run_arc_jobs_v875
    _BASE_MIXED_RUN = mixed.run_mixed_actor_jobs
    mixed.run_mixed_actor_jobs = _run_mixed_actor_jobs_v875

    _BASE_WAIT_QUIESCENT = V82ContinuousMemoryRuntime.wait_quiescent
    V82ContinuousMemoryRuntime.wait_quiescent = _wait_quiescent_v875
    _BASE_FINAL_SNAPSHOT = V82ContinuousMemoryRuntime.final_snapshot
    V82ContinuousMemoryRuntime.final_snapshot = _final_snapshot_v875

    _BASE_PERSIST_GENERIC_WIN = restored.persist_generic_win_v872
    restored.persist_generic_win_v872 = _persist_generic_win_v875

    _INSTALLED = True
