from __future__ import annotations

import json
import os
import resource
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

PROFILE_NAME = "hypothesis_suite_runtime_profile.json"
_INSTALLED = False
_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("v6_suite_runtime_profile", default=None)
_ORIGINALS: dict[str, Any] = {}


def _resource_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # Linux reports KiB; macOS reports bytes. ARC runtime is Linux, but keep the
    # conversion conservative for local development on Darwin.
    rss_mb = rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0
    return {
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "max_rss_mb": float(rss_mb),
    }


def _add_timing(name: str, seconds: float) -> None:
    profile = _ACTIVE.get()
    if profile is None:
        return
    timings = profile.setdefault("timings", {})
    timings[name] = float(timings.get(name, 0.0) or 0.0) + float(seconds)
    counts = profile.setdefault("call_counts", {})
    counts[name] = int(counts.get(name, 0) or 0) + 1


def _timed(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        _add_timing(name, time.perf_counter() - started)


def _profile_from_summary(summary: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    suite_timings = dict(summary.get("timings") or {})
    derivation = dict(summary.get("derivation_summary") or {})
    derive_steps = {
        str(key): float(value)
        for key, value in dict(derivation.get("timings") or {}).items()
        if str(key).endswith("_seconds")
    }
    derive_total = float(suite_timings.get("derive_seconds", 0.0) or 0.0)
    report_total = float(suite_timings.get("report_seconds", 0.0) or 0.0)
    evaluator_total = float(suite_timings.get("evaluator_total_seconds", 0.0) or 0.0)
    suite_total = float(suite_timings.get("suite_total_seconds", wall_seconds) or wall_seconds)
    return {
        "suite_total_seconds": suite_total,
        "wrapper_wall_seconds": float(wall_seconds),
        "derive_reported_seconds": derive_total,
        "report_reported_seconds": report_total,
        "evaluator_reported_seconds": evaluator_total,
        "derive_step_seconds": derive_steps,
        "top_level_unaccounted_seconds": max(0.0, suite_total - derive_total - report_total),
        "report_non_evaluator_seconds": max(0.0, report_total - evaluator_total),
    }


def _database_sizes(memory_dir: Path | None) -> dict[str, int]:
    if memory_dir is None:
        return {}
    result: dict[str, int] = {}
    for name in (
        "current_state.sqlite",
        "graph.sqlite",
        "replay_queue.sqlite",
        "direct_streaming_fold_manifest.sqlite",
    ):
        path = Path(memory_dir) / name
        try:
            if path.is_file():
                result[name] = int(path.stat().st_size)
        except OSError:
            pass
    return result


def _existing_evaluator_counts(output_dir: Path) -> dict[str, Any]:
    path = Path(output_dir) / "hypothesis_evaluator_profile.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return {
        "shared_cache_rows_indexed": payload.get("shared_cache_rows_indexed"),
        "shared_cache_table_counts": payload.get("shared_cache_table_counts", {}),
        "evaluator_workers": payload.get("evaluator_workers"),
    }


def _heartbeat(output_dir: Path, phase: str, epoch_id: str | None, started: float, stop: threading.Event) -> None:
    from v6 import hypothesis_suite_report as suite

    interval = max(5.0, float(os.getenv("ARC_HYPOTHESIS_PROFILE_HEARTBEAT_SECONDS", "30")))
    while not stop.wait(interval):
        elapsed = time.perf_counter() - started
        resources = _resource_snapshot()
        suite.log_hypothesis_progress(
            output_dir,
            phase,
            "heartbeat",
            epoch_id=epoch_id,
            start_time=time.time() - elapsed,
            extra={**resources, "profiler": "suite_runtime_v1"},
        )
        profile = _ACTIVE.get()
        # ContextVars do not propagate into new threads by default, therefore
        # heartbeat data is emitted to the phase log; the parent records counts.
        del profile


def _phase_profiled(output_dir: Path, epoch_id: str | None, name: str, callback: Callable[[], Any], timings: dict[str, float]) -> Any:
    stop = threading.Event()
    started = time.perf_counter()
    thread = threading.Thread(
        target=_heartbeat,
        args=(Path(output_dir), str(name), epoch_id, started, stop),
        daemon=True,
        name=f"suite-profile-{name[-24:]}",
    )
    thread.start()

    def profiled_callback() -> Any:
        return _timed(f"callback.{name}", callback)

    try:
        return _ORIGINALS["_phase"](output_dir, epoch_id, name, profiled_callback, timings)
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _simple_wrapper(name: str, original_key: str) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _timed(name, _ORIGINALS[original_key], *args, **kwargs)

    return wrapped


@contextmanager
def _snapshot_profiled(memory_dir: Path | None) -> Iterator[Path | None]:
    started_total = time.perf_counter()
    manager = _ORIGINALS["read_only_evidence_snapshot"](memory_dir)
    started_setup = time.perf_counter()
    evidence_dir = manager.__enter__()
    _add_timing("report.snapshot_setup", time.perf_counter() - started_setup)
    try:
        yield evidence_dir
    except BaseException as exc:
        started_teardown = time.perf_counter()
        manager.__exit__(type(exc), exc, exc.__traceback__)
        _add_timing("report.snapshot_teardown", time.perf_counter() - started_teardown)
        raise
    else:
        started_teardown = time.perf_counter()
        manager.__exit__(None, None, None)
        _add_timing("report.snapshot_teardown", time.perf_counter() - started_teardown)
    finally:
        _add_timing("report.snapshot_total", time.perf_counter() - started_total)


def _run_profiled(*args: Any, **kwargs: Any) -> Any:
    output_dir = Path(kwargs.get("output_dir") or ".")
    memory_dir_raw = kwargs.get("memory_dir")
    memory_dir = None if memory_dir_raw is None else Path(memory_dir_raw)
    profile: dict[str, Any] = {
        "profiler_version": "suite_runtime_v1",
        "epoch_id": kwargs.get("epoch_id"),
        "timings": {},
        "call_counts": {},
        "resource_start": _resource_snapshot(),
        "database_sizes_before": _database_sizes(memory_dir),
    }
    token = _ACTIVE.set(profile)
    started = time.perf_counter()
    try:
        result = _ORIGINALS["run_hypothesis_suite_report"](*args, **kwargs)
    except BaseException as exc:
        profile["status"] = "failed"
        profile["exception_type"] = type(exc).__name__
        profile["exception_message"] = str(exc)
        raise
    else:
        profile["status"] = "done"
        if isinstance(result, dict):
            profile["reported"] = _profile_from_summary(result, time.perf_counter() - started)
        return result
    finally:
        wall = time.perf_counter() - started
        profile["wall_seconds"] = float(wall)
        profile["resource_end"] = _resource_snapshot()
        profile["database_sizes_after"] = _database_sizes(memory_dir)
        profile["evaluator_cache"] = _existing_evaluator_counts(output_dir)
        profile["timings_sorted_desc"] = [
            {"name": key, "seconds": float(value)}
            for key, value in sorted(
                dict(profile.get("timings") or {}).items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            (output_dir / PROFILE_NAME).write_text(
                json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
            )
        finally:
            _ACTIVE.reset(token)


def install_suite_runtime_profiler() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v6 import hypothesis_suite_report as suite

    names = (
        "run_hypothesis_suite_report",
        "_phase",
        "migrate_memory_dir",
        "ensure_memory_layout",
        "read_only_evidence_snapshot",
        "memory_fingerprint",
        "validate_hypothesis_provenance",
        "evaluate_hypotheses_read_only",
        "apply_decision_envelope",
        "_write_normalized_result",
        "build_hypothesis_suite_summary",
        "_write_suite_summary",
    )
    for name in names:
        _ORIGINALS[name] = getattr(suite, name)

    suite._phase = _phase_profiled
    suite.migrate_memory_dir = _simple_wrapper("derive.schema_migration", "migrate_memory_dir")
    suite.ensure_memory_layout = _simple_wrapper("derive.ensure_memory_layout", "ensure_memory_layout")
    suite.read_only_evidence_snapshot = _snapshot_profiled
    suite.memory_fingerprint = _simple_wrapper("report.memory_fingerprint", "memory_fingerprint")
    suite.validate_hypothesis_provenance = _simple_wrapper("report.provenance_validation", "validate_hypothesis_provenance")
    suite.evaluate_hypotheses_read_only = _simple_wrapper("report.evaluators_wall", "evaluate_hypotheses_read_only")
    suite.apply_decision_envelope = _simple_wrapper("report.decision_envelopes", "apply_decision_envelope")
    suite._write_normalized_result = _simple_wrapper("report.normalized_result_writes", "_write_normalized_result")
    suite.build_hypothesis_suite_summary = _simple_wrapper("summary.build", "build_hypothesis_suite_summary")
    suite._write_suite_summary = _simple_wrapper("summary.write", "_write_suite_summary")
    suite.run_hypothesis_suite_report = _run_profiled
    _INSTALLED = True
