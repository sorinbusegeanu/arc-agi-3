from __future__ import annotations

import json
import resource
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

PROFILE_NAME = "hypothesis_suite_runtime_profile.json"
PROFILER_VERSION = "suite_runtime_v3"
_INSTALLED = False
_TLS = threading.local()
_ORIGINALS: dict[str, Any] = {}


def _state() -> dict[str, Any]:
    state = getattr(_TLS, "profile_state", None)
    if state is None:
        state = {"timings": {}, "call_counts": {}}
        _TLS.profile_state = state
    return state


def _reset_state() -> None:
    _TLS.profile_state = {"timings": {}, "call_counts": {}}


def _resource_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    rss_mb = rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0
    return {
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "max_rss_mb": float(rss_mb),
    }


def _add_timing(name: str, seconds: float) -> None:
    state = _state()
    timings = state.setdefault("timings", {})
    timings[name] = float(timings.get(name, 0.0) or 0.0) + float(seconds)
    counts = state.setdefault("call_counts", {})
    counts[name] = int(counts.get(name, 0) or 0) + 1


def _timed(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        _add_timing(name, time.perf_counter() - started)


def _simple_wrapper(name: str, original_key: str) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _timed(name, _ORIGINALS[original_key], *args, **kwargs)

    return wrapped


def _profile_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    suite_timings = dict(summary.get("timings") or {})
    derivation = dict(summary.get("derivation_summary") or {})
    derive_steps = {
        str(key): float(value)
        for key, value in dict(derivation.get("timings") or {}).items()
        if str(key).endswith("_seconds")
    }
    for key, value in summary.items():
        if str(key).startswith("DERIVE.") and str(key).endswith("_seconds"):
            try:
                derive_steps.setdefault(str(key), float(value))
            except (TypeError, ValueError):
                pass
    derive_total = float(suite_timings.get("derive_seconds", summary.get("derive_seconds", 0.0)) or 0.0)
    report_total = float(suite_timings.get("report_seconds", summary.get("report_seconds", 0.0)) or 0.0)
    evaluator_total = float(suite_timings.get("evaluator_total_seconds", summary.get("evaluator_total_seconds", 0.0)) or 0.0)
    suite_total = float(suite_timings.get("suite_total_seconds", summary.get("suite_total_seconds", 0.0)) or 0.0)
    return {
        "suite_total_seconds": suite_total,
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


def _load_evaluator_profile(output_dir: Path) -> dict[str, Any]:
    path = Path(output_dir) / "hypothesis_evaluator_profile.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _concept_validation_profile() -> dict[str, Any]:
    try:
        from v6.concept_validation_sparse_cache import get_last_concept_validation_profile
        payload = get_last_concept_validation_profile()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _phase_log_summary(output_dir: Path) -> dict[str, Any]:
    from v6 import hypothesis_suite_report as suite

    path = Path(output_dir) / suite.SUITE_PHASE_LOG_NAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    per_phase: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        phase = str(row.get("phase") or "unknown")
        status = str(row.get("status") or "unknown")
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        item = per_phase.setdefault(phase, {"events": 0, "heartbeats": 0})
        item["events"] = int(item.get("events", 0)) + 1
        if status in {"heartbeat", "progress"}:
            item["heartbeats"] = int(item.get("heartbeats", 0)) + 1
        if row.get("seconds_elapsed") is not None:
            try:
                item["last_seconds_elapsed"] = float(row["seconds_elapsed"])
            except (TypeError, ValueError):
                pass
        item["last_status"] = status
    return {
        "event_count": sum(status_counts.values()),
        "status_counts": status_counts,
        "phases": per_phase,
    }


@contextmanager
def _snapshot_profiled(memory_dir: Path | None) -> Iterator[Path | None]:
    started_total = time.perf_counter()
    manager = _ORIGINALS["read_only_evidence_snapshot"](memory_dir)
    started_setup = time.perf_counter()
    try:
        evidence_dir = manager.__enter__()
    except BaseException:
        _add_timing("report.snapshot_setup", time.perf_counter() - started_setup)
        _add_timing("report.snapshot_total", time.perf_counter() - started_total)
        raise
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


def _write_suite_summary_profiled(
    summary: Mapping[str, Any],
    output_dir: Path,
    *,
    hypothesis_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    started = time.perf_counter()
    _ORIGINALS["_write_suite_summary"](
        summary,
        output_dir,
        hypothesis_results=hypothesis_results,
    )
    _add_timing("summary.write", time.perf_counter() - started)

    output_dir = Path(output_dir)
    memory_raw = summary.get("memory_dir")
    memory_dir = None if memory_raw in (None, "") else Path(str(memory_raw))
    reported = _profile_from_summary(summary)
    state = _state()
    timings = dict(state.get("timings") or {})
    evaluator_profile = _load_evaluator_profile(output_dir)
    concept_profile = _concept_validation_profile()
    profile = {
        "profiler_version": PROFILER_VERSION,
        "epoch_id": summary.get("epoch_id"),
        "status": "done",
        "reported": reported,
        "instrumented_timings": timings,
        "call_counts": dict(state.get("call_counts") or {}),
        "instrumented_timings_sorted_desc": [
            {"name": key, "seconds": float(value)}
            for key, value in sorted(timings.items(), key=lambda item: float(item[1]), reverse=True)
        ],
        "resource_end": _resource_snapshot(),
        "database_sizes_bytes": _database_sizes(memory_dir),
        "phase_log": _phase_log_summary(output_dir),
        "evaluator_profile": evaluator_profile,
        "concept_validation_fastpath_profile": concept_profile,
    }
    derive_steps = reported.get("derive_step_seconds") or {}
    if derive_steps:
        slowest = max(derive_steps.items(), key=lambda item: float(item[1]))
        profile["slowest_derive_step"] = {"name": slowest[0], "seconds": float(slowest[1])}

    (output_dir / PROFILE_NAME).write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        from v6 import hypothesis_suite_report as suite
        suite.log_hypothesis_progress(
            output_dir,
            "PROFILE.runtime",
            "done",
            epoch_id=None if summary.get("epoch_id") is None else str(summary.get("epoch_id")),
            extra={
                "profiler": PROFILER_VERSION,
                "suite_total_seconds": reported.get("suite_total_seconds"),
                "derive_seconds": reported.get("derive_reported_seconds"),
                "report_seconds": reported.get("report_reported_seconds"),
                "top_level_unaccounted_seconds": reported.get("top_level_unaccounted_seconds"),
                "slowest_derive_step": profile.get("slowest_derive_step"),
                "concept_validation_profile_available": bool(concept_profile),
                **_resource_snapshot(),
            },
        )
    finally:
        _reset_state()


def install_suite_runtime_profiler() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v6 import hypothesis_suite_report as suite

    names = (
        "migrate_memory_dir",
        "ensure_memory_layout",
        "read_only_evidence_snapshot",
        "memory_fingerprint",
        "validate_hypothesis_provenance",
        "apply_decision_envelope",
        "_write_normalized_result",
        "build_hypothesis_suite_summary",
        "_write_suite_summary",
    )
    for name in names:
        _ORIGINALS[name] = getattr(suite, name)

    suite.migrate_memory_dir = _simple_wrapper("derive.schema_migration", "migrate_memory_dir")
    suite.ensure_memory_layout = _simple_wrapper("derive.ensure_memory_layout", "ensure_memory_layout")
    suite.read_only_evidence_snapshot = _snapshot_profiled
    suite.memory_fingerprint = _simple_wrapper("report.memory_fingerprint", "memory_fingerprint")
    suite.validate_hypothesis_provenance = _simple_wrapper("report.provenance_validation", "validate_hypothesis_provenance")
    suite.apply_decision_envelope = _simple_wrapper("report.decision_envelopes", "apply_decision_envelope")
    suite._write_normalized_result = _simple_wrapper("report.normalized_result_writes", "_write_normalized_result")
    suite.build_hypothesis_suite_summary = _simple_wrapper("summary.build", "build_hypothesis_suite_summary")
    suite._write_suite_summary = _write_suite_summary_profiled
    _INSTALLED = True
