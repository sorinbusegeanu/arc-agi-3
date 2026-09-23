from __future__ import annotations

import time
from threading import Event
from typing import Any

from v9.telemetry import build_primary_dashboard
from . import post_sampling_progress as post_progress


_LIVE_DIAGNOSTIC_KEYS = (
    "sampled_steps",
    "actor_produced_steps",
    "causally_admitted_steps",
    "publication_drained_steps",
    "ingested_steps",
    "sampling_backlog",
    "publication_backlog",
    "canonical_ingest_backlog",
    "sampling_rate",
    "ingestion_rate",
    "derivation_rate",
    "ingest_queue_depth",
    "derivation_queue_depth",
    "canonical_batch_size",
    "canonical_apply_latency_ms",
    "model_version",
    "run_phase",
)


def _diagnostic_snapshot(runtime: Any | None) -> dict[str, Any]:
    if runtime is None:
        return {}
    telemetry = getattr(runtime, "unified_telemetry", None)
    if telemetry is None:
        return {}
    try:
        return dict(telemetry.diagnostic_metrics())
    except Exception:
        return {}


def _set_run_phase_without_runtime_lock(value: str, runtime: Any | None = None) -> None:
    selected = runtime if runtime is not None else getattr(post_progress, "_current_runtime", None)
    if selected is None:
        return
    telemetry = getattr(selected, "unified_telemetry", None)
    gauges = getattr(telemetry, "gauges", None)
    if gauges is None:
        return
    try:
        gauges["run_phase"] = str(value)
    except Exception:
        return


def _heartbeat_loop_without_runtime_lock(stop: Event) -> None:
    while not stop.wait(max(0.1, float(post_progress._progress_interval_seconds))):
        runtime = post_progress._current_runtime
        with post_progress._state_lock:
            sampling_active = bool(post_progress._sampling_active)
            sampling_started = float(post_progress._sampling_started)
            sampling_label = str(post_progress._sampling_label)
        diagnostic = _diagnostic_snapshot(runtime)
        if sampling_active:
            elapsed = max(0.0, time.monotonic() - sampling_started)
            produced = int(diagnostic.get("sampled_steps", diagnostic.get("actor_produced_steps", 0)))
            published = int(
                diagnostic.get(
                    "causally_admitted_steps",
                    diagnostic.get("publication_drained_steps", 0),
                )
            )
            ingested = int(diagnostic.get("ingested_steps", 0))
            publication_backlog = int(
                diagnostic.get("publication_backlog", max(0, produced - published))
            )
            ingest_backlog = int(
                diagnostic.get("canonical_ingest_backlog", max(0, published - ingested))
            )
            post_progress.print(
                f"{time.strftime('[%H:%M]')} {sampling_label}/pipeline still active "
                f"elapsed={elapsed:.0f}s sampled={produced} published={published} "
                f"ingested={ingested} pub_backlog={publication_backlog} "
                f"ingest_backlog={ingest_backlog}",
                flush=True,
            )
            continue
        detail, elapsed = post_progress._phase_state()
        if not post_progress._active or not detail:
            continue
        post_progress.print(
            f"{time.strftime('[%H:%M]')} post-sampling phase={detail} elapsed={elapsed:.0f}s",
            flush=True,
        )


def install(runtime_cls: type) -> None:
    if getattr(runtime_cls, "_live_observability_installed", False):
        return

    current_dashboard = runtime_cls.dashboard_metrics

    def dashboard_metrics(self: Any) -> dict[str, Any]:
        # The wrapped dashboard path is already non-blocking. It may, however,
        # return an empty primary dashboard before the first exact runtime cut if
        # canonical mutation owns the runtime lock. Always construct a populated
        # lock-independent primary dashboard from the latest stable metrics cache
        # plus live unified telemetry.
        try:
            snapshot = dict(current_dashboard(self))
        except Exception as exc:
            snapshot = {
                "dashboard_metrics_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            }

        diagnostic = _diagnostic_snapshot(self)
        stable = dict(getattr(self, "_metrics_cache", {}) or {})
        for key, value in snapshot.items():
            if key not in {"primary_dashboard", "telemetry_diagnostics"}:
                stable[key] = value
        stable["telemetry_diagnostics"] = diagnostic

        primary = build_primary_dashboard(stable, diagnostic)
        existing_primary = dict(snapshot.get("primary_dashboard", {}) or {})
        primary.update(existing_primary)

        # Reapply the lock-independent counters last so a stale cache cannot
        # overwrite current pipeline progress.
        live_primary = build_primary_dashboard({}, diagnostic)
        for key in (
            "sampling_backlog",
            "ingest_queue_depth",
            "derivation_queue_depth",
            "sampling_rate",
            "ingestion_rate",
            "derivation_rate",
            "canonical_batch_size",
            "canonical_apply_latency_ms",
            "ModelVersion",
            "GPU_memory_GB",
        ):
            if key in live_primary:
                primary[key] = live_primary[key]

        run_phase = diagnostic.get("run_phase")
        if run_phase is not None:
            primary["run_phase"] = str(run_phase)
            snapshot["run_phase"] = str(run_phase)
        for key in _LIVE_DIAGNOSTIC_KEYS:
            if key in diagnostic:
                snapshot[key] = diagnostic[key]
        snapshot["telemetry_diagnostics"] = diagnostic
        snapshot["primary_dashboard"] = primary
        return snapshot

    runtime_cls.dashboard_metrics = dashboard_metrics
    runtime_cls._live_observability_installed = True

    # The heartbeat itself must never touch runtime.set_telemetry_gauge(), because
    # that can wait behind canonical mutation and silence stdout exactly when the
    # heartbeat is needed most.
    post_progress._set_run_phase = _set_run_phase_without_runtime_lock
    post_progress._heartbeat_loop = _heartbeat_loop_without_runtime_lock
