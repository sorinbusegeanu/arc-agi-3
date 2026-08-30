from __future__ import annotations

"""v8.63: isolate clean runs and keep mixed sampling globally authoritative."""

from pathlib import Path


_INSTALLED = False
_BASE_RUNTIME_INIT = None
_BASE_MARK_SAMPLING_COMPLETE = None
_BASE_REQUEST_FINAL_PEER_DRAIN = None
_BASE_RUN_MIXED_ACTOR_JOBS = None


def _purge_orphan_optimizer_state(root: str | Path) -> tuple[str, ...]:
    """Remove optimizer sidecars that have no restorable canonical graph state."""

    optimizer_root = Path(root) / "trajectory_optimizer"
    removed: list[str] = []
    for directory in (optimizer_root / "inbox", optimizer_root / "solutions_inbox"):
        if not directory.is_dir():
            continue
        for path in tuple(directory.glob("*.json")):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed.append(str(path.relative_to(Path(root))))
    for path in (
        optimizer_root / "validated.json",
        optimizer_root / "best_successful.json",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(str(path.relative_to(Path(root))))
    return tuple(sorted(removed))


def _runtime_init_v863(self, config, *args, **kwargs) -> None:
    from v8.snapshot import latest_complete_snapshot

    restore = bool(getattr(config, "restore", False))
    root = Path(getattr(config, "root"))
    restorable = latest_complete_snapshot(root) if restore else None
    removed = () if restorable is not None else _purge_orphan_optimizer_state(root)
    _BASE_RUNTIME_INIT(self, config, *args, **kwargs)
    self._v863_clean_run_purged_optimizer = removed


def _mixed_sampling_deferred(runtime) -> bool:
    return bool(getattr(runtime, "_v863_mixed_sampling_active", False))


def _mark_sampling_complete_v863(runtime) -> None:
    if _mixed_sampling_deferred(runtime):
        runtime._v863_mixed_final_drain_pending = True
        return
    return _BASE_MARK_SAMPLING_COMPLETE(runtime)


def _request_final_peer_drain_v863(runtime) -> None:
    if _mixed_sampling_deferred(runtime):
        runtime._v863_mixed_final_drain_pending = True
        return
    return _BASE_REQUEST_FINAL_PEER_DRAIN(runtime)


def _run_mixed_actor_jobs_v863(runtime, jobs, **kwargs):
    """Let ARC and generic jobs finish before declaring global sampling complete."""

    prior = bool(getattr(runtime, "_v863_mixed_sampling_active", False))
    runtime._v863_mixed_sampling_active = True
    runtime._v863_mixed_final_drain_pending = False
    completed = False
    try:
        result = _BASE_RUN_MIXED_ACTOR_JOBS(runtime, jobs, **kwargs)
        completed = True
        return result
    finally:
        runtime._v863_mixed_sampling_active = prior
        if completed and not prior:
            runtime._v863_mixed_final_drain_pending = False
            _BASE_REQUEST_FINAL_PEER_DRAIN(runtime)


def install_research_integrity_v863() -> None:
    global _INSTALLED, _BASE_RUNTIME_INIT, _BASE_MARK_SAMPLING_COMPLETE
    global _BASE_REQUEST_FINAL_PEER_DRAIN, _BASE_RUN_MIXED_ACTOR_JOBS
    if _INSTALLED:
        return

    from v8 import lease_dispatch_continuity_v839 as dispatch
    from v8 import mixed_environment_v859 as mixed
    from v8 import shutdown_semantics_v089 as shutdown
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    _BASE_MARK_SAMPLING_COMPLETE = shutdown._mark_sampling_complete
    _BASE_REQUEST_FINAL_PEER_DRAIN = dispatch._request_final_peer_drain
    _BASE_RUN_MIXED_ACTOR_JOBS = mixed.run_mixed_actor_jobs

    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v863
    shutdown._mark_sampling_complete = _mark_sampling_complete_v863
    dispatch._request_final_peer_drain = _request_final_peer_drain_v863
    mixed.run_mixed_actor_jobs = _run_mixed_actor_jobs_v863
    _INSTALLED = True
