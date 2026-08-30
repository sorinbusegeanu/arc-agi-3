from __future__ import annotations

"""v8.63: isolate clean runs and keep mixed sampling globally authoritative."""

from pathlib import Path


_INSTALLED = False
_BASE_MARK_SAMPLING_COMPLETE = None
_BASE_REQUEST_FINAL_PEER_DRAIN = None
_BASE_RUN_MIXED_ACTOR_JOBS = None


def _purge_orphan_optimizer_state(root: str | Path) -> tuple[str, ...]:
    """Remove optimizer sidecars that have no restorable canonical graph state."""

    root = Path(root)
    optimizer_root = root / "trajectory_optimizer"
    removed: list[str] = []
    for directory in (optimizer_root / "inbox", optimizer_root / "solutions_inbox"):
        if not directory.is_dir():
            continue
        for path in tuple(directory.glob("*.json")):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed.append(str(path.relative_to(root)))
    for path in (
        optimizer_root / "validated.json",
        optimizer_root / "best_successful.json",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(str(path.relative_to(root)))
    return tuple(sorted(removed))


def prepare_clean_continuous_run(args) -> tuple[str, ...]:
    """Discard optimizer behavior only when this CLI run has no graph to restore."""

    from v8.snapshot import latest_complete_snapshot

    root = Path(getattr(args, "root"))
    restore = not bool(getattr(args, "no_restore", False))
    restorable = latest_complete_snapshot(root) if restore else None
    if restorable is not None:
        return ()
    return _purge_orphan_optimizer_state(root)


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
    global _INSTALLED, _BASE_MARK_SAMPLING_COMPLETE
    global _BASE_REQUEST_FINAL_PEER_DRAIN, _BASE_RUN_MIXED_ACTOR_JOBS
    if _INSTALLED:
        return

    from v8 import lease_dispatch_continuity_v839 as dispatch
    from v8 import mixed_environment_v859 as mixed
    from v8 import shutdown_semantics_v089 as shutdown

    _BASE_MARK_SAMPLING_COMPLETE = shutdown._mark_sampling_complete
    _BASE_REQUEST_FINAL_PEER_DRAIN = dispatch._request_final_peer_drain
    _BASE_RUN_MIXED_ACTOR_JOBS = mixed.run_mixed_actor_jobs

    shutdown._mark_sampling_complete = _mark_sampling_complete_v863
    dispatch._request_final_peer_drain = _request_final_peer_drain_v863
    mixed.run_mixed_actor_jobs = _run_mixed_actor_jobs_v863
    _INSTALLED = True
