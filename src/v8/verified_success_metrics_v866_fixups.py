from __future__ import annotations

"""v8.66 authority-composition fixes.

Keep v8.50 as the public actor/reporting authority while inserting verified-success
accounting underneath it. This preserves historical wrapper contracts and prevents
verified-success process scope from leaking into unrelated reporter consumers.
"""

_INSTALLED = False
_HISTORICAL_PERIODIC_PROGRESS = None


def _periodic_progress_composed_v866(rows, total_steps, baseline=None) -> str:
    from v8 import verified_success_metrics_v866 as v866

    root = v866._configured_success_root()
    if root is None or not root.exists():
        return _HISTORICAL_PERIODIC_PROGRESS(rows, total_steps, baseline)
    return v866._periodic_progress_line_v866(rows, total_steps, baseline)


def install_verified_success_metrics_v866_fixups() -> None:
    global _INSTALLED, _HISTORICAL_PERIODIC_PROGRESS
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import learning_effectiveness_report_v850 as v850
    from v8 import lease_dispatch_continuity_v839 as v839
    from v8 import reporter
    from v8 import verified_success_metrics_v866 as v866

    # Preserve the established public chain exactly: v850 -> v839 -> historical.
    # Insert v8.66 beneath v839 so v850._BASE_RUN_ACTOR_JOBS remains v839, as
    # required by the runtime authority contract.
    historical_actor_base = v839._BASE_RUN_ACTOR_JOBS
    v866._BASE_ACTOR_RUN_JOBS = historical_actor_base
    v839._BASE_RUN_ACTOR_JOBS = v866._run_actor_jobs_v866
    actor_module.run_actor_jobs = v850._run_actor_jobs_v850

    # v8.50 remains the public reporter formatter. v8.66 supplies verified outcome
    # metrics only while a live verified-success root exists. A stale environment
    # variable from a completed/deleted run must not change unrelated reporters.
    _HISTORICAL_PERIODIC_PROGRESS = v850._BASE_PERIODIC_PROGRESS_LINE
    v866._BASE_PERIODIC_PROGRESS = _HISTORICAL_PERIODIC_PROGRESS
    v850._BASE_PERIODIC_PROGRESS_LINE = _periodic_progress_composed_v866
    reporter.format_periodic_progress_line = v850._periodic_progress_line_v850

    _INSTALLED = True
