from __future__ import annotations

"""v8.66 authority-composition fixes.

Keep v8.50 as the public actor/reporting authority while inserting verified-success
accounting underneath it. This preserves historical wrapper contracts and prevents
verified-success process scope from leaking into unrelated reporter consumers.
"""

_INSTALLED = False


def install_verified_success_metrics_v866_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import learning_effectiveness_report_v850 as v850
    from v8 import reporter
    from v8 import verified_success_metrics_v866 as v866

    # v8.50 remains the public actor wrapper. v8.66 runs directly beneath it so
    # run-scope capture composes with, rather than replaces, the established API.
    historical_actor_base = v850._BASE_RUN_ACTOR_JOBS
    v866._BASE_ACTOR_RUN_JOBS = historical_actor_base
    v850._BASE_RUN_ACTOR_JOBS = v866._run_actor_jobs_v866
    actor_module.run_actor_jobs = v850._run_actor_jobs_v850

    # The reporter follows the same composition rule. Without an active verified
    # success scope v8.66 delegates to the historical formatter and v8.50 keeps its
    # compatibility presentation. With an active scope, v8.66 supplies the verified
    # outcome line through v8.50's existing public formatter.
    historical_progress_base = v850._BASE_PERIODIC_PROGRESS_LINE
    v866._BASE_PERIODIC_PROGRESS = historical_progress_base
    v850._BASE_PERIODIC_PROGRESS_LINE = v866._periodic_progress_line_v866
    reporter.format_periodic_progress_line = v850._periodic_progress_line_v850

    _INSTALLED = True
