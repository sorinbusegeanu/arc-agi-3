from __future__ import annotations

"""Preserve v8.43 as the public allocation-reporting authority.

v8.49 adds observational action-learning output beneath the existing dispatch
lifecycle wrapper rather than replacing its exact public hook identities.
"""

_INSTALLED = False


def install_action_learning_report_v849_authority_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import action_learning_report_v849 as report
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import lease_dispatch_lifecycle_v843 as v843

    # Preserve the exact historical public chain while inserting v8.49 beneath it:
    # perf writer -> v8.43 dispatch guard -> v8.49 report -> previous writer.
    lower_log = v843._BASE_WRITE_ALLOCATION_LOG
    report._BASE_WRITE_ALLOCATION_LOG = lower_log
    v843._BASE_WRITE_ALLOCATION_LOG = report._write_allocation_log_v849
    perf._write_allocation_log_live = v843._write_allocation_log_v843

    lower_stdout = v843._BASE_ALLOCATION_STDOUT
    report._BASE_ALLOCATION_STDOUT = lower_stdout
    v843._BASE_ALLOCATION_STDOUT = report._allocation_stdout_v849
    perf._allocation_stdout_live = v843._allocation_stdout_v843

    _INSTALLED = True
