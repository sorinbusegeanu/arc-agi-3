from __future__ import annotations

"""v8.77 generic worker result delivery integrity.

A multiprocessing.Queue uses a feeder thread. A generic worker can finish its Python
body after ``put()`` while the queue feeder is still flushing the result to the
parent. The parent may then observe every worker as exited before the final result
is readable and incorrectly report a missing actor result.

This late layer keeps the v8.75 scheduling authority unchanged and adds one child-
side delivery barrier: close the child's result-queue handle and join its feeder
thread before the process exits.
"""


_INSTALLED = False
_BASE_GENERIC_PROCESS_WORKER = None


def _generic_process_worker_v877(**kwargs) -> None:
    result_queue = kwargs.get("result_queue")
    try:
        return _BASE_GENERIC_PROCESS_WORKER(**kwargs)
    finally:
        if result_queue is None:
            return
        try:
            result_queue.close()
        except (AttributeError, ValueError, OSError):
            pass
        try:
            result_queue.join_thread()
        except (AttributeError, RuntimeError, ValueError, OSError):
            pass


def install_generic_result_flush_v877() -> None:
    global _INSTALLED, _BASE_GENERIC_PROCESS_WORKER
    if _INSTALLED:
        return

    from v8 import mixed_research_runtime_integrity_v875 as v875

    _BASE_GENERIC_PROCESS_WORKER = v875._generic_process_worker_v875
    v875._generic_process_worker_v875 = _generic_process_worker_v877
    _INSTALLED = True
