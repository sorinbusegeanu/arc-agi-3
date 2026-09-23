from __future__ import annotations

import time
from typing import Any

from .bounded_transfer_validation import run_transfer_validation_interval
from .progress import InlineProgress


_status: InlineProgress | None = None
_suppress_nested = False


def _phase(detail: str) -> None:
    global _status
    if _status is None:
        _status = InlineProgress(f"{time.strftime('[%H:%M]')} post-sampling")
    _status.update(detail)


def _finish(detail: str = "complete") -> None:
    global _status
    if _status is not None:
        _status.finish(detail)
        _status = None


def install(epoch_runner_module: Any, runtime_cls: type) -> None:
    """Install compact single-line visibility for long post-sampling work."""
    if getattr(epoch_runner_module, "_post_sampling_progress_installed", False):
        return
    epoch_runner_module._post_sampling_progress_installed = True

    # The epoch helper resolves this module-global at call time.
    epoch_runner_module.run_transfer_validation_interval = (
        run_transfer_validation_interval
    )

    original_train = epoch_runner_module.train_hgt_epoch

    def train_with_progress(*args: Any, **kwargs: Any):
        _phase("HGT training")
        result = original_train(*args, **kwargs)
        _phase(
            f"HGT trained model={getattr(result, 'model_version', 'unknown')}"
        )
        return result

    epoch_runner_module.train_hgt_epoch = train_with_progress

    original_lifecycle = epoch_runner_module.run_lifecycle_maintenance

    def lifecycle_with_progress(*args: Any, **kwargs: Any):
        _phase("lifecycle/compaction")
        result = original_lifecycle(*args, **kwargs)
        _finish("lifecycle complete")
        return result

    epoch_runner_module.run_lifecycle_maintenance = lifecycle_with_progress

    original_wait = runtime_cls.wait_quiescent

    def wait_with_progress(self: Any, *args: Any, **kwargs: Any):
        if not _suppress_nested:
            _phase("quiescent/drain")
        return original_wait(self, *args, **kwargs)

    runtime_cls.wait_quiescent = wait_with_progress

    original_flush = runtime_cls.flush_deferred_memory_updates

    def flush_with_progress(self: Any, *args: Any, **kwargs: Any):
        if not _suppress_nested:
            _phase("memory flush")
        return original_flush(self, *args, **kwargs)

    runtime_cls.flush_deferred_memory_updates = flush_with_progress

    original_replay = runtime_cls.replay_once

    def replay_with_progress(self: Any, *args: Any, **kwargs: Any):
        _phase("replay")
        return original_replay(self, *args, **kwargs)

    runtime_cls.replay_once = replay_with_progress

    original_snapshot = runtime_cls.snapshot

    def snapshot_with_progress(self: Any, *args: Any, **kwargs: Any):
        global _suppress_nested
        _phase("snapshot")
        previous = _suppress_nested
        _suppress_nested = True
        try:
            return original_snapshot(self, *args, **kwargs)
        finally:
            _suppress_nested = previous
            _finish("snapshot complete")

    runtime_cls.snapshot = snapshot_with_progress
