from __future__ import annotations

import inspect

from v9.runtime import concurrent_transfer_validation as concurrent


def test_validation_capacity_only_uses_sampling_slots_that_are_released() -> None:
    assert concurrent.released_validation_capacity(
        target_actor_slots=111,
        active_actors=111,
        pending_sampling_jobs=0,
        max_workers=30,
    ) == 0
    assert concurrent.released_validation_capacity(
        target_actor_slots=111,
        active_actors=100,
        pending_sampling_jobs=0,
        max_workers=30,
    ) == 11
    assert concurrent.released_validation_capacity(
        target_actor_slots=111,
        active_actors=70,
        pending_sampling_jobs=0,
        max_workers=30,
    ) == 30
    assert concurrent.released_validation_capacity(
        target_actor_slots=30,
        active_actors=20,
        pending_sampling_jobs=10,
        max_workers=30,
    ) == 0


def test_deadline_reports_all_concrete_unprocessed_validation_work() -> None:
    assert concurrent.deadline_unprocessed(pending_trials=17, active_trials=9) == 26
    source = inspect.getsource(concurrent.ConcurrentTransferValidationSession._expire_deadline)
    assert "unprocessed/wasted=" in source
    assert "transfer_validation_unprocessed_wasted" in source


def test_concurrent_validation_wraps_only_full_training_sampling() -> None:
    assert concurrent._EPOCH_BRANCH.match("epoch-3:bootstrap") is not None
    assert concurrent._EPOCH_BRANCH.match("epoch-3:selected_policy") is not None
    assert concurrent._EPOCH_BRANCH.match("epoch-3:hgt_candidate_eval") is None
    assert concurrent._EPOCH_BRANCH.match("epoch-3:hgt_parent_eval") is None


def test_post_sampling_deadline_starts_after_sampling_completion() -> None:
    source = inspect.getsource(concurrent.ConcurrentTransferValidationSession.mark_sampling_complete)
    assert "time.monotonic() + self.time_budget" in source
    supervisor = inspect.getsource(concurrent.ConcurrentTransferValidationSession._supervise)
    assert "active == 0" in supervisor
    assert "pending == 0" in supervisor
