from __future__ import annotations

import json

import pytest

from v6.hypothesis_suite_report import SUITE_PHASE_LOG_NAME, hypothesis_phase


def test_hypothesis_phase_records_failed_event_and_reraises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="phase failure"):
        with hypothesis_phase(
            tmp_path,
            "test_phase",
            epoch_id="epoch_0001",
            total=1,
            unit="phase",
            enabled=False,
            leave=False,
            log_every=1,
        ):
            raise RuntimeError("phase failure")

    events = [json.loads(line) for line in (tmp_path / SUITE_PHASE_LOG_NAME).read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["starting", "failed"]
    assert events[-1]["exception_type"] == "RuntimeError"
    assert events[-1]["exception_message"] == "phase failure"
    assert events[-1]["seconds_elapsed"] >= 0.0
