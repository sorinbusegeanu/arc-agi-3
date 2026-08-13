from __future__ import annotations

import json

from v6 import hypothesis_suite_report as suite
from v6 import suite_runtime_profiler as profiler


def test_runtime_profiler_is_installed() -> None:
    assert suite.run_hypothesis_suite_report is profiler._run_profiled
    assert suite._phase is profiler._phase_profiled


def test_profile_from_summary_separates_derive_report_and_unaccounted() -> None:
    summary = {
        "timings": {
            "derive_seconds": 80.0,
            "report_seconds": 15.0,
            "evaluator_total_seconds": 5.0,
            "suite_total_seconds": 100.0,
        },
        "derivation_summary": {
            "timings": {
                "DERIVE.role_transfer_attempts_seconds": 50.0,
                "DERIVE.future_options_seconds": 20.0,
            }
        },
    }
    result = profiler._profile_from_summary(summary, 101.0)
    assert result["derive_reported_seconds"] == 80.0
    assert result["report_reported_seconds"] == 15.0
    assert result["evaluator_reported_seconds"] == 5.0
    assert result["top_level_unaccounted_seconds"] == 5.0
    assert result["report_non_evaluator_seconds"] == 10.0
    assert result["derive_step_seconds"]["DERIVE.role_transfer_attempts_seconds"] == 50.0


def test_profiled_run_writes_runtime_artifact(tmp_path, monkeypatch) -> None:
    def fake_run(**kwargs):
        del kwargs
        return {
            "timings": {
                "derive_seconds": 2.0,
                "report_seconds": 1.0,
                "evaluator_total_seconds": 0.25,
                "suite_total_seconds": 3.5,
            },
            "derivation_summary": {
                "timings": {"DERIVE.future_options_seconds": 1.5}
            },
        }

    monkeypatch.setitem(profiler._ORIGINALS, "run_hypothesis_suite_report", fake_run)
    result = profiler._run_profiled(
        run_dir=tmp_path / "run",
        memory_dir=None,
        output_dir=tmp_path / "reports",
        epoch_id="epoch_0005",
    )
    assert result["timings"]["suite_total_seconds"] == 3.5
    path = tmp_path / "reports" / profiler.PROFILE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["epoch_id"] == "epoch_0005"
    assert payload["reported"]["derive_reported_seconds"] == 2.0
    assert "resource_start" in payload
    assert "resource_end" in payload
