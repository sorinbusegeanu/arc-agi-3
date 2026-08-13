from __future__ import annotations

import json

from v6 import suite_runtime_profiler as profiler


def test_runtime_profiler_is_installed() -> None:
    assert profiler._INSTALLED is True
    assert "_write_suite_summary" in profiler._ORIGINALS
    assert "validate_hypothesis_provenance" in profiler._ORIGINALS


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
    result = profiler._profile_from_summary(summary)
    assert result["derive_reported_seconds"] == 80.0
    assert result["report_reported_seconds"] == 15.0
    assert result["evaluator_reported_seconds"] == 5.0
    assert result["top_level_unaccounted_seconds"] == 5.0
    assert result["report_non_evaluator_seconds"] == 10.0
    assert result["derive_step_seconds"]["DERIVE.role_transfer_attempts_seconds"] == 50.0


def test_profiled_summary_write_creates_runtime_artifact(tmp_path, monkeypatch) -> None:
    def fake_write(summary, output_dir, *, hypothesis_results=None):
        del summary, hypothesis_results
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(profiler._ORIGINALS, "_write_suite_summary", fake_write)
    profiler._reset_state()
    profiler._add_timing("derive.schema_migration", 0.25)
    summary = {
        "epoch_id": "epoch_0005",
        "memory_dir": None,
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
    output_dir = tmp_path / "reports"
    profiler._write_suite_summary_profiled(summary, output_dir, hypothesis_results={})
    payload = json.loads((output_dir / profiler.PROFILE_NAME).read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["epoch_id"] == "epoch_0005"
    assert payload["reported"]["derive_reported_seconds"] == 2.0
    assert payload["instrumented_timings"]["derive.schema_migration"] == 0.25
    assert "resource_end" in payload
