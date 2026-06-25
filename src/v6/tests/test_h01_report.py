from __future__ import annotations

import json
import sys
from pathlib import Path

from v6.cli import build_parser, main
from v6.hypothesis_h01_report import (
    H01_JSON_NAME,
    H01_MD_NAME,
    H01_READY_JSON_NAME,
    H01_READY_TXT_NAME,
    H01_TXT_NAME,
    evaluate_h01_contingency_emergence,
    find_h01_ready_runs,
)


def _write_v05c_report(
    run_dir: Path,
    *,
    runs: list[dict] | None = None,
    validation_overrides: dict | None = None,
) -> None:
    validation = {
        "memory_record_count": 120,
        "context_contradiction_count": 6,
        "repeated_contradiction_count": 2,
        "context_expansion_suggested_count": 3,
    }
    if validation_overrides:
        validation.update(validation_overrides)
    payload = {
        "runs": runs or [],
        "sampler_comparison": [],
        "best_by_game": [],
        "summary_by_family": [],
        "validation": validation,
    }
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "interaction_sampling_v05c_report.txt").write_text("stub\n", encoding="utf-8")


def _run_row(
    game: str,
    sampler: str,
    *,
    total_interactions: int,
    stable_contingency_count: int,
    prediction_accuracy: float = 0.0,
    context_lift: float | None = None,
    memory_record_count: int = 10,
) -> dict:
    row = {
        "game": game,
        "sampler_name": sampler,
        "run_status": "ok",
        "total_interactions": total_interactions,
        "memory_record_count": memory_record_count,
        "stable_contingency_count": stable_contingency_count,
        "prediction_accuracy": prediction_accuracy,
    }
    if context_lift is not None:
        row["context_lift"] = context_lift
    return row


def test_h01_missing_report_is_inconclusive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()

    result = evaluate_h01_contingency_emergence(run_dir, out_dir)

    assert result["decision"] == "INCONCLUSIVE"
    assert result["input_report_found"] is False


def test_h01_interactions_without_contingencies_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(
        run_dir,
        runs=[
            _run_row("tt01", "random_baseline", total_interactions=100, stable_contingency_count=0, prediction_accuracy=0.2),
            _run_row("pb02", "mixed", total_interactions=110, stable_contingency_count=0, prediction_accuracy=0.3),
        ],
    )

    result = evaluate_h01_contingency_emergence(run_dir, out_dir)

    assert result["decision"] == "INVALID"
    assert result["stable_contingency_count"] == 0


def test_h01_stable_contingencies_across_games_is_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(
        run_dir,
        runs=[
            _run_row("tt01", "random_baseline", total_interactions=100, stable_contingency_count=3, prediction_accuracy=0.7),
            _run_row("pb02", "mixed", total_interactions=110, stable_contingency_count=4, prediction_accuracy=0.8),
        ],
    )

    result = evaluate_h01_contingency_emergence(run_dir, out_dir)

    assert result["decision"] == "VALID"
    assert result["stable_contingency_count"] == 7
    assert result["percentage_games_with_stable_contingency"] == 100.0


def test_h01_sparse_contingencies_is_partially_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(
        run_dir,
        runs=[
            _run_row("tt01", "random_baseline", total_interactions=100, stable_contingency_count=1, prediction_accuracy=0.6),
            _run_row("pb02", "mixed", total_interactions=110, stable_contingency_count=0, prediction_accuracy=0.0),
        ],
    )

    result = evaluate_h01_contingency_emergence(run_dir, out_dir)

    assert result["decision"] == "PARTIALLY_VALID"
    assert "sparse" in result["scientific_conclusion"]


def test_find_h01_ready_runs_finds_h01_contingency_emergence_output(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    run_dir = runs_root / "h01_contingency_emergence"
    run_dir.mkdir(parents=True)
    _write_v05c_report(
        run_dir,
        runs=[_run_row("tt01", "random_baseline", total_interactions=100, stable_contingency_count=2, prediction_accuracy=0.6)],
    )

    result = find_h01_ready_runs(runs_root, out_dir)

    assert result["candidate_count"] == 1
    assert result["ready_count"] == 1
    assert result["recommended_run"]["run_dir"] == str(run_dir)
    assert (out_dir / H01_READY_JSON_NAME).exists()
    assert (out_dir / H01_READY_TXT_NAME).exists()


def test_h01_cli_parser_includes_commands(tmp_path: Path, monkeypatch) -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "hypothesis-h01-report",
            "--run-dir",
            "runs/v6/hypothesis_tests/h01_contingency_emergence",
            "--output-dir",
            "runs/v6/hypothesis_tests/results/h01",
        ]
    )
    assert args.command == "hypothesis-h01-report"

    args = parser.parse_args(
        [
            "find-h01-ready-runs",
            "--runs-root",
            "runs/v6",
            "--output-dir",
            "runs/v6/hypothesis_tests/results/h01_ready_runs",
        ]
    )
    assert args.command == "find-h01-ready-runs"

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(
        run_dir,
        runs=[_run_row("tt01", "random_baseline", total_interactions=100, stable_contingency_count=2, prediction_accuracy=0.6)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v6.cli",
            "hypothesis-h01-report",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(out_dir),
        ],
    )

    assert main() == 0
    assert (out_dir / H01_JSON_NAME).exists()
    assert (out_dir / H01_TXT_NAME).exists()
    assert (out_dir / H01_MD_NAME).exists()
