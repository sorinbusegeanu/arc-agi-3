from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.hypothesis_h03_report import (
    H03_JSON_NAME,
    H03_MD_NAME,
    H03_READY_JSON_NAME,
    H03_READY_TXT_NAME,
    H03_TXT_NAME,
    find_h03_ready_runs,
    run_h03_on_best_ready_run,
)


def _write_v05c_report(run_dir: Path, **validation_overrides) -> None:
    validation = {
        "memory_record_count": 12,
        "carrier_object_candidate_count": 0,
        "emergent_object_carrier_count": 0,
        "carrier_context_action_fallback_candidate_count": 0,
        "emergent_context_action_fallback_count": 0,
    }
    validation.update(validation_overrides)
    payload = {"validation": validation}
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_contingency_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE contingencies (
                contingency_id INTEGER PRIMARY KEY,
                context_action TEXT,
                effect_signature TEXT,
                support INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO contingencies VALUES (?, ?, ?, ?)",
            [
                (1, "ctxA|0", "eff-1", 3),
                (2, "ctxB|1", "eff-1", 4),
                (3, "ctxC|0", "eff-2", 2),
            ],
        )
        connection.commit()


def test_find_h03_ready_runs_empty_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    runs_root.mkdir()

    result = find_h03_ready_runs(runs_root, out_dir)

    assert result["candidate_count"] == 0
    assert result["ready_count"] == 0


def test_find_h03_ready_run_missing_carrier_fields_is_not_ready(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    run_dir = runs_root / "run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(
        run_dir,
        carrier_object_candidate_count=None,
        emergent_object_carrier_count=None,
        carrier_context_action_fallback_candidate_count=None,
        emergent_context_action_fallback_count=None,
    )
    _write_contingency_db(run_dir / "seed_0.sqlite")

    result = find_h03_ready_runs(runs_root, out_dir)

    assert result["ready_count"] == 0
    assert result["runs"][0]["missing_required_carrier_fields"]


def test_find_h03_ready_run_with_family_artifact_is_ready(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    run_dir = runs_root / "run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_contingency_db(run_dir / "seed_0.sqlite")
    (run_dir / "m2_families.json").write_text(json.dumps({"families": [{"family_id": "f1", "member_count": 2}]}), encoding="utf-8")

    result = find_h03_ready_runs(runs_root, out_dir)

    assert result["ready_count"] == 1
    assert result["runs"][0]["has_family_artifacts"] is True
    assert result["runs"][0]["h03_ready"] is True


def test_find_h03_ready_run_with_derivable_db_schema_is_ready(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    run_dir = runs_root / "run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_contingency_db(run_dir / "seed_0.sqlite")

    result = find_h03_ready_runs(runs_root, out_dir)

    assert result["ready_count"] == 1
    assert result["runs"][0]["h03_ready"] is True


def test_find_h03_run_best_with_ready_run_creates_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    run_dir = runs_root / "run"
    run_dir.mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_contingency_db(run_dir / "seed_0.sqlite")

    result = run_h03_on_best_ready_run(runs_root, out_dir, max_db_files=1)

    assert result["recommended_run"]["run_dir"] == str(run_dir)
    assert (out_dir / H03_READY_JSON_NAME).exists()
    assert (out_dir / H03_READY_TXT_NAME).exists()
    assert (out_dir / H03_JSON_NAME).exists()
    assert (out_dir / H03_TXT_NAME).exists()
    assert (out_dir / H03_MD_NAME).exists()
