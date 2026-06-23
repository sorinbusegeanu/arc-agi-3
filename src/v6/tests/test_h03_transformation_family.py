from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.hypothesis_h03_report import (
    H03_JSON_NAME,
    H03_MD_NAME,
    H03_TXT_NAME,
    evaluate_h03_transformation_family_formation,
)


def _write_v05c_report(run_dir: Path, **validation_overrides) -> None:
    validation = {
        "memory_record_count": 12,
        "carrier_candidate_count": 3,
        "emergent_carrier_count": 0,
        "carrier_spatial_candidate_count": 0,
        "carrier_object_candidate_count": 0,
        "emergent_object_carrier_count": 0,
        "carrier_context_action_fallback_candidate_count": 0,
        "emergent_context_action_fallback_count": 0,
    }
    validation.update(validation_overrides)
    payload = {"validation": validation}
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_contingency_db(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
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
        connection.executemany("INSERT INTO contingencies VALUES (?, ?, ?, ?)", rows)
        connection.commit()


def _write_family_db(path: Path, rows: list[tuple[str, int, int, float]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE transformation_families (
                family_id TEXT,
                member_count INTEGER,
                support_count INTEGER,
                compression_gain REAL
            )
            """
        )
        connection.executemany("INSERT INTO transformation_families VALUES (?, ?, ?, ?)", rows)
        connection.commit()


def test_h03_missing_input_report_is_inconclusive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] == "INCONCLUSIVE"


def test_h03_report_exists_but_no_db_or_artifacts_is_inconclusive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] == "INCONCLUSIVE"


def test_h03_non_singleton_family_compression_can_be_valid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA|0", "eff-1", 3),
            (2, "ctxB|1", "eff-1", 4),
            (3, "ctxC|0", "eff-2", 3),
            (4, "ctxD|1", "eff-2", 2),
            (5, "ctxE|0", "eff-2", 2),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["transformation_family_count"] == 2
    assert result["compression_ratio"] > 1.0
    assert result["compression_gain"] > 0.0
    assert result["decision"] == "VALID"


def test_h03_all_singleton_families_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA|0", "eff-1", 3),
            (2, "ctxB|1", "eff-2", 4),
            (3, "ctxC|0", "eff-3", 2),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] == "INVALID"


def test_h03_object_carriers_already_emergent_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir, emergent_object_carrier_count=1)
    _write_contingency_db(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA|0", "eff-1", 3),
            (2, "ctxB|1", "eff-1", 4),
            (3, "ctxC|0", "eff-2", 2),
            (4, "ctxD|1", "eff-2", 2),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] == "INVALID"


def test_h03_missing_carrier_fields_with_strong_family_evidence_is_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(
        run_dir,
        carrier_object_candidate_count=None,
        emergent_object_carrier_count=None,
        carrier_context_action_fallback_candidate_count=None,
        emergent_context_action_fallback_count=None,
    )
    _write_contingency_db(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA|0", "eff-1", 3),
            (2, "ctxB|1", "eff-1", 4),
            (3, "ctxC|0", "eff-2", 2),
            (4, "ctxD|1", "eff-2", 2),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] == "PARTIALLY_VALID"
    assert any("Pre-object timing unavailable" in item for item in result["missing_evidence"])


def test_h03_family_artifact_json_is_read(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    families = [
        {"family_id": "f1", "member_count": 3, "support_count": 3},
        {"family_id": "f2", "member_count": 2, "support_count": 2},
    ]
    (run_dir / "m2_families.json").write_text(json.dumps({"families": families}), encoding="utf-8")

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["transformation_family_count"] == 2
    assert result["artifact_paths_used"]


def test_h03_direct_compression_gain_column_is_used(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_family_db(
        run_dir / "seed_0.sqlite",
        [
            ("f1", 3, 3, 0.66),
            ("f2", 2, 2, 0.50),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["compression_gain"] is not None
    assert result["mean_family_compression_gain"] is not None
    assert (out_dir / H03_JSON_NAME).exists()
    assert (out_dir / H03_TXT_NAME).exists()
    assert (out_dir / H03_MD_NAME).exists()
