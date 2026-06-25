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


def _write_contingency_db_with_sampler_columns(
    path: Path,
    rows: list[tuple[int, str, int, str, str, str]],
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE contingencies (
                contingency_id INTEGER PRIMARY KEY,
                context_signature TEXT,
                action INTEGER,
                effect_signature TEXT,
                game TEXT,
                sampler_name TEXT
            )
            """
        )
        connection.executemany("INSERT INTO contingencies VALUES (?, ?, ?, ?, ?, ?)", rows)
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


def test_h03_global_merge_merges_same_effect_signature_across_dbs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db(run_dir / "seed_0.sqlite", [(1, "ctxA|0", "eff-1", 3), (2, "ctxB|1", "eff-1", 3)])
    _write_contingency_db(run_dir / "seed_1.sqlite", [(1, "ctxC|0", "eff-1", 3), (2, "ctxD|1", "eff-1", 3)])

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["global_family_count_before_merge"] == 2
    assert result["global_family_count_after_merge"] == 1
    assert result["transformation_family_count"] == 1
    assert result["families_merged_across_shards"] == 1


def test_h03_integer_family_ids_do_not_merge_when_semantics_differ(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    with sqlite3.connect(run_dir / "seed_0.sqlite") as connection:
        connection.execute(
            "CREATE TABLE transformation_families (family_id INTEGER, effect_signature TEXT, member_count INTEGER, support_count INTEGER)"
        )
        connection.executemany("INSERT INTO transformation_families VALUES (?, ?, ?, ?)", [(1, "eff-a", 2, 2)])
        connection.commit()
    with sqlite3.connect(run_dir / "seed_1.sqlite") as connection:
        connection.execute(
            "CREATE TABLE transformation_families (family_id INTEGER, effect_signature TEXT, member_count INTEGER, support_count INTEGER)"
        )
        connection.executemany("INSERT INTO transformation_families VALUES (?, ?, ?, ?)", [(1, "eff-b", 2, 2)])
        connection.commit()

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["transformation_family_count"] == 2
    assert result["families_merged_across_shards"] == 0


def test_h03_same_effect_signature_across_games_counts_cross_game(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    (run_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_1").mkdir(parents=True)
    (run_dir / "sampling_v05c" / "pb02" / "mixed" / "steps_1").mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_contingency_db((run_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_1" / "seed_0.sqlite"), [(1, "ctxA|0", "eff-1", 3)])
    _write_contingency_db((run_dir / "sampling_v05c" / "pb02" / "mixed" / "steps_1" / "seed_1.sqlite"), [(1, "ctxB|1", "eff-1", 3)])

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["family_cross_game_count"] == 1
    assert result["families_merged_across_games"] == 1


def test_h03_same_effect_signature_across_samplers_counts_cross_sampler(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    (run_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_1").mkdir(parents=True)
    (run_dir / "sampling_v05c" / "tt01" / "low_confidence" / "steps_1").mkdir(parents=True)
    _write_v05c_report(run_dir)
    _write_contingency_db((run_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_1" / "seed_0.sqlite"), [(1, "ctxA|0", "eff-1", 3)])
    _write_contingency_db((run_dir / "sampling_v05c" / "tt01" / "low_confidence" / "steps_1" / "seed_1.sqlite"), [(1, "ctxB|1", "eff-1", 3)])

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["family_cross_sampler_count"] == 1
    assert result["families_merged_across_samplers"] == 1


def test_h03_derived_families_populate_cross_context_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 0, "eff-1", "tt01", "mixed"),
            (2, "ctxB", 1, "eff-1", "tt01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["family_cross_context_count"] == 1


def test_h03_max_rows_reduces_row_count_used(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db(
        run_dir / "seed_0.sqlite",
        [(index, f"ctx{index}", f"eff-{index % 2}", 3) for index in range(1, 11)],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir, max_rows=3)

    assert result["row_count_used"] == 3
    assert result["max_rows_applied"] is True


def test_h03_singleton_ratio_drops_after_global_merge(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db(run_dir / "seed_0.sqlite", [(1, "ctxA|0", "eff-1", 3)])
    _write_contingency_db(run_dir / "seed_1.sqlite", [(1, "ctxB|1", "eff-1", 3)])

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["global_family_count_before_merge"] == 2
    assert result["transformation_family_count"] == 1
    assert result["singleton_family_ratio"] == 0.0


def test_h03_singleton_diagnostics_include_breakdowns_and_relaxed_ratio(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 0, "eff-1", "tt01", "mixed"),
            (2, "ctxB", 1, "eff-2", "tt01", "mixed"),
            (3, "ctxC", 1, "eff-17", "tt01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["top_singleton_family_signatures"]
    assert result["singleton_families_by_game"]["tt01"] >= 1
    assert result["singleton_families_by_sampler"]["mixed"] >= 1
    assert result["singleton_families_by_action"]["1"] >= 1
    assert result["singleton_families_by_effect_type"]
    assert result["singleton_ratio_strict"] is not None
    assert result["singleton_ratio_relaxed"] is not None


def test_h03_safe_relaxed_merge_reduces_singleton_ratio(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 3, "[127.83,-7.99,-0.008,0,0]", "gr01", "mixed"),
            (2, "ctxB", 3, "[127.84,-7.99,-0.009,0,0]", "gr01", "mixed"),
            (3, "ctxC", 3, "[127.85,-7.99,-0.010,0,0]", "gr01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["singleton_ratio_relaxed"] < result["singleton_ratio_strict"]


def test_h03_unsafe_relaxed_merge_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 3, "[127.83,-7.99,-0.008,0,0]", "gr01", "mixed"),
            (2, "ctxB", 5, "[127.84,-7.99,-0.009,0,0]", "gr01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["merge_safety_passed"] is False
    assert result["unsafe_relaxed_merge_count"] >= 1


def test_h03_mixed_effect_types_increment_unsafe_relaxed_merge_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 3, "[127.83,-7.99,-0.008,0,0]", "gr01", "mixed"),
            (2, "ctxB", 3, "[127.84,7.99,0.009,0,0]", "gr01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["unsafe_relaxed_merge_count"] >= 1


def test_h03_official_decision_remains_strict(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 3, "[127.83,-7.99,-0.008,0,0]", "gr01", "mixed"),
            (2, "ctxB", 3, "[127.84,-7.99,-0.009,0,0]", "gr01", "mixed"),
            (3, "ctxC", 3, "[127.85,-7.99,-0.010,0,0]", "gr01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] in {"PARTIALLY_VALID", "INVALID", "INCONCLUSIVE"}


def test_h03_relaxed_decision_candidate_can_be_valid_separately(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_v05c_report(run_dir)
    _write_contingency_db_with_sampler_columns(
        run_dir / "seed_0.sqlite",
        [
            (1, "ctxA", 3, "[64.0,0.0,0.0,1.0,0.0]", "fs02", "mixed"),
            (2, "ctxB", 3, "[64.0,0.0,0.0,1.0,0.0]", "fs02", "mixed"),
            (3, "ctxA", 3, "[127.83,-7.99,-0.008,0,0]", "gr01", "mixed"),
            (4, "ctxB", 3, "[127.84,-7.99,-0.009,0,0]", "gr01", "mixed"),
            (5, "ctxC", 3, "[127.85,-7.99,-0.010,0,0]", "gr01", "mixed"),
            (6, "ctxD", 3, "[127.86,-7.99,-0.011,0,0]", "gr01", "mixed"),
        ],
    )

    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)

    assert result["decision"] == "PARTIALLY_VALID"
    assert result["relaxed_decision_candidate"] == "VALID"
