from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.hypothesis_h03_report import (
    DIRECT_FAMILY_UNAVAILABLE_MESSAGE,
    _has_compact_h03_contingency_evidence,
    _has_compact_h03_family_evidence,
    _usable_h03_family_evidence,
    evaluate_h03_transformation_family_formation,
)
from v6.memory.compact_memory import ensure_memory_layout
from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest


def test_h03_raw_free_compact_helper_accepts_compact_evidence() -> None:
    result = {
        "transformation_family_count": 3,
        "stable_transformation_family_count": 2,
        "family_member_count_total": 10,
        "stable_contingency_count": 5,
    }
    direct_metrics = {"usable_direct_family_evidence": False}
    assert _has_compact_h03_family_evidence(result) is True
    assert _has_compact_h03_contingency_evidence(result) is True
    assert _usable_h03_family_evidence(
        result=result,
        direct_metrics=direct_metrics,
        streamed_compact_only=True,
    ) is True


def test_h03_raw_free_compact_helper_rejects_missing_compact_evidence() -> None:
    result = {
        "transformation_family_count": 0,
        "stable_transformation_family_count": 0,
        "family_member_count_total": 0,
        "stable_contingency_count": 0,
    }
    direct_metrics = {"usable_direct_family_evidence": False}
    assert _usable_h03_family_evidence(
        result=result,
        direct_metrics=direct_metrics,
        streamed_compact_only=True,
    ) is False


def test_h03_raw_direct_evidence_still_works() -> None:
    assert _usable_h03_family_evidence(
        result={},
        direct_metrics={"usable_direct_family_evidence": True},
        streamed_compact_only=False,
    ) is True


def test_h03_direct_streaming_compact_report_uses_compact_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    memory_dir = tmp_path / "memory"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_direct_streaming_fold_manifest(memory_dir)
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        conn.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, context_level, action, effect_signature, support_count, stability_score
            ) VALUES
                (1, 'ctx|a1', 1, 1, 'eff:1', 5, 1.0),
                (2, 'ctx|a2', 1, 2, 'eff:2', 4, 1.0)
            """
        )
        conn.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
            ) VALUES
                (1, 'fam:1', 5, 3, 1, 10, 1.0),
                (2, 'fam:2', 4, 4, 1, 10, 1.0),
                (3, 'fam:3', 3, 3, 1, 10, 1.0)
            """
        )
        conn.commit()
    report = {
        "validation": {
            "memory_record_count": 10,
            "stable_contingency_count": 5,
            "discovered_contingency_count": 5,
            "carrier_candidate_count": 0,
            "emergent_carrier_count": 0,
            "carrier_spatial_candidate_count": 0,
            "carrier_object_candidate_count": 0,
            "emergent_object_carrier_count": 0,
            "carrier_context_action_fallback_candidate_count": 0,
            "emergent_context_action_fallback_count": 0,
        }
    }
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps(report), encoding="utf-8")

    result = evaluate_h03_transformation_family_formation(
        run_dir=run_dir,
        output_dir=output_dir,
        memory_dir=memory_dir,
        scan_all_dbs=True,
    )

    assert result["evidence_source"] == "direct_streaming_manifest_and_compact_memory"
    assert result["direct_streaming_compact_only"] is True
    assert result["raw_epoch_db_available"] is False
    assert result["usable_direct_family_evidence"] is False
    assert result["usable_compact_family_evidence"] is True
    assert result["usable_h03_family_evidence"] is True
    assert DIRECT_FAMILY_UNAVAILABLE_MESSAGE not in result["missing_evidence"]
