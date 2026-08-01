from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from v6.hypothesis_suite_report import SUITE_PHASE_LOG_NAME, hypothesis_phase
from v6.hypothesis_h06_report import evaluate_h06_role_transfer
from v6.memory.compact_memory import ensure_memory_layout


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


def _seed_h06_diagnostics(memory_dir: Path, *, pair_count: int = 5) -> None:
    paths = ensure_memory_layout(memory_dir)
    reasons = ("success", "role_mismatch", "low_similarity", "insufficient_source_support", "no_source_profile")
    with sqlite3.connect(paths.current_state) as conn:
        for role, role_type in (("role-a", "mover"), ("role-b", "blocker")):
            conn.execute(
                "INSERT INTO role_candidates (role_signature, role_type) VALUES (?, ?)",
                (role, role_type),
            )
        for index in range(pair_count):
            reason = reasons[index % len(reasons)]
            success = int(reason == "success")
            kind = "cross_context" if index % 2 else "cross_game"
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                    target_scope_type, target_scope_key, similarity_score, transfer_score, reuse_success,
                    failure_reason, best_margin, source_carrier_count, candidate_role_count
                ) VALUES (?, ?, ?, 'game', ?, 'game', ?, ?, 0.5, ?, ?, ?, ?, ?)
                """,
                (
                    f"attempt-{index:03d}", "role-a" if index % 2 == 0 else "role-b", kind,
                    f"g{index:03d}", f"t{index:03d}", 0.85 if success else 0.25,
                    success, reason, 0.25 if success else -0.05, 8 if success else 1, 4 if success else 1,
                ),
            )
        conn.execute(
            "INSERT INTO memory_summary (key, value_json) VALUES ('higher_order_transfer_summary', ?)",
            (json.dumps({"total_possible_transfer_attempts": pair_count * 2, "sampled_transfer_attempts": pair_count, "skipped_by_cap_count": pair_count, "max_attempts_per_role": 3, "max_attempts_per_target_scope": 1}),),
        )
        conn.commit()


def test_h06_reports_transfer_diagnosis_and_failed_validity_gates(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_h06_diagnostics(memory_dir)
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06", already_derived=True)

    assert result["cross_game_attempt_count"] == 3
    assert result["cross_context_attempt_count"] == 2
    assert result["role_mismatch_count"] + result["low_similarity_count"] + result["insufficient_source_support_count"] + result["no_source_profile_count"] + result["other_failure_count"] + result["successful_transfer_count"] == result["transfer_attempt_count"]
    assert result["transfer_by_source_game"] == sorted(result["transfer_by_source_game"], key=lambda row: (-row["attempt_count"], row["source_game"]))
    assert result["transfer_by_role"][0]["role_type"] in {"mover", "blocker"}
    assert len(result["similarity_score_buckets"]) == 4
    assert len(result["source_carrier_count_buckets"]) == 4
    assert result["sampling_fraction"] == 0.5
    assert result["h06_validity_gates"]["min_transfer_success_rate"]["passed"] is False
    assert result["missing_evidence"]
    assert (tmp_path / "h06" / "h06_transfer_by_game_pair.jsonl").exists()


def test_h06_bounds_main_game_pair_rows_and_writes_complete_jsonl(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_h06_diagnostics(memory_dir, pair_count=105)
    output_dir = tmp_path / "h06"
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=output_dir, already_derived=True)

    assert len(result["transfer_by_game_pair"]) == 100
    rows = (output_dir / "h06_transfer_by_game_pair.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 105
