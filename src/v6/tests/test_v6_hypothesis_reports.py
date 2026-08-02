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
            source_game = f"g{index:03d}"
            target_game = f"t{index:03d}" if kind == "cross_game" else source_game
            source_context = f"ctx_s{index:03d}" if kind == "cross_context" else None
            target_context = f"ctx_t{index:03d}" if kind == "cross_context" else None
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                    target_scope_type, target_scope_key, source_game_key, target_game_key,
                    source_context_key, target_context_key, source_carrier_signature, source_role_signature,
                    predicted_target_role_signature, observed_target_role_signature, provenance_mode,
                    similarity_score, transfer_score, reuse_success,
                    failure_reason, best_margin, source_carrier_count, source_evidence_support_count,
                    support_gate_passed, similarity_gate_passed, role_match_gate_passed, candidate_role_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'single_source', ?, 0.5, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"attempt-{index:03d}", "role-a" if index % 2 == 0 else "role-b", kind,
                    "game" if kind == "cross_game" else "context",
                    source_game if kind == "cross_game" else source_context,
                    "game" if kind == "cross_game" else "context",
                    target_game if kind == "cross_game" else target_context,
                    source_game, target_game, source_context, target_context,
                    f"carrier-source-{index}", "role-a" if index % 2 == 0 else "role-b",
                    "role-a" if index % 2 == 0 else "role-b", "role-a" if index % 2 == 0 else "role-b",
                    0.85 if success else 0.25,
                    success, reason, 0.25 if success else -0.05, 1, 8 if success else 1,
                    int(success), int((0.85 if success else 0.25) >= 0.60), int(success), 4 if success else 1,
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


def test_h06_excludes_legacy_and_invalid_provenance_from_verified_rates(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_h06_diagnostics(memory_dir, pair_count=2)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                target_scope_type, target_scope_key, reuse_success, failure_reason
            ) VALUES ('legacy-row', 'role-a', 'cross_game', 'not_game', 'target', 'game', 'target', 1, 'success')
            """
        )
        conn.execute(
            """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, transfer_kind, source_game_key, target_game_key,
                    source_carrier_signature, source_role_signature, provenance_mode, reuse_success, failure_reason
                ) VALUES ('invalid-row', 'role-a', 'cross_game', 'same', 'same', 'carrier-a', 'role-a', 'single_source', 1, 'success')
            """
        )
        conn.commit()
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06", already_derived=True)
    assert result["legacy_transfer_provenance_count"] == 1
    assert result["same_game_marked_cross_game_count"] == 1
    assert result["invalid_provenance_attempt_count"] == 1
    assert result["recorded_transfer_attempt_count"] == result["transfer_attempt_count"] + 2
    assert "invalid_transfer_provenance" in result["consistency_warnings"]
    assert all(row["source_game"] != "unknown" for row in result["transfer_by_source_game"])


def test_h06_separates_verified_and_multi_source_transfer_rates(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_h06_diagnostics(memory_dir, pair_count=2)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                target_scope_type, target_scope_key, source_game_key, target_game_key,
                source_role_signature, source_carrier_signatures_json, source_game_keys_json,
                provenance_mode, provenance_status, reuse_success, failure_reason,
                source_carrier_count, source_evidence_support_count
            ) VALUES (
                'multi-source-row', 'role-a', 'cross_game', 'game', 'g_multi', 'game', 't_multi',
                'g_multi', 't_multi', 'role-a', '["carrier-a", "carrier-b"]', '["g_multi", "g_other"]',
                'multi_source', 'aggregate_source', 1, 'success', 2, 4
            )
            """
        )
        conn.execute(
            """
            INSERT INTO memory_summary (key, value_json) VALUES ('higher_order_transfer_summary', ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (json.dumps({
                "candidate_transfer_attempt_count": 10,
                "sampled_profile_attempt_count": 5,
                "expanded_transfer_attempt_count": 6,
                "persisted_transfer_attempt_count": 6,
            }),),
        )
        conn.commit()

    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06", already_derived=True)

    assert result["recorded_transfer_attempt_count"] == 3
    assert result["verified_single_source_attempt_count"] == 2
    assert result["multi_source_attempt_count"] == 1
    assert result["multi_source_success_rate"] == 1.0
    assert result["aggregate_transfer_attempt_count"] == 3
    assert result["sampling_fraction"] == 0.5
    assert all(row["source_game"] != "g_multi" for row in result["transfer_by_source_game"])
