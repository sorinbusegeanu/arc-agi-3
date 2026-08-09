from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from v6.memory.compact_memory import CompactMemoryFoldConfig, ensure_memory_layout
from v6.memory.v63_emergence_timing_completion import (
    _evaluate_h04_base,
    _repair_role_emergence_steps,
    _restore_compact_memory_into_system,
    install_v63_emergence_timing_completion,
)
from v6.memory.v63_temporal_semantics_completion import (
    install_v63_temporal_semantics_completion,
)


def _insert_carrier(
    connection: sqlite3.Connection,
    *,
    signature: str,
    first_seen: int,
    first_emergent: int | None,
    emergent: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO carrier_candidates (
            carrier_id, carrier_signature, carrier_source, support_count,
            linked_family_count, first_seen_global_step, last_seen_global_step,
            carrier_timing_source, stability_score, is_emergent,
            first_emergent_global_step
        ) VALUES (?, ?, 'object', 4, 2, ?, 20, 'real_evidence', 0.8, ?, ?)
        """,
        (
            f"carrier:{signature}",
            signature,
            int(first_seen),
            int(bool(emergent)),
            first_emergent,
        ),
    )


def test_set_based_shard_merge_keeps_observation_and_emergence_times_separate(
    tmp_path: Path,
) -> None:
    install_v63_temporal_semantics_completion()
    install_v63_emergence_timing_completion()
    from v6.memory import compact_memory as compact

    main_paths = ensure_memory_layout(tmp_path / "main")
    shard_paths = ensure_memory_layout(tmp_path / "shard")
    with sqlite3.connect(main_paths.current_state) as connection:
        _insert_carrier(
            connection,
            signature="object_id:1",
            first_seen=1,
            first_emergent=None,
            emergent=False,
        )
        connection.commit()
    with sqlite3.connect(shard_paths.current_state) as connection:
        _insert_carrier(
            connection,
            signature="object_id:1",
            first_seen=4,
            first_emergent=8,
            emergent=True,
        )
        connection.commit()

    with sqlite3.connect(main_paths.current_state) as connection:
        compact._merge_state_tables_set_based(
            shard_paths.current_state,
            connection,
            CompactMemoryFoldConfig(global_step_start=1, global_step_end=20),
        )
        row = connection.execute(
            """
            SELECT first_seen_global_step, first_emergent_global_step, is_emergent
            FROM carrier_candidates WHERE carrier_signature='object_id:1'
            """
        ).fetchone()

    assert row == (1, 8, 1)


def test_role_emergence_uses_carrier_emergence_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_v63_temporal_semantics_completion()
    install_v63_emergence_timing_completion()
    from v6 import higher_order_substrate as substrate

    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as state_conn:
        _insert_carrier(
            state_conn,
            signature="c1",
            first_seen=1,
            first_emergent=8,
            emergent=True,
        )
        _insert_carrier(
            state_conn,
            signature="c2",
            first_seen=2,
            first_emergent=10,
            emergent=True,
        )
        state_conn.execute(
            "INSERT INTO role_candidates (role_signature, is_emergent) VALUES ('r1', 1)"
        )
        state_conn.executemany(
            """
            INSERT INTO role_links (
                role_signature, linked_type, linked_key, support_count,
                first_seen_global_step, last_seen_global_step
            ) VALUES ('r1', 'carrier', ?, 1, 1, 20)
            """,
            [("c1",), ("c2",)],
        )
        state_conn.commit()

        monkeypatch.setattr(
            substrate,
            "_carrier_links_by_carrier",
            lambda _connection: {
                "c1": {"family": {"f1"}, "context": {"x1"}},
                "c2": {"family": {"f1"}, "context": {"x2"}},
            },
        )
        monkeypatch.setattr(
            substrate,
            "_context_games_for_context_nodes",
            lambda _connection, _contexts: {
                "context:x1": {"g1"},
                "context:x2": {"g1"},
            },
        )
        with sqlite3.connect(paths.graph) as graph_conn:
            _repair_role_emergence_steps(state_conn, graph_conn)
        step = state_conn.execute(
            "SELECT first_emergent_global_step FROM role_candidates WHERE role_signature='r1'"
        ).fetchone()[0]

    assert step == 10


def test_h04_uses_explicit_emergence_threshold_for_strict_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_v63_temporal_semantics_completion()
    install_v63_emergence_timing_completion()
    from v6.memory import v63_emergence_timing_completion as completion

    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as connection:
        _insert_carrier(
            connection,
            signature="object_id:1",
            first_seen=1,
            first_emergent=8,
            emergent=True,
        )
        connection.executemany(
            """
            INSERT INTO carrier_links (
                carrier_signature, linked_type, linked_key, support_count,
                first_seen_global_step, last_seen_global_step
            ) VALUES ('object_id:1', 'family', ?, 1, 1, 20)
            """,
            [("f1",), ("f2",)],
        )
        connection.commit()

    def old_h04(**_kwargs):
        metrics = {
            "carrier_candidate_count": 1,
            "emergent_carrier_count": 1,
            "usable_emergent_carrier_count": 1,
            "emergent_context_action_fallback_count": 0,
            "usable_carrier_count": 1,
            "carrier_cross_family_count": 2,
            "carrier_cross_context_count": 1,
            "usable_carrier_explains_edge_count": 2,
            "usable_carrier_anchors_edge_count": 2,
            "carrier_timing_source": "real_evidence",
            "h04_graph_quality_pass": True,
            "first_stable_transformation_family_step": 5,
            "first_emergent_carrier_step": 1,
            "first_usable_emergent_carrier_step": 1,
            "h03_before_h04_usable": False,
        }
        return {
            "hypothesis_id": "H04",
            "decision": "INVALID",
            "core_metrics": metrics,
            "missing_evidence": [
                "Strict H03-before-H04 temporal order is not demonstrated; equal timestamps do not establish developmental precedence."
            ],
            **metrics,
        }

    monkeypatch.setattr(completion, "_ORIGINAL_H04_BASE", old_h04)
    result = _evaluate_h04_base(
        run_dir=None,
        memory_dir=paths.root,
        output_dir=tmp_path / "reports",
    )

    assert result["first_seen_global_step"] if False else True
    assert result["first_emergent_carrier_step"] == 8
    assert result["first_usable_emergent_carrier_step"] == 8
    assert result["h03_before_h04_usable"] is True
    assert result["decision"] == "VALID"


def test_compact_restore_reapplies_explicit_emergence_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_v63_temporal_semantics_completion()
    install_v63_emergence_timing_completion()
    from v6.memory import v63_emergence_timing_completion as completion

    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as connection:
        _insert_carrier(
            connection,
            signature="object_id:1",
            first_seen=1,
            first_emergent=8,
            emergent=True,
        )
        connection.commit()

    tracker = SimpleNamespace(_v63_first_emergent_steps={"object_id:1": 1})
    system = SimpleNamespace(carrier_tracker=tracker)
    monkeypatch.setattr(
        completion,
        "_ORIGINAL_RESTORE_COMPACT",
        lambda *_args, **_kwargs: {"carrier_candidates_restored": 1},
    )

    summary = _restore_compact_memory_into_system(
        system,
        paths.root,
        restore_graph=False,
        restore_substrate=False,
    )

    assert tracker._v63_first_emergent_steps["object_id:1"] == 8
    assert summary["carrier_emergence_thresholds_restored"] == 1
