from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.main import V6Config, V6System
from v6.memory.compact_memory import ensure_memory_layout, stable_family_int_id
from v6.memory.compact_memory_restore import load_compact_memory_into_system


class _ToggleEnv:
    def __init__(self) -> None:
        import numpy as np

        self.grid = np.zeros((3, 3), dtype=int)

    def observe(self):
        return self.grid.copy()

    def step(self, action: int):
        self.grid[1, 1] = int(action)
        return self.grid.copy()

    def available_actions(self) -> list[int]:
        return [0, 1]


def _seed_compact_memory(memory_dir: Path) -> str:
    paths = ensure_memory_layout(memory_dir)
    family_signature = 'centroid:[2,1,0,0,0]'
    effect_signature = family_signature
    canonical_key = '["ctx",1]|a0|e' + effect_signature
    stable_id = 424242
    with sqlite3.connect(paths.current_state) as connection:
        connection.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                mean_replay_priority, representative_example_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, canonical_key, "tt01", "mixed", 2, 0, effect_signature, 25, 1, 10, 1.0, 0.1, 0.9, 1),
        )
        connection.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (stable_id, family_signature, family_signature, "unknown", "unknown", "unknown", 25, 5, 1, 10, 5.0),
        )
        connection.execute("INSERT INTO family_identity_map (canonical_signature, stable_family_id) VALUES (?, ?)", (family_signature, stable_id))
        connection.execute(
            """
            INSERT INTO family_members (
                family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (family_signature, canonical_key, 5, 1, 10),
        )
        connection.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, stability_score, is_emergent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("carrier-a", "carrier-a", "object", 4, 2, 2, 10, 0.5, 1),
        )
        connection.execute("INSERT INTO memory_summary (key, value_json) VALUES (?, ?)", ("total_interactions_seen", json.dumps(25)))
        connection.commit()
    with sqlite3.connect(paths.replay_queue) as connection:
        connection.execute(
            """
            INSERT INTO replay_queue (
                replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step, last_seen_global_step, compact_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("7", "interaction", "7", 0.95, "contradiction_linked", 5, 10, json.dumps({"context_signature": '["ctx",1]', "family_id": stable_id})),
        )
        connection.commit()
    with sqlite3.connect(paths.graph) as connection:
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("family:" + family_signature, "family", family_signature, 1, 10, 5))
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("carrier:carrier-a", "carrier", "carrier-a", 2, 10, 4))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("carrier:carrier-a->explains->family:" + family_signature, "carrier:carrier-a", "family:" + family_signature, "explains", 2, 10, 4, 1.0))
        connection.commit()
    paths.summary_json.write_text(json.dumps({"fold_summary": {"stable_contingencies_added": 1}, "total_interactions_seen": 25}, indent=2), encoding="utf-8")
    return family_signature


def test_epoch_2_restores_compact_memory_into_live_system(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    family_signature = _seed_compact_memory(memory_dir)
    system = V6System(
        env=_ToggleEnv(),
        config=V6Config(database_path=":memory:", memory_input_dir=str(memory_dir), restore_compact_memory=True),
    )
    try:
        summary = system.compact_memory_restore_summary
        assert summary["stable_contingencies_restored"] == 1
        assert summary["transformation_families_restored"] == 1
        assert summary["carrier_candidates_restored"] == 1
        assert summary["replay_candidates_restored"] == 1
        assert summary["graph_nodes_restored"] >= 2
        predicted = system.predictor.predict_multi_scale({2: ("ctx", 1)}, 0)
        assert predicted == 424242
        stable = system.contingency_learner.stable_contingencies()[0]
        assert stable.context_level == 2
        assert system.contingency_learner.best_stable_for_action({2: ("ctx", 1)}, 0) is not None
        assert system.memory_lifecycle.replay_candidates
        assert system.carrier_tracker.build_candidates()
        assert system.graph.count_edges_of_type("explains") >= 1
    finally:
        system.close()


def test_restore_function_populates_live_objects(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_compact_memory(memory_dir)
    system = V6System(env=_ToggleEnv(), config=V6Config(database_path=":memory:"))
    try:
        summary = load_compact_memory_into_system(system, memory_dir)
        assert summary["memory_summary_loaded"] is True
        assert len(system.contingency_learner.stable_contingencies()) == 1
        assert system.contingency_learner.stable_contingencies()[0].context_level == 2
        assert len(system.clusterer.families) == 1
        family_signature = next(iter(system.clusterer.semantic_family_members))
        assert system.clusterer.semantic_family_members[family_signature]
        restored_id = next(iter(system.clusterer.families))
        assert restored_id == 424242
        assert len(system.memory_lifecycle.replay_candidates) == 1
    finally:
        system.close()
