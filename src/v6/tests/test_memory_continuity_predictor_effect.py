from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.main import V6Config, V6System
from v6.memory.compact_memory import ensure_memory_layout, stable_family_int_id


class _Env:
    def __init__(self) -> None:
        import numpy as np

        self.grid = np.zeros((2, 2), dtype=int)

    def observe(self):
        return self.grid.copy()

    def step(self, action: int):
        return self.grid.copy()

    def available_actions(self) -> list[int]:
        return [1]


def test_compact_memory_changes_predictor_before_sampling(tmp_path: Path) -> None:
    memory = ensure_memory_layout(tmp_path / "memory")
    family_signature = "centroid:[2,1,0,0,0]"
    family_id = stable_family_int_id(family_signature)
    with sqlite3.connect(memory.current_state) as connection:
        connection.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority, representative_example_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, '["ctx",1]|a1|ecentroid:[2,1,0,0,0]', "tt01", "mixed", 2, 1, family_signature, 30, 1, 10, 1.0, 0.0, 0.0, 1),
        )
        connection.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (family_id, family_signature, family_signature, "unknown", "unknown", "unknown", 30, 3, 1, 10, 1.0),
        )
        connection.execute("INSERT INTO family_identity_map (canonical_signature, stable_family_id) VALUES (?, ?)", (family_signature, family_id))
        connection.commit()
    memory.summary_json.write_text(json.dumps({"fold_summary": {"stable_contingencies_added": 1}}, indent=2), encoding="utf-8")

    system = V6System(env=_Env(), config=V6Config(database_path=":memory:", memory_input_dir=str(memory.root), restore_compact_memory=True))
    try:
        prediction = system.predictor.predict_multi_scale({2: ("ctx", 1)}, 1)
        assert prediction == family_id
    finally:
        system.close()
