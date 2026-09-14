from __future__ import annotations

from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
from v9.runtime.memory_pipeline_v2 import build_commit_plan
from v9.runtime.multiprocess import EncodedTransition


def _transition() -> EncodedTransition:
    return EncodedTransition(
        actor_id=1,
        producer_sequence=1,
        global_step=0,
        environment_identity=("synthetic", "semantic-test", "default", "one"),
        episode_id=1,
        observation_schema_id=11,
        before_signature=101,
        action_id=2,
        after_signature=102,
        available_actions_after=2,
        primary_valence=1,
        symbols=(),
        curriculum_step=None,
        game_scenario="semantic-test",
        action_schema_id=12,
        available_action_set_signature=13,
        boundary_scope="EPISODE",
        task_success=True,
        semantic_before=((2, 1001, 6, 3, 3.0),),
        semantic_action=((8, 2001, 23, 2, 1.0),),
        semantic_options=((8, 2001, 23, 2, 1.0),),
        semantic_after=((2, 1001, 6, 4, 4.0),),
        semantic_delta=((9, 1001, 13, 4, 4.0),),
    )


def test_semantics_survive_into_m0_m1_payloads() -> None:
    prepared = prepare_ingestion(IngestionTask(1, 1, _transition()))
    plan = build_commit_plan(prepared)
    m0_payload = plan.base_writes[0].payload
    m1_payload = plan.base_writes[1].payload
    normalized = plan.normalized_write.payload
    assert m0_payload["semantic_before"] == [[2, 1001, 6, 3, 3.0]]
    assert m0_payload["semantic_after"] == [[2, 1001, 6, 4, 4.0]]
    assert m1_payload["semantic_action"] == [[8, 2001, 23, 2, 1.0]]
    assert normalized["semantic_effects"] == [[9, 1001, 13, 4, 4.0]]
    assert normalized["semantic_after"] == [[2, 1001, 6, 4, 4.0]]


def test_hgt_semantic_row_reader_includes_after_state() -> None:
    from v9.hgt.training import _semantic_rows
    payload = {
        "semantic_before": [[2, 1, 6, 3, 3.0]],
        "semantic_action": [[8, 2, 23, 1, 1.0]],
        "semantic_after": [[2, 1, 6, 4, 4.0]],
        "semantic_effects": [[9, 1, 13, 4, 4.0]],
    }
    rows = _semantic_rows(payload)
    assert len(rows) == 4
    assert (2, 1, 6, 4, 4.0) in rows


def test_arc_semantic_component_hashing_handles_multicell_shapes() -> None:
    import numpy as np
    from v9.environments.arc.adapter import ARCAdapter
    adapter = object.__new__(ARCAdapter)
    grid = np.array([
        [1, 1, 0],
        [1, 0, 2],
        [0, 0, 2],
    ], dtype=np.int64)
    facts = ARCAdapter.semantic_observation(adapter, grid)
    assert facts
    assert any(int(row[0]) == 5 for row in facts)
