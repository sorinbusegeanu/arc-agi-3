from __future__ import annotations

from random import Random

from v7.derivation.planning_runtime import Phase1PlanningBuilder
from v7.environment.cognition import DecisionContext, LocalCognitionOverlay
from v7.environment.phase1_policy import (
    Phase1ActionScorer,
    StrategyExecutionCursor,
    select_phase1_action,
)
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ContingencyIndexMutation
from v7.memory.models import EdgeMutation, NodeMutation
from v7.memory.planning import (
    REL_PLAN_FAILURE,
    PersistentPlanningGraph,
)
from v7.memory.writer import CanonicalMemoryWriter


def _episode(
    memory_id: MemoryId,
    *,
    step: int,
    context: int,
    next_context: int | None,
    action: int,
    terminal: int,
    changed: int = 1,
) -> EvidenceRecord:
    return EvidenceRecord(
        memory_id=memory_id,
        evidence_type=int(EvidenceType.EPISODE),
        generation_id=1,
        source_game="g1",
        source_context=str(context),
        source_global_step=step,
        payload={
            "context_signature": context,
            "context_signatures": [1, 2, context],
            "next_context_signatures": []
            if next_context is None
            else [1, 2, next_context],
            "action_id": action,
            "terminal_polarity": terminal,
            "changed_cells": changed,
            "future_option_delta": 1.0 if terminal > 0 else 0.0,
        },
    )


def test_phase1_builder_persists_planning_graph_and_executable_procedure(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    m1_a = MemoryId(1)
    m1_b = MemoryId(2)
    m1_bad = MemoryId(3)
    writer.apply_mutation_batch(
        (
            NodeMutation(m1_a, MemoryLevel.M1, 100, support_delta=4),
            NodeMutation(m1_b, MemoryLevel.M1, 100, support_delta=4),
            NodeMutation(m1_bad, MemoryLevel.M1, 100, support_delta=4),
        )
    )
    writer.apply_contingency_index_batch(
        (
            ContingencyIndexMutation(10, 1, m1_a),
            ContingencyIndexMutation(20, 2, m1_b),
            ContingencyIndexMutation(10, 3, m1_bad),
        )
    )
    writer.apply_edge_batch(
        (EdgeMutation(m1_bad, REL_PLAN_FAILURE, m1_bad, support_delta=3),)
    )

    store = EvidenceStore(tmp_path / "evidence.sqlite")
    try:
        store.append_evidence_batch(
            (
                _episode(
                    m1_a,
                    step=1,
                    context=10,
                    next_context=20,
                    action=1,
                    terminal=0,
                ),
                _episode(
                    m1_b,
                    step=2,
                    context=20,
                    next_context=None,
                    action=2,
                    terminal=1,
                ),
                EvidenceRecord(
                    memory_id=None,
                    evidence_type=int(EvidenceType.TRAJECTORY),
                    generation_id=1,
                    source_game="g1",
                    source_context="level_0000",
                    source_global_step=2,
                    payload={
                        "level_key": "level_0000",
                        "steps_to_success": 2,
                        "future_option_per_action": 0.5,
                        "action_sequence": [1, 2],
                        "context_sequence": [10, 20],
                        "success": True,
                    },
                ),
            )
        )
        builder = Phase1PlanningBuilder(writer, store)
        first = builder.derive()
        edge_support_after_first = dict(getattr(writer, "_edge_support"))
        second = builder.derive()
        edge_support_after_second = dict(getattr(writer, "_edge_support"))

        assert first.procedures == 1
        assert second.procedures == 0
        assert edge_support_after_second == edge_support_after_first

        _, view, _ = writer.commit_generation()
        planning = PersistentPlanningGraph.from_view(view)
        signal = planning.evaluate(view, (m1_a,))
        bad = planning.evaluate(view, (m1_bad,))

        assert signal.reachable_nodes == 1
        assert signal.success_reachability > 0.0
        assert bad.failure_risk == 1.0
        assert len(planning.strategies) == 1
        procedure = next(iter(planning.strategies.values()))
        assert [step.context_signature for step in procedure.steps] == [10, 20]
        assert [step.action_id for step in procedure.steps] == [1, 2]
    finally:
        store.close()


def test_phase1_strategy_cursor_replays_successful_sequence_and_avoids_failure(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    m1_a = MemoryId(1)
    m1_b = MemoryId(2)
    m1_bad = MemoryId(3)
    writer.apply_mutation_batch(
        (
            NodeMutation(m1_a, MemoryLevel.M1, 100, support_delta=5),
            NodeMutation(m1_b, MemoryLevel.M1, 100, support_delta=5),
            NodeMutation(m1_bad, MemoryLevel.M1, 100, support_delta=5),
        )
    )
    writer.apply_contingency_index_batch(
        (
            ContingencyIndexMutation(10, 1, m1_a),
            ContingencyIndexMutation(20, 2, m1_b),
            ContingencyIndexMutation(10, 3, m1_bad),
        )
    )
    writer.apply_edge_batch(
        (EdgeMutation(m1_bad, REL_PLAN_FAILURE, m1_bad, support_delta=5),)
    )
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    try:
        store.append_evidence_batch(
            (
                _episode(
                    m1_a,
                    step=1,
                    context=10,
                    next_context=20,
                    action=1,
                    terminal=0,
                ),
                _episode(
                    m1_b,
                    step=2,
                    context=20,
                    next_context=None,
                    action=2,
                    terminal=1,
                ),
                EvidenceRecord(
                    memory_id=None,
                    evidence_type=int(EvidenceType.TRAJECTORY),
                    generation_id=1,
                    source_game="g1",
                    source_context="level_0000",
                    source_global_step=2,
                    payload={
                        "level_key": "level_0000",
                        "steps_to_success": 2,
                        "future_option_per_action": 0.5,
                        "action_sequence": [1, 2],
                        "context_sequence": [10, 20],
                        "success": True,
                    },
                ),
            )
        )
        Phase1PlanningBuilder(writer, store).derive()
        _, view, _ = writer.commit_generation()
        planning = PersistentPlanningGraph.from_view(view)
        scorer = Phase1ActionScorer(planning)
        overlay = LocalCognitionOverlay()
        cursor = StrategyExecutionCursor()
        rng = Random(7)

        first_context = DecisionContext((100, 101, 10, 102), 0, 0)
        first_rows = scorer.score_actions(
            view=view,
            contexts=first_context,
            actions=(1, 3),
            overlay=overlay,
        )
        by_action = {row.action_id: row for row in first_rows}
        assert by_action[3].persistent_failure_risk == 1.0
        assert by_action[1].score > by_action[3].score

        first = select_phase1_action(
            view=view,
            planning=planning,
            cursor=cursor,
            contexts=first_context,
            decisions=first_rows,
            rng=rng,
            epsilon=1.0,
        )
        assert first.mode == "strategy"
        assert first.decision.action_id == 1
        assert first.strategy_id is not None
        cursor.observe_outcome(
            selected_strategy_id=first.strategy_id,
            terminal_polarity=0,
            next_context_signature=20,
        )

        second_context = DecisionContext((100, 101, 20, 103), 0, 0)
        second_rows = scorer.score_actions(
            view=view,
            contexts=second_context,
            actions=(2, 3),
            overlay=overlay,
        )
        second = select_phase1_action(
            view=view,
            planning=planning,
            cursor=cursor,
            contexts=second_context,
            decisions=second_rows,
            rng=rng,
            epsilon=1.0,
        )
        assert second.mode == "strategy"
        assert second.decision.action_id == 2
    finally:
        store.close()
