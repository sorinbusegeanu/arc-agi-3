from __future__ import annotations

from dataclasses import replace

from v7.context_evidence import ContextEpisodeEvidence
from v7.environment.cognition import ContextualActionScorer, DecisionContext, DecisionSupport
from v7.environment.online_sampling import _maybe_select_probe, _raw_probe_strength
from v7.environment.phase1_policy import (
    Phase1ActionDecision,
    Phase1ActionScorer,
    Phase1Selection,
    StrategyExecutionCursor,
    _is_transfer_frontier,
    _local_policy_confidence,
    select_phase1_action,
)
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore
from v7.memory.gate_validation import DEFAULT_GATE_POLICIES
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ActionAggregateDelta, RoleIndexMutation
from v7.memory.lifecycle_runtime import MemoryLifecycleRuntime
from v7.memory.models import NodeMutation, ScoreMutation
from v7.memory.planning import PersistentPlanningGraph
from v7.memory.read_view import MemoryReadView
from v7.memory.state import CognitiveState, GateId, GateValidationState
from v7.memory.writer import CanonicalMemoryWriter
from v7.runtime import V7Runtime, V7RuntimeConfig


class _ZeroRandom:
    def random(self) -> float:
        return 0.0

    def choices(self, population, weights=None, k=1):
        del weights, k
        return [population[0]]


def _decision(
    action_id: int,
    *,
    score: float,
    exploration: float = 0.0,
    confidence: float = 0.0,
    local_support: int = 0,
    context_rank: int = 3,
    failure: float = 0.0,
    contradiction: float = 0.0,
) -> Phase1ActionDecision:
    return Phase1ActionDecision(
        action_id=action_id,
        score=score,
        support=DecisionSupport(
            context_signature=11,
            local_support=local_support,
            context_rank=context_rank,
        ),
        exploration_score=exploration,
        failure_risk=failure,
        contradiction_risk=contradiction,
        future_reachability=0.0,
        memory_confidence=confidence,
        persistent_reachability=0.0,
        persistent_success_reachability=0.0,
        persistent_failure_risk=0.0,
        dead_end_risk=0.0,
        option_loss_risk=0.0,
    )


def _empty_view() -> MemoryReadView:
    return MemoryReadView.freeze(
        generation_id=1,
        nodes={},
        scores={},
        adjacency={},
    )


def _probe_role_view() -> tuple[MemoryReadView, MemoryId]:
    writer = CanonicalMemoryWriter(gate_candidates=True)
    memory_id = MemoryId(17)
    writer.apply_mutation_batch(
        (
            NodeMutation(
                memory_id,
                MemoryLevel.M3,
                300,
                support_delta=12,
                cognitive_state=int(CognitiveState.PROBE_ONLY),
                validation_state=int(GateValidationState.PROBE_ELIGIBLE),
                gate_id=int(GateId.G23R),
            ),
        )
    )
    writer.apply_score_batch(
        (
            ScoreMutation(
                memory_id,
                significance=1.0,
                learning_value=1.0,
                transfer_prior=1.0,
                explanatory_potential=1.0,
            ),
        )
    )
    writer.apply_role_index_batch((RoleIndexMutation(11, 1, memory_id, None),))
    _state, view, _delta = writer.commit_generation()
    return view, memory_id


def test_probe_strength_can_cross_reachable_gate_thresholds() -> None:
    view, memory_id = _probe_role_view()
    strength = _raw_probe_strength(view, memory_id)
    assert 0.08 * strength > DEFAULT_GATE_POLICIES[GateId.G23R].minimum_causal_gain
    assert DEFAULT_GATE_POLICIES[GateId.G34].minimum_causal_gain < 0.08
    assert DEFAULT_GATE_POLICIES[GateId.G45].minimum_causal_gain < 0.10
    assert DEFAULT_GATE_POLICIES[GateId.G56].minimum_causal_gain < 0.12


def test_probe_cannot_replace_a_much_better_selected_action() -> None:
    view, _memory_id = _probe_role_view()
    scorer = Phase1ActionScorer(PersistentPlanningGraph.from_view(view))
    weak = _decision(1, score=0.10)
    strong = _decision(2, score=0.80)
    selection = Phase1Selection(strong, "memory")
    updated, probe_id, contribution = _maybe_select_probe(
        view=view,
        scorer=scorer,
        contexts=DecisionContext((11,), 11, 11),
        decisions=(weak, strong),
        selection=selection,
        rng=_ZeroRandom(),
        epsilon=0.10,
    )
    assert updated.decision.action_id == 2
    assert probe_id is None
    assert contribution == 0.0


def test_gate_credit_is_conditioned_on_observed_outcome(tmp_path) -> None:
    runtime = V7Runtime(
        V7RuntimeConfig.from_path(
            tmp_path / "runtime",
            restore=False,
            derivation_workers=1,
        )
    )
    try:
        memory_id = MemoryId(31)
        runtime.writer.apply_mutation_batch(
            (
                NodeMutation(
                    memory_id,
                    MemoryLevel.M4,
                    400,
                    support_delta=4,
                    cognitive_state=int(CognitiveState.PROBE_ONLY),
                    validation_state=int(GateValidationState.PROBE_ELIGIBLE),
                    gate_id=int(GateId.G34),
                ),
            )
        )
        evidence = ContextEpisodeEvidence(
            context_signature=11,
            action_id=1,
            outcome_signature=2,
            success=False,
            source_game="heldout",
            source_context="11",
            source_global_step=5,
            terminal_polarity=-1,
            future_option_delta=-4.0,
            decision_memory_contributions=((int(memory_id), 0.08),),
        )
        runtime._write_decision_gate_trials((evidence,))
        row = runtime.lifecycle_evidence.connection.execute(
            "SELECT contribution,causal_gain,terminal_gain,intervention_type "
            "FROM gate_trials WHERE memory_id=?",
            (int(memory_id),),
        ).fetchone()
        assert row is not None
        assert row[0] == 0.08
        assert row[1] == -0.08
        assert row[2] == -0.08
        assert row[3] == "decision_outcome_ablation"
    finally:
        runtime.close()


def test_failure_and_contradiction_reduce_confidence_and_keep_frontier() -> None:
    confidence = _local_policy_confidence(
        0.95,
        signature_count=5,
        context_rank=3,
        local_support=2,
        failure_risk=1.0,
        contradiction_risk=0.0,
    )
    assert confidence < 0.70
    risky = _decision(
        1,
        score=1.0,
        confidence=0.95,
        local_support=2,
        context_rank=3,
        failure=1.0,
    )
    assert _is_transfer_frontier(DecisionContext((1, 2, 3, 4, 5), 3, 5), risky)


def test_high_confidence_does_not_bypass_epsilon_exploration() -> None:
    view = _empty_view()
    planning = PersistentPlanningGraph.from_view(view)
    memory = _decision(
        1,
        score=1.0,
        exploration=0.1,
        confidence=0.99,
        local_support=2,
        context_rank=3,
    )
    exploratory = _decision(
        2,
        score=0.0,
        exploration=1.0,
        confidence=0.0,
        local_support=2,
        context_rank=3,
    )
    selection = select_phase1_action(
        view=view,
        planning=planning,
        cursor=StrategyExecutionCursor(),
        contexts=DecisionContext((1, 2, 3, 4, 5), 3, 5),
        decisions=(memory, exploratory),
        rng=_ZeroRandom(),
        epsilon=0.10,
    )
    assert selection.mode == "exploration"
    assert selection.decision.action_id == 2
    assert selection.effective_epsilon >= 0.02


def test_lifecycle_does_not_age_unused_memory_by_epoch(tmp_path) -> None:
    store = EvidenceLifecycleStore(tmp_path / "lifecycle.sqlite")
    try:
        writer = CanonicalMemoryWriter()
        memory_id = MemoryId(41)
        writer.apply_mutation_batch(
            (
                NodeMutation(
                    memory_id,
                    MemoryLevel.M4,
                    400,
                    support_delta=4,
                    cognitive_state=int(CognitiveState.ACTIVE),
                    validation_state=int(GateValidationState.VALIDATED),
                    gate_id=int(GateId.G34),
                ),
            )
        )
        _state, view, _delta = writer.commit_generation()
        runtime = MemoryLifecycleRuntime(evidence_lifecycle=store)
        runtime.run(view, writer=writer)
        first = store.lifecycle_window(memory_id)
        assert first is not None
        runtime.run(view, writer=writer)
        second = store.lifecycle_window(memory_id)
        assert second is not None
        assert second.consecutive_low_windows == first.consecutive_low_windows
        assert second.consecutive_harm_windows == first.consecutive_harm_windows
        assert second.last_generation == first.last_generation
    finally:
        store.close()


def test_lifetime_action_aggregate_is_reporting_only() -> None:
    writer = CanonicalMemoryWriter()
    writer.apply_action_aggregate_batch(
        (
            ActionAggregateDelta(
                action_id=9,
                future_option_sum_delta=100.0,
                future_option_count_delta=100,
                positive_count_delta=100,
            ),
        )
    )
    _state, view, _delta = writer.commit_generation()
    assert ContextualActionScorer._global_prior(view, 9) == 0.0
