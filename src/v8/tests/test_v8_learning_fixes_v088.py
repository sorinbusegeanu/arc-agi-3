from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import learning_fixes_v088 as learning
from v8.arena import EdgeRecord, NodeRecord
from v8.learning_fixes_v088 import (
    ActorProgress,
    _coherent_cached_transfer_cut,
    _memory_free_action,
    _probe_policy_v088,
    _record_terminal_efficiency_feedback,
    _run_automatic_transfer_experiments_v088,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.promotion import EvidenceGatedPromotionEngine
from v8.publication import _StrategyRow
from v8.transfer import TransferCandidate


def node(
    level,
    memory_type,
    key,
    *,
    support=4,
    significance=1.0,
    learning=1.0,
    transfer=0.5,
    explanatory=2.0,
    future=0.0,
    cognitive=CognitiveState.ACTIVE,
    validation=ValidationState.STRUCTURAL,
    game_mask=3,
    success=0.0,
    cost=0.0,
    attempts=0.0,
):
    uid = MemoryUid.from_key(level, memory_type, key)
    return NodeRecord(
        uid,
        (uid.hi ^ uid.lo) & ((1 << 64) - 1),
        int(level),
        int(memory_type),
        tuple(key),
        int(support),
        float(significance),
        0.0,
        float(learning),
        float(transfer),
        float(explanatory),
        float(future),
        1.0,
        10,
        int(game_mask),
        int(cognitive),
        int(validation),
        float(success),
        float(cost),
        float(attempts),
    )


class V088LearningFixTests(unittest.TestCase):
    def test_coherent_transfer_cut_reuses_matching_record_cache(self):
        node_arena = object()
        edge_arena = object()
        row = node(MemoryLevel.M4, MemoryType.CONCEPT, (7, 8, 9))
        edge = object()
        view = SimpleNamespace(
            _nodes=(node_arena,),
            _edges=(edge_arena,),
            _strategy_version=(2, 4),
            _record_cache={
                id(node_arena): ((row,), 2),
                id(edge_arena): ((edge,), 4),
            },
            _node_by_uid={row.uid: row},
        )

        self.assertEqual(
            _coherent_cached_transfer_cut(view),
            ((row,), (edge,)),
        )
        transfer_edge = object()
        view._v839_transfer_version = (4,)
        view._v839_transfer_edges = (transfer_edge,)
        self.assertEqual(
            _coherent_cached_transfer_cut(view),
            ((row,), (transfer_edge,)),
        )
        view._strategy_version = (2, 6)
        self.assertIsNone(_coherent_cached_transfer_cut(view))

    def test_terminal_efficiency_feedback_uses_existing_index_without_graph_scan(self):
        strategy = node(
            MemoryLevel.M7,
            MemoryType.STRATEGY,
            (1, 2, 3, 4),
        )
        credit = SimpleNamespace(
            uid=strategy.uid,
            level=int(MemoryLevel.M7),
            valence_sum=1.0,
            weight=1.0,
        )
        submitted = []
        evidence = []

        class ReadView:
            _node_by_uid = {strategy.uid: strategy}

            def node_records(self, **_kwargs):
                raise AssertionError("feedback must not rescan the live graph")

        peers = SimpleNamespace(
            _existing_proposal=lambda row, **kwargs: (row, kwargs),
            _submit=submitted.append,
            _append_evidence=lambda *args, **kwargs: evidence.append((args, kwargs)),
        )
        runtime = SimpleNamespace(peers=peers, read_view=ReadView())
        result = SimpleNamespace(
            game_id="ez01",
            primary_valence_credits=(credit,),
        )

        _record_terminal_efficiency_feedback(runtime, (result,))

        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0][0][0], "terminal_strategy_efficiency")

    def test_concept_identity_preserves_full_role_descriptor(self):
        role = node(MemoryLevel.M3, MemoryType.ROLE, (11, 22, 1, 33), support=4, transfer=0.75)
        candidates = EvidenceGatedPromotionEngine().propose((role,), (), budget=16)
        concepts = [item for item in candidates if int(item.level) == int(MemoryLevel.M4)]
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].key_parts, role.key_parts)
        self.assertEqual(
            concepts[0].uid,
            MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, role.key_parts),
        )

    def test_candidate_concept_can_form_probe_only_m5_scaffold(self):
        concept = node(
            MemoryLevel.M4,
            MemoryType.CONCEPT,
            (11, 22, 1, 33),
            support=4,
            transfer=0.75,
            explanatory=2.0,
            cognitive=CognitiveState.CANDIDATE,
            validation=ValidationState.STRUCTURAL,
        )
        candidates = EvidenceGatedPromotionEngine().propose((concept,), (), budget=16)
        self.assertTrue(any(int(item.level) == int(MemoryLevel.M5) for item in candidates))

    def test_lower_level_churn_cannot_starve_eligible_concept_formation(self):
        rows = []
        for index in range(80):
            rows.extend(
                (
                    node(
                        MemoryLevel.M1,
                        MemoryType.CONTINGENCY,
                        (index * 100 + 1, index, index, 0),
                    ),
                    node(
                        MemoryLevel.M1,
                        MemoryType.CONTINGENCY,
                        (index * 100 + 2, index, index, 0),
                    ),
                )
            )
        role = node(
            MemoryLevel.M3,
            MemoryType.ROLE,
            (11, 22, 1, 33),
            support=6,
            transfer=0.75,
            explanatory=3.0,
        )
        rows.append(role)

        candidates = EvidenceGatedPromotionEngine().propose(tuple(rows), (), budget=16)
        concepts = [item for item in candidates if int(item.level) == int(MemoryLevel.M4)]

        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].parents, (role.uid,))
        self.assertEqual(concepts[0].key_parts, role.key_parts)

    def test_missing_higher_tiers_share_reserved_promotion_capacity(self):
        low = []
        for index in range(40):
            low.extend(
                (
                    node(MemoryLevel.M1, MemoryType.CONTINGENCY, (index * 10 + 1, index, index, 0)),
                    node(MemoryLevel.M1, MemoryType.CONTINGENCY, (index * 10 + 2, index, index, 0)),
                )
            )
        role = node(
            MemoryLevel.M3,
            MemoryType.ROLE,
            (101, 102, 0, 103),
            support=6,
            transfer=0.75,
            explanatory=3.0,
        )
        concept = node(
            MemoryLevel.M4,
            MemoryType.CONCEPT,
            (201, 202, 0, 203),
            support=4,
            transfer=0.75,
            explanatory=2.0,
            cognitive=CognitiveState.CANDIDATE,
        )
        consequences = tuple(
            node(
                MemoryLevel.M5,
                MemoryType.CONSEQUENCE,
                (301 + index, 302 + index, 303 + index, 0),
                support=4,
                transfer=0.75,
                explanatory=2.0,
            )
            for index in range(8)
        )

        candidates = EvidenceGatedPromotionEngine().propose(
            tuple((*low, role, concept, *consequences)), (), budget=16
        )
        levels = {int(item.level) for item in candidates}

        self.assertTrue(
            {int(MemoryLevel.M4), int(MemoryLevel.M5), int(MemoryLevel.M6)} <= levels
        )

    def test_reserved_promotion_builds_executable_lineage_to_m7(self):
        contingency = node(
            MemoryLevel.M1,
            MemoryType.CONTINGENCY,
            (7, 3, 9, 11),
            support=5,
            future=0.0,
        )
        role = node(
            MemoryLevel.M3,
            MemoryType.ROLE,
            (401, 0, 1, 402),
            support=6,
            transfer=0.75,
            explanatory=3.0,
            future=0.0,
        )
        rows = [contingency, role]
        edges = [
            EdgeRecord(
                role.uid,
                int(RelationType.EXPLAINS),
                contingency.uid,
                1,
                1,
            )
        ]
        engine = EvidenceGatedPromotionEngine()

        for generation in range(1, 6):
            candidates = engine.propose(tuple(rows), tuple(edges), budget=16)
            additions = [
                candidate
                for candidate in candidates
                if int(candidate.level) >= int(MemoryLevel.M4)
                and all(row.uid != candidate.uid for row in rows)
            ]
            for candidate in additions:
                rows.append(
                    node(
                        candidate.level,
                        candidate.memory_type,
                        candidate.key_parts,
                        support=candidate.support,
                        significance=candidate.significance,
                        learning=candidate.learning_value,
                        transfer=candidate.transfer_prior,
                        explanatory=candidate.explanatory_reach,
                        future=candidate.future_option_delta,
                        cognitive=CognitiveState(candidate.cognitive_state),
                        validation=ValidationState(candidate.validation_state),
                    )
                )
                for parent_index, parent in enumerate(candidate.parents):
                    relation = (
                        RelationType.LEADS_TO
                        if int(candidate.level) == int(MemoryLevel.M7) and parent_index == 0
                        else RelationType.DEPENDS_ON
                        if int(candidate.level) == int(MemoryLevel.M7)
                        else RelationType.EXPLAINS
                    )
                    edges.append(
                        EdgeRecord(candidate.uid, int(relation), parent, 1, generation)
                    )

        strategies = [row for row in rows if int(row.level) == int(MemoryLevel.M7)]
        self.assertTrue(strategies)
        parents = {}
        for edge in edges:
            if int(edge.relation_type) in {
                int(RelationType.EXPLAINS),
                int(RelationType.LEADS_TO),
            }:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)
        self.assertTrue(
            learning._has_cached_ancestor(parents, strategies[0].uid, role.uid)
        )

    def test_memory_free_probe_never_reads_memory_policy(self):
        class FakeEnv:
            def __init__(self, **kwargs):
                del kwargs
                self.last_levels_completed = 0
                self.last_outcome_polarity = "neutral"

            def observe(self):
                return ((0,),)

            def available_actions(self):
                return [1, 2, 3]

            def step(self, action):
                self.last_outcome_polarity = "neutral"
                return ((int(action),),)

            def reset(self):
                return ((0,),)

        class NoMemoryReadView:
            def __getattr__(self, name):
                raise AssertionError(f"memory read attempted in OFF condition: {name}")

        with patch("v7.environment.arc_adapter.ArcGridEnvironment", FakeEnv):
            metric, used = _probe_policy_v088(
                read_view=NoMemoryReadView(),
                game_id="fake",
                env_root=None,
                seed=7,
                steps=8,
                required_ancestor=None,
            )
        self.assertEqual(used, 0)
        self.assertEqual(metric, 0.0)

    def test_unexecutable_transfer_candidate_does_not_consume_budget(self):
        candidate_uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (1, 2, 3, 4))
        candidate = TransferCandidate(candidate_uid, 1, 0.8, (101,), MemoryUid.zero(), ())

        class Transfer:
            def candidates(self, nodes, provenance=None):
                del nodes, provenance
                return (candidate,)

        class Peers:
            transfer = Transfer()

            def record_transfer_trial(self, *args, **kwargs):
                raise AssertionError("unexecutable candidate must not become a trial")

        class ReadView:
            def node_records(self):
                return ()

            def source_games(self, uid):
                del uid
                return frozenset()

        class Runtime:
            peers = Peers()
            read_view = ReadView()

        with patch("v8.learning_fixes_v088._held_out_games", return_value=("hold01",)), patch(
            "v8.learning_fixes_v088._probe_policy_v088", return_value=(0.0, 0)
        ):
            summary = _run_automatic_transfer_experiments_v088(
                Runtime(),
                games=("train01",),
                env_root=None,
                seed=0,
                steps_per_trial=4,
                max_trials=2,
            )
        self.assertEqual(summary.attempted, 0)
        self.assertEqual(summary.completed, 0)

    def test_unexecutable_transfer_candidate_scan_is_bounded(self):
        candidates = tuple(
            TransferCandidate(
                MemoryUid.from_key(
                    MemoryLevel.M4,
                    MemoryType.CONCEPT,
                    (index + 1, 2, 3, 4),
                ),
                1,
                1.0 - index / 1000.0,
                (101,),
                MemoryUid.zero(),
                (),
            )
            for index in range(100)
        )

        class Transfer:
            def candidates(self, nodes, provenance=None):
                del nodes, provenance
                return candidates

        class Peers:
            transfer = Transfer()

            def record_transfer_trial(self, *args, **kwargs):
                raise AssertionError("unexecutable candidate must not become a trial")

        class ReadView:
            def node_records(self):
                return ()

            def source_games(self, uid):
                del uid
                return frozenset()

        class Runtime:
            peers = Peers()
            read_view = ReadView()

        with patch(
            "v8.learning_fixes_v088._held_out_games", return_value=("hold01",)
        ), patch(
            "v8.learning_fixes_v088._probe_policy_v088", return_value=(0.0, 0)
        ) as probe:
            summary = _run_automatic_transfer_experiments_v088(
                Runtime(),
                games=("train01",),
                env_root=None,
                seed=0,
                steps_per_trial=4,
                max_trials=2,
            )

        self.assertEqual(summary.attempted, 0)
        self.assertEqual(summary.completed, 0)
        self.assertEqual(probe.call_count, 2)

    def test_efficiency_is_relative_to_same_outcome_alternatives(self):
        from v8 import behavior_recovery as behavior_module

        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        fast_uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, outcome.hi, outcome.lo, 9))
        slow_uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (2, outcome.hi, outcome.lo, 9))

        class View:
            _preferred_outcomes = set()
            _node_by_uid = {}

            def strategy_has_ancestor(self, strategy_uid, ancestor_uid):
                del strategy_uid, ancestor_uid
                return True

        fast = _StrategyRow(1, outcome, fast_uid, 5, 0.6, 2.0, 9, False, False)
        slow = _StrategyRow(2, outcome, slow_uid, 5, 0.6, 8.0, 9, False, False)

        def scores(rows):
            return behavior_module._score_strategy_rows(
                View(),
                rows,
                available={row.action_id for row in rows},
                outcome_uid=None,
                required_ancestor=None,
                excluded_strategies=frozenset(),
                ignore_preference=True,
                cross_context=False,
            )

        paired = scores((fast, slow))
        self.assertEqual(paired[0].strategy_uid, fast_uid)
        self.assertGreater(paired[0].score, paired[1].score)
        self.assertAlmostEqual(scores((fast,))[0].score, scores((slow,))[0].score)

    def test_progress_reports_best_and_last_solve_lengths(self):
        from v8 import diagnostics

        rows = (
            ActorProgress(1, "ez01", 1000, 3, 0, 5, first_win_step=205, best_win_steps=180, last_win_steps=190),
        )
        line = diagnostics.format_game_rate_line(rows)
        self.assertIn("ez01:B=180,L=190", line)
        self.assertNotIn("best_win_actions=", line)
        self.assertNotIn("last_win_actions=", line)

    def test_memory_free_action_is_seeded_but_memory_independent(self):
        from random import Random

        self.assertEqual(_memory_free_action((1, 2, 3), Random(11)), _memory_free_action((1, 2, 3), Random(11)))


if __name__ == "__main__":
    unittest.main()
