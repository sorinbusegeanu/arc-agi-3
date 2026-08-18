from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import v8  # noqa: F401 - installs the chronological runtime stack
from v8 import behavior_recovery as behavior
from v8 import environment_neutrality_v837 as v837
from v8 import optimizer_budget_control_v830 as v830
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_transfer_v833 as transfer
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8 import trajectory_target_minimization_v820 as v820
from v8.environment_contract import (
    BoundaryEvent,
    BoundaryScope,
    OptimizationScopeKind,
    optimization_scope_for,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64
from v8.structural_events import NormalizedPrimitive, StructuralFact


def _node(uid, level, memory_type, key_parts=(), **kwargs):
    values = dict(
        uid=uid,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=tuple(key_parts),
        transfer_prior=0.0,
        significance=0.0,
        learning_value=0.0,
        support_count=1,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _edge(relation, source, target, score=1.0):
    return SimpleNamespace(
        relation_type=int(relation),
        source_uid=source,
        target_uid=target,
        score=float(score),
    )


class EnvironmentNeutralityV837Tests(unittest.TestCase):
    def test_legacy_arc_target_decodes_to_generic_boundary(self):
        target = optimizer.TrajectoryTarget(5, "WIN")
        self.assertIsInstance(target, v837.V837TrajectoryTarget)
        self.assertEqual(target.boundary_scope, BoundaryScope.EPISODE.value)
        self.assertEqual(target.primary_valence, +1)
        restored = optimizer.TrajectoryTarget.from_dict(target.to_dict())
        self.assertEqual(restored, target)

    def test_complete_target_scope_is_episode_valence_not_full_win(self):
        anchor = optimizer.ReplayAnchor("env-a", 0, (), None)
        target = optimizer.TrajectoryTarget(5, "WIN")
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (3, 3)),
            anchor,
            target,
            (3, 3),
        )
        scope = optimization_scope_for(source)
        self.assertEqual(scope.kind, OptimizationScopeKind.BOUNDARY)
        self.assertEqual(scope.label(), "EPISODE:+1")
        candidate = optimizer.TrajectoryCandidate(
            optimizer._candidate_id(source, "DELETE_ACTION", (3,)),
            source,
            "DELETE_ACTION",
            (3,),
            0,
            1,
        )
        game, key, cost = v830._candidate_scope(candidate)
        self.assertEqual(game, "env-a")
        self.assertEqual(key, scope.legacy_budget_key())
        self.assertEqual(cost, 2)

    def test_shared_raw_action_id_without_correspondence_is_not_transfer(self):
        sampler = portfolio.PortfolioSampler("target", seed=1)
        sampler.begin_lease(1)
        strategy_uid = MemoryUid(10, 10)
        outcome_uid = MemoryUid(20, 20)
        source_role = MemoryUid(30, 30)
        source_m1 = MemoryUid(40, 40)
        source_game = int(stable_u64("source", person=b"v8-game"))
        strategy = SimpleNamespace(
            action_id=3,
            outcome_uid=outcome_uid,
            strategy_uid=strategy_uid,
            support=10,
            reliability=0.9,
            mean_cost=2.0,
            probationary=False,
        )
        nodes = {
            strategy_uid: _node(strategy_uid, MemoryLevel.M7, MemoryType.STRATEGY, (3, 20, 20, 0)),
            source_role: _node(source_role, MemoryLevel.M3, MemoryType.ROLE),
            source_m1: _node(source_m1, MemoryLevel.M1, MemoryType.CONTINGENCY, (11, 3, 12, 13)),
        }
        fake = SimpleNamespace(
            _strategy_version=(1,),
            _strategy_fallback=(strategy,),
            _node_by_uid=nodes,
            _parents={strategy_uid: {source_role}, source_role: {source_m1}},
        )
        fake._refresh_strategy_cache = lambda: None
        fake.edge_records = lambda: (
            _edge(RelationType.GAME_PROVENANCE, source_m1, MemoryUid(0, source_game)),
        )
        with mock.patch.object(behavior, "_CURRENT_ACTOR_VIEW", fake), mock.patch.object(
            behavior, "strategy_can_control", return_value=True
        ):
            self.assertIsNone(transfer._cross_game_transfer_action(sampler, (1, 3)))

    def test_formal_correspondence_grounds_foreign_a3_to_target_a1(self):
        sampler = portfolio.PortfolioSampler("target", seed=2)
        sampler.begin_lease(2)
        strategy_uid = MemoryUid(101, 1)
        outcome_uid = MemoryUid(102, 2)
        source_role = MemoryUid(103, 3)
        target_role = MemoryUid(104, 4)
        source_m1 = MemoryUid(105, 5)
        target_m1 = MemoryUid(106, 6)
        source_game = int(stable_u64("source", person=b"v8-game"))
        target_game = int(stable_u64("target", person=b"v8-game"))
        strategy = SimpleNamespace(
            action_id=3,
            outcome_uid=outcome_uid,
            strategy_uid=strategy_uid,
            support=10,
            reliability=0.95,
            mean_cost=2.0,
            probationary=False,
        )
        nodes = {
            strategy_uid: _node(strategy_uid, MemoryLevel.M7, MemoryType.STRATEGY, (3, 102, 2, 0), transfer_prior=0.7),
            source_role: _node(source_role, MemoryLevel.M3, MemoryType.ROLE),
            target_role: _node(target_role, MemoryLevel.M3, MemoryType.ROLE),
            source_m1: _node(source_m1, MemoryLevel.M1, MemoryType.CONTINGENCY, (11, 3, 12, 13)),
            target_m1: _node(target_m1, MemoryLevel.M1, MemoryType.CONTINGENCY, (21, 1, 22, 23)),
        }
        fake = SimpleNamespace(
            _strategy_version=(2,),
            _strategy_fallback=(strategy,),
            _node_by_uid=nodes,
            _parents={
                strategy_uid: {source_role},
                source_role: {source_m1},
                target_role: {target_m1},
            },
        )
        fake._refresh_strategy_cache = lambda: None
        fake.edge_records = lambda: (
            _edge(RelationType.GAME_PROVENANCE, source_m1, MemoryUid(0, source_game)),
            _edge(RelationType.GAME_PROVENANCE, target_m1, MemoryUid(0, target_game)),
            _edge(RelationType.TRANSFER_CORRESPONDENCE, source_role, target_role, 0.9),
        )
        with mock.patch.object(behavior, "_CURRENT_ACTOR_VIEW", fake), mock.patch.object(
            behavior, "strategy_can_control", return_value=True
        ):
            selected = transfer._cross_game_transfer_action(sampler, (1, 3))
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 1)
        self.assertEqual(selected[1], "M7_CORRESPONDENCE")
        self.assertNotEqual(selected[0], strategy.action_id)

    def test_m1n_reuse_grounds_to_current_environment_action(self):
        sampler = portfolio.PortfolioSampler("target", seed=3)
        sampler.begin_lease(3)
        normalized_uid = MemoryUid(201, 1)
        source_m1 = MemoryUid(202, 2)
        target_m1 = MemoryUid(203, 3)
        source_game = int(stable_u64("source", person=b"v8-game"))
        target_game = int(stable_u64("target", person=b"v8-game"))
        token = StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, 99).token
        nodes = {
            normalized_uid: _node(
                normalized_uid,
                MemoryLevel.M1,
                MemoryType.CONTINGENCY,
                (token,),
                significance=0.8,
                support_count=8,
            ),
            source_m1: _node(source_m1, MemoryLevel.M1, MemoryType.CONTINGENCY, (11, 3, 12, 13)),
            target_m1: _node(target_m1, MemoryLevel.M1, MemoryType.CONTINGENCY, (21, 1, 22, 23)),
        }
        fake = SimpleNamespace(
            _strategy_version=(3,),
            _strategy_fallback=(),
            _node_by_uid=nodes,
            _parents={normalized_uid: {source_m1, target_m1}},
        )
        fake._refresh_strategy_cache = lambda: None
        fake.edge_records = lambda: (
            _edge(RelationType.GAME_PROVENANCE, source_m1, MemoryUid(0, source_game)),
            _edge(RelationType.GAME_PROVENANCE, target_m1, MemoryUid(0, target_game)),
        )
        with mock.patch.object(behavior, "_CURRENT_ACTOR_VIEW", fake):
            selected = transfer._cross_game_transfer_action(sampler, (1, 3))
        self.assertEqual(selected[:2], (1, "M1N_GROUNDED"))

    def test_generic_boundary_ends_random_rollout_without_arc_fields(self):
        from v8 import decision_point_sampling_v821 as sampling

        sampler = portfolio.PortfolioSampler("symbolic", seed=4)
        sampler.begin_lease(4)
        sampler._v833_random_rollout = True
        sampler.base.current = sampling.Intervention("RANDOM_WALK", (0, 10), 7, ())
        sampler.observe_transition(
            before_context=10,
            after_context=11,
            action=7,
            after_actions=(7, 8),
            history_after=(7,),
            structural_changed=True,
            boundary_event=BoundaryEvent(BoundaryScope.SUBEPISODE, +1, True),
        )
        self.assertFalse(sampler._v833_random_rollout)
        self.assertTrue(sampler.saw_progress)

    def test_non_arc_environment_can_validate_generic_episode_target(self):
        class SymbolicEnv:
            def __init__(self):
                self.state = 0
                self.event = BoundaryEvent()

            def reset(self):
                self.state = 0
                self.event = BoundaryEvent()

            def observe(self):
                return self.state

            def available_actions(self):
                return (1, 2)

            def step(self, action):
                if int(action) == 1:
                    self.state = 1
                elif int(action) == 2 and self.state == 1:
                    self.state = 2
                    self.event = BoundaryEvent(BoundaryScope.EPISODE, +1, False)
                return self.state

            def cognitive_boundary_event(self):
                return self.event

        service = SimpleNamespace(
            _v837_environment_factory=lambda **_kwargs: SymbolicEnv(),
            _v818_prefix_for=lambda _candidate: (),
        )
        target = optimizer.TrajectoryTarget(
            0,
            "BOUNDARY",
            BoundaryScope.EPISODE.value,
            +1,
            False,
        )
        anchor = optimizer.ReplayAnchor("symbolic", 0, (), None)
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (1, 1, 2)),
            anchor,
            target,
            (1, 1, 2),
        )
        candidate = optimizer.TrajectoryCandidate(
            optimizer._candidate_id(source, "DELETE_ACTION", (1, 2)),
            source,
            "DELETE_ACTION",
            (1, 2),
            1,
            1,
        )
        validator = v837._EnvironmentReplayValidator(service, "symbolic")
        result = validator.validate(candidate)
        self.assertTrue(result.success)
        self.assertEqual(result.successes, len(v818._VALIDATION_SEEDS))
        self.assertTrue(v820._is_arc_validator(service))


if __name__ == "__main__":
    unittest.main()
