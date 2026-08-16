from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

import v8
from v7.environment.arc_adapter import ArcGridEnvironment
from v8.action_targeting_v810 import is_structural_click_token, structural_click_targets
from v8.arena import EdgeRecord, NodeRecord
from v8.behavior_recovery import CausalEvidenceGatedPromotionEngine
from v8.future_options import FutureOptionEstimator
from v8.learning_blockers_v055 import (
    _SEQUENCE_MARKER,
    _composite_candidates,
    _path_for_composite,
    control_context_signature,
    is_composite_strategy,
    pack_action_choice,
    richer_outcome_key,
    unpack_action_choice,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.preference import PreferenceEstimator
from v8.world_model import WorldModelEstimator


def node(
    *,
    level,
    memory_type,
    key_parts,
    support=4,
    uid=None,
    cognitive_state=CognitiveState.ACTIVE,
    validation_state=ValidationState.STRUCTURAL,
    valence_sum=0.0,
    valence_weight=0.0,
):
    uid = uid or MemoryUid.from_key(level, memory_type, key_parts)
    return NodeRecord(
        uid=uid,
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=tuple(key_parts),
        support_count=int(support),
        significance_sum=float(support),
        prediction_error_sum=0.0,
        learning_value_sum=float(support),
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=0.0,
        score_weight=float(support),
        updated_watermark=1,
        cognitive_state=int(cognitive_state),
        validation_state=int(validation_state),
        primary_valence_sum=float(valence_sum),
        primary_valence_sq_sum=abs(float(valence_sum)),
        primary_valence_weight=float(valence_weight),
    )


class _FakeRaw:
    def __init__(self, grid, *, levels=0, state="NOT_FINISHED", actions=(1, 6)):
        self.frame = np.asarray(grid, dtype=np.int64)
        self.levels_completed = int(levels)
        self.state = state
        self.available_actions = list(actions)


class _FakeEnv:
    def __init__(self):
        self.calls = []
        self.levels = 0
        self.raw = _FakeRaw([[0, 0], [0, 0]], levels=0)

    def reset(self):
        self.levels = 0
        self.raw = _FakeRaw([[0, 0], [0, 0]], levels=0)
        return self.raw

    @staticmethod
    def _action_id(action):
        try:
            return int(action)
        except TypeError:
            for name in ("id", "action_id", "value"):
                value = getattr(action, name, None)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            text = str(action)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits)
            raise

    def step(self, action, data=None):
        action_id = self._action_id(action)
        self.calls.append((action_id, data))
        if data is None and action_id == 1:
            self.levels += 1
        self.raw = _FakeRaw([[self.levels, 0], [0, 0]], levels=self.levels)
        return self.raw


def _factory(**_kwargs):
    return _FakeEnv()


class LearningBlockersV055Tests(unittest.TestCase):
    def test_complex_action_tokens_round_trip_all_coordinates(self):
        seen = set()
        for y in range(64):
            for x in range(64):
                token = pack_action_choice(6, x, y)
                self.assertNotIn(token, seen)
                seen.add(token)
                action, data = unpack_action_choice(token)
                self.assertEqual(action, 6)
                self.assertEqual(data, {"x": x, "y": y})
        self.assertEqual(len(seen), 4096)
        self.assertEqual(unpack_action_choice(3), (3, None))

    def test_environment_exposes_structural_targets_and_executes_coordinate_payload(self):
        env = ArcGridEnvironment(game_id="fixture", env_factory=_factory)
        first = env.available_actions()
        self.assertIn(1, first)
        self.assertNotIn(6, first)
        complex_tokens = [value for value in first if is_structural_click_token(value)]
        self.assertTrue(complex_tokens)
        self.assertLessEqual(len(complex_tokens), 96)
        target = structural_click_targets(env.observe())[0]
        self.assertIn(target.token, first)
        env.step(target.token)
        self.assertEqual(env.env.calls[-1], (6, {"x": target.x, "y": target.y}))
        second = env.available_actions()
        self.assertEqual(set(first), set(second))

    def test_level_advancement_remains_positive_primitive_signal(self):
        env = ArcGridEnvironment(game_id="fixture", env_factory=_factory)
        env.step(1)
        self.assertTrue(env.level_completed_event)
        self.assertEqual(env.last_outcome_polarity, "positive")

    def test_control_context_preserves_spatial_arrangement_and_game_scope(self):
        prior = os.environ.get("ARC_AGI3_V8_CONTROL_SCOPE")
        try:
            os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = "game-a"
            a = np.array([[1, 0], [0, 2]], dtype=np.int64)
            b = np.array([[0, 1], [2, 0]], dtype=np.int64)
            sig_a = control_context_signature(a)
            self.assertNotEqual(sig_a, control_context_signature(b))
            os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = "game-b"
            self.assertNotEqual(sig_a, control_context_signature(a))
        finally:
            if prior is None:
                os.environ.pop("ARC_AGI3_V8_CONTROL_SCOPE", None)
            else:
                os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = prior

    def test_m6_key_keeps_full_consequence_signature(self):
        c1 = node(
            level=MemoryLevel.M5,
            memory_type=MemoryType.CONSEQUENCE,
            key_parts=(10, 11, 0x1234, 1),
        )
        c2 = node(
            level=MemoryLevel.M5,
            memory_type=MemoryType.CONSEQUENCE,
            key_parts=(10, 11, 0x11234, 1),
        )
        self.assertNotEqual(richer_outcome_key(c1), richer_outcome_key(c2))
        self.assertEqual(richer_outcome_key(c1)[1], 0x1234)
        self.assertEqual(richer_outcome_key(c2)[1], 0x11234)

    def test_world_model_does_not_merge_only_on_future_and_valence(self):
        a = node(
            level=MemoryLevel.M5,
            memory_type=MemoryType.CONSEQUENCE,
            key_parts=(1, 1, 100, 1),
        )
        b = node(
            level=MemoryLevel.M5,
            memory_type=MemoryType.CONSEQUENCE,
            key_parts=(2, 2, 200, 1),
        )
        self.assertEqual(WorldModelEstimator(min_consequences=2).propose((a, b)), ())
        c = node(
            level=MemoryLevel.M5,
            memory_type=MemoryType.CONSEQUENCE,
            key_parts=(2, 2, 100, 1),
        )
        self.assertEqual(len(WorldModelEstimator(min_consequences=2).propose((a, c))), 1)

    def test_multi_action_m7_is_formed_from_causal_path(self):
        m1a = node(
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=(10, 1, 101, 20),
        )
        m1b = node(
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=(20, 2, 102, 30),
        )
        outcome = node(
            level=MemoryLevel.M6,
            memory_type=MemoryType.OUTCOME,
            key_parts=(0, 102, 99),
        )
        edge = EdgeRecord(
            outcome.uid,
            int(RelationType.LEADS_TO),
            m1b.uid,
            1,
            1,
        )
        engine = CausalEvidenceGatedPromotionEngine(min_contingency_support=1)
        rows = _composite_candidates(engine, (m1a, m1b, outcome), (edge,), limit=8)
        self.assertTrue(rows)
        candidate = rows[0]
        self.assertTrue(int(candidate.key_parts[0]) & _SEQUENCE_MARKER)
        self.assertEqual(candidate.parents[0], outcome.uid)
        self.assertIn(m1a.uid, candidate.parents)
        self.assertIn(m1b.uid, candidate.parents)

    def test_composite_path_is_reconstructable_for_execution(self):
        m1a = node(
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=(10, 1, 101, 20),
        )
        m1b = node(
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=(20, 2, 102, 30),
        )
        outcome = node(
            level=MemoryLevel.M6,
            memory_type=MemoryType.OUTCOME,
            key_parts=(0, 102, 99),
        )
        edge = EdgeRecord(outcome.uid, int(RelationType.LEADS_TO), m1b.uid, 1, 1)
        engine = CausalEvidenceGatedPromotionEngine(min_contingency_support=1)
        candidate = _composite_candidates(
            engine, (m1a, m1b, outcome), (edge,), limit=8
        )[0]
        strategy = node(
            level=MemoryLevel.M7,
            memory_type=MemoryType.STRATEGY,
            key_parts=candidate.key_parts,
            uid=candidate.uid,
        )
        view = SimpleNamespace(
            _behavior_strategy_dependencies={strategy.uid: {m1a.uid, m1b.uid}},
            _node_by_uid={
                m1a.uid: m1a,
                m1b.uid: m1b,
                outcome.uid: outcome,
                strategy.uid: strategy,
            },
        )
        path = _path_for_composite(view, strategy, 10)
        self.assertEqual(tuple(int(row.key_parts[1]) for row in path), (1, 2))
        self.assertTrue(is_composite_strategy(strategy))

    def test_long_credit_and_future_option_horizon_are_installed(self):
        from v8 import primary_valence

        self.assertEqual(primary_valence._VALENCE_HORIZON, 4096)
        self.assertAlmostEqual(primary_valence._VALENCE_GAMMA, 0.995)
        self.assertEqual(FutureOptionEstimator().horizon, 8)

    def test_independent_repeated_preference_probes_remain_evidence(self):
        estimator = PreferenceEstimator(support_threshold=2)
        a = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2))
        b = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 3))
        kwargs = dict(
            outcome_a=a,
            outcome_b=b,
            context_bucket=0,
            chosen_outcome=a,
            both_reachable=True,
            preference_influenced=False,
        )
        self.assertTrue(estimator.record_probe(**kwargs))
        self.assertTrue(estimator.record_probe(**kwargs))
        self.assertEqual(estimator.evaluate()[0].clean_probe_count, 2)

    def test_runtime_metadata_marks_learning_capability_layer(self):
        self.assertEqual(
            v8.ContinuousMemoryRuntime.scientific_semantics_version,
            "v8.5-learning-capability",
        )


if __name__ == "__main__":
    unittest.main()
