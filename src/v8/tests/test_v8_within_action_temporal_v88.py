from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import v8
from v7.environment.arc_adapter import ArcGridEnvironment
from v8 import development
from v8 import model
from v8 import normalized_memory_v086 as normalized
from v8 import within_action_temporal_v88 as temporal
from v8.model import EventId, MemoryLevel, MemoryType, MemoryUid
from v8.structural_events import NormalizedPrimitive, normalized_fact_kind


class _MultiFrameEngine:
    def __init__(self, frames) -> None:
        self.frames = tuple(np.asarray(frame, dtype=np.int64) for frame in frames)
        self.step_calls = 0
        self._reset = SimpleNamespace(
            frame=np.zeros_like(self.frames[-1]),
            state="NOT_FINISHED",
            levels_completed=0,
            available_actions=[1, 2, 3],
        )

    def reset(self):
        return self._reset

    def step(self, _action):
        self.step_calls += 1
        return SimpleNamespace(
            frame=[frame.copy() for frame in self.frames],
            state="NOT_FINISHED",
            levels_completed=0,
            available_actions=[1, 2, 3],
        )


class AdapterExposureTests(unittest.TestCase):
    def test_multiframe_response_exposes_all_frames_and_returns_settled_frame(self) -> None:
        f1 = np.array([[1, 0], [0, 0]], dtype=np.int64)
        f2 = np.array([[0, 1], [0, 0]], dtype=np.int64)
        f3 = np.array([[0, 0], [0, 1]], dtype=np.int64)
        engine = _MultiFrameEngine((f1, f2, f3))
        env = ArcGridEnvironment(game_id="fake", env_factory=lambda **_: engine)

        result = env.step(1)
        np.testing.assert_array_equal(result, f3)
        np.testing.assert_array_equal(env.frame, f3)
        self.assertEqual(len(env.all_frames), 3)
        self.assertEqual(len(env.animation_frames), 2)
        for actual, expected in zip(env.all_frames, (f1, f2, f3)):
            np.testing.assert_array_equal(actual, expected)

        # Returned arrays are detached from adapter state.
        exposed = env.all_frames
        exposed[0][0, 0] = 99
        self.assertEqual(int(env.all_frames[0][0, 0]), 1)

        step_result = env.cognitive_step_result()
        self.assertIsNotNone(step_result)
        self.assertEqual(step_result.within_action_trace.frame_count, 3)
        self.assertEqual(engine.step_calls, 1)
        self.assertEqual(env._v88_last_temporal_descriptor.transition_count, 3)

    def test_single_frame_keeps_settled_behavior_and_adds_no_temporal_fact(self) -> None:
        final = np.array([[0, 1], [0, 0]], dtype=np.int64)
        engine = _MultiFrameEngine((final,))
        env = ArcGridEnvironment(game_id="fake", env_factory=lambda **_: engine)
        env.reset()
        result = env.step(1)
        np.testing.assert_array_equal(result, final)
        self.assertEqual(len(env.all_frames), 1)
        self.assertEqual(env._v88_last_temporal_descriptor.transition_count, 1)
        _history, _actions, _elapsed, facts = normalized._LAST_ACTOR_EXTRAS
        temporal_facts = [
            token
            for token in facts
            if normalized_fact_kind(int(token)) == NormalizedPrimitive.AUTONOMOUS_CHANGE
        ]
        self.assertEqual(temporal_facts, [])


class TraceDerivationTests(unittest.TestCase):
    def test_temporal_trace_is_order_sensitive_and_records_internal_origin(self) -> None:
        before = np.array([[1, 0, 0]], dtype=np.int64)
        f1 = np.array([[0, 1, 0]], dtype=np.int64)
        f2 = np.array([[0, 0, 1]], dtype=np.int64)
        forward = temporal.derive_temporal_trace(before, (f1, f2))
        reverse = temporal.derive_temporal_trace(before, (f2, f1))
        self.assertNotEqual(forward.trace_signature, reverse.trace_signature)
        self.assertNotEqual(forward.family_signature, reverse.family_signature)
        self.assertEqual(forward.micro_transitions[0].origin, "ACTION_TRIGGERED")
        self.assertEqual(forward.micro_transitions[1].origin, "INTERNAL_EVOLUTION")

    def test_temporal_prediction_violation_activates_only_after_support(self) -> None:
        tracker = temporal.TemporalPredictionTracker(minimum_support=2)
        self.assertEqual(tracker.prediction_error(1, 10, 3, 100), 0.0)
        tracker.observe(1, 10, 3, 100)
        self.assertEqual(tracker.prediction_error(1, 10, 3, 200), 0.0)
        tracker.observe(1, 10, 3, 100)
        self.assertEqual(tracker.prediction_error(1, 10, 3, 100), 0.0)
        self.assertEqual(tracker.prediction_error(1, 10, 3, 200), 1.0)


class CodecAndMemoryTests(unittest.TestCase):
    @staticmethod
    def _event(**temporal_fields):
        values = dict(
            temporal_trace_signature=0,
            temporal_transition_count=0,
            temporal_family_signature=0,
            carrier_lineage_signature=0,
            temporal_prediction_error=0.0,
        )
        values.update(temporal_fields)
        return model.ExperienceEvent(
            EventId(1, 2),
            3,
            4,
            5,
            6,
            7,
            8,
            3,
            9,
            10,
            11,
            0.0,
            2,
            0,
            12,
            13,
            0.0,
            **values,
        )

    def test_experience_codec_round_trip_and_legacy_restore(self) -> None:
        event = self._event(
            temporal_trace_signature=101,
            temporal_transition_count=3,
            temporal_family_signature=202,
            carrier_lineage_signature=303,
            temporal_prediction_error=1.0,
        )
        decoded = model.decode_experience(model.encode_experience(event))
        self.assertEqual(decoded.temporal_trace_signature, 101)
        self.assertEqual(decoded.temporal_transition_count, 3)
        self.assertEqual(decoded.temporal_family_signature, 202)
        self.assertEqual(decoded.carrier_lineage_signature, 303)
        self.assertEqual(decoded.temporal_prediction_error, 1.0)

        legacy_event = temporal._BASE_EXPERIENCE(
            EventId(1, 2), 3, 4, 5, 6, 7, 8, 3, 9, 10, 11, 0.0, 2, 0, 12, 13, 0.0
        )
        restored = model.decode_experience(temporal._BASE_ENCODE_EXPERIENCE(legacy_event))
        self.assertEqual(restored.temporal_transition_count, 0)
        self.assertEqual(restored.temporal_trace_signature, 0)

    def test_normalized_pipeline_round_trip_preserves_temporal_fields(self) -> None:
        event = self._event(
            temporal_trace_signature=101,
            temporal_transition_count=4,
            temporal_family_signature=202,
            carrier_lineage_signature=303,
            temporal_prediction_error=0.5,
        )
        row = normalized.V86PipelineEvent(
            event,
            MemoryUid.zero(),
            0,
            1,
            99,
            100,
            2,
            (),
        )
        decoded = model.decode_pipeline(model.encode_pipeline(row))
        self.assertEqual(decoded.experience.temporal_trace_signature, 101)
        self.assertEqual(decoded.experience.temporal_transition_count, 4)
        self.assertEqual(decoded.experience.temporal_family_signature, 202)
        self.assertEqual(decoded.experience.carrier_lineage_signature, 303)
        self.assertEqual(decoded.history_signature, 99)

    def test_temporal_evidence_strengthens_m0_m1_without_changing_m6_identity(self) -> None:
        plain = self._event()
        temporal_event = self._event(
            temporal_trace_signature=101,
            temporal_transition_count=3,
            temporal_family_signature=202,
            temporal_prediction_error=1.0,
        )
        plain_pipe = model.PipelineEvent(plain)
        temporal_pipe = model.PipelineEvent(temporal_event)
        m1_plain = development.derive_proposal(MemoryLevel.M1, plain_pipe)
        m1_temporal = development.derive_proposal(MemoryLevel.M1, temporal_pipe)
        self.assertGreater(m1_temporal.prediction_error_sum, m1_plain.prediction_error_sum)
        self.assertGreater(m1_temporal.explanatory_sum, m1_plain.explanatory_sum)
        self.assertEqual(
            development._key_for(MemoryLevel.M6, plain_pipe),
            development._key_for(MemoryLevel.M6, temporal_pipe),
        )

    def test_temporal_facts_feed_existing_m2_and_m3_promotion(self) -> None:
        before = np.array([[1, 0, 0]], dtype=np.int64)
        right1 = np.array([[0, 1, 0]], dtype=np.int64)
        right2 = np.array([[0, 0, 1]], dtype=np.int64)
        grow1 = np.array([[1, 1, 0]], dtype=np.int64)
        grow2 = np.array([[1, 1, 1]], dtype=np.int64)
        first = temporal.temporal_fact_tokens(
            temporal.derive_temporal_trace(before, (right1, right2))
        )[0]
        second = temporal.temporal_fact_tokens(
            temporal.derive_temporal_trace(before, (grow1, grow2))
        )[0]
        self.assertNotEqual(first, second)
        self.assertEqual(normalized_fact_kind(first), NormalizedPrimitive.AUTONOMOUS_CHANGE)
        self.assertEqual(normalized_fact_kind(second), NormalizedPrimitive.AUTONOMOUS_CHANGE)

        def node(token, support=3):
            return SimpleNamespace(
                uid=MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (int(token),)),
                level=int(MemoryLevel.M1),
                memory_type=int(MemoryType.CONTINGENCY),
                key_parts=(int(token),),
                support_count=int(support),
                future_option_delta=0.0,
            )

        m1a, m1b = node(first), node(second)

        class M2Engine:
            min_contingency_support = 2
            min_family_members = 2
            min_family_compression = 0.0

            @staticmethod
            def _admissible(_row):
                return True

        m2 = normalized._normalized_m2_candidates(M2Engine(), (m1a, m1b), limit=8)
        self.assertTrue(m2)
        family_candidate = m2[0]
        family = SimpleNamespace(
            uid=family_candidate.uid,
            level=int(MemoryLevel.M2),
            memory_type=int(MemoryType.FAMILY),
            key_parts=family_candidate.key_parts,
            support_count=6,
            future_option_delta=0.0,
        )

        class M3Engine:
            min_carrier_family_support = 1
            min_carrier_persistence = 2

            @staticmethod
            def _admissible(_row):
                return True

            @staticmethod
            def _children(_edges):
                return {family.uid: (m1a.uid,)}

            @staticmethod
            def _future_bucket(_value):
                return 0

        m3 = normalized._normalized_m3_candidates(
            M3Engine(), (m1a, family), (), limit=8
        )
        self.assertTrue(m3)
        self.assertEqual(int(m3[0].level), int(MemoryLevel.M3))

    def test_animation_length_never_inflates_strategy_action_cost(self) -> None:
        from v8 import trajectory_optimizer_v814 as optimizer

        anchor = optimizer.ReplayAnchor("fake", 0)
        target = optimizer.TrajectoryTarget(1, "LEVEL")
        source = optimizer.SuccessfulTrajectory("x", anchor, target, (3,))
        self.assertEqual(source.cost, 1)


class VersionTests(unittest.TestCase):
    def test_runtime_is_marked_v88(self) -> None:
        from v8.runtime_v82 import V82ContinuousMemoryRuntime

        self.assertEqual(V82ContinuousMemoryRuntime.within_action_temporal_version, "v8.8")


if __name__ == "__main__":
    unittest.main()
