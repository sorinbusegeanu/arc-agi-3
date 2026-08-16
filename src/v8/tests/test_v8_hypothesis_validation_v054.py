from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v8 import primary_valence as primary
from v8.arena import EdgeRecord, NodeRecord
from v8.hypothesis_validation_v054 import (
    TrajectoryReplanningTrialResult,
    _auto_outcome_holdout,
    _auto_transfer_trials,
    _observed_with_trajectory_replanning,
    _record_actor_results_with_validation,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.outcomes import OutcomeEquivalenceEstimator
from v8.transfer import TransferValidator
from v8.world_model import WorldModelEstimator


def node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key: tuple[int, ...],
    *,
    support: int = 4,
    valence_sum: float = 0.0,
    valence_weight: float = 0.0,
) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(level, memory_type, key),
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=int(support),
        significance_sum=float(support),
        prediction_error_sum=0.0,
        learning_value_sum=float(support),
        transfer_prior_sum=float(support),
        explanatory_sum=float(support),
        future_option_sum=float(key[-1] if key else 0) * float(support),
        score_weight=float(support),
        updated_watermark=10,
        game_mask=0,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
        primary_valence_sum=float(valence_sum),
        primary_valence_sq_sum=abs(float(valence_sum)),
        primary_valence_weight=float(valence_weight),
        positive_valence_count=float(valence_weight if valence_sum > 0 else 0.0),
        negative_valence_count=float(valence_weight if valence_sum < 0 else 0.0),
    )


def provenance(uid: MemoryUid, game_hash: int) -> EdgeRecord:
    return EdgeRecord(
        uid,
        int(RelationType.GAME_PROVENANCE),
        MemoryUid(0, int(game_hash)),
        1,
        10,
    )


def correspondence(source: MemoryUid, target: MemoryUid, score: float) -> EdgeRecord:
    return EdgeRecord(
        source,
        int(RelationType.TRANSFER_CORRESPONDENCE),
        target,
        1,
        10,
        score_sum=float(score),
        score_weight=1.0,
    )


class WorldModelFormationTests(unittest.TestCase):
    def test_world_model_groups_matching_consequence_structure_across_concepts(self) -> None:
        a = node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (10, 11, 111, 1))
        b = node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (20, 21, 111, 1))
        components = WorldModelEstimator().propose((a, b))
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0].consequences), {a.uid, b.uid})
        self.assertEqual(components[0].key_parts, (111, 1, 0))

    def test_world_model_keeps_opposite_primary_valence_profiles_separate(self) -> None:
        positive = node(
            MemoryLevel.M5,
            MemoryType.CONSEQUENCE,
            (10, 11, 111, 1),
            valence_sum=3.0,
            valence_weight=3.0,
        )
        negative = node(
            MemoryLevel.M5,
            MemoryType.CONSEQUENCE,
            (20, 21, 111, 1),
            valence_sum=-3.0,
            valence_weight=3.0,
        )
        self.assertEqual(WorldModelEstimator().propose((positive, negative)), ())


class AutomaticTransferValidationTests(unittest.TestCase):
    def test_structural_correspondence_no_longer_auto_validates_transfer(self) -> None:
        source = node(MemoryLevel.M4, MemoryType.CONCEPT, (1, 1))
        alternative = node(MemoryLevel.M4, MemoryType.CONCEPT, (2, 2))
        target = node(MemoryLevel.M4, MemoryType.CONCEPT, (3, 3))
        edges = (
            provenance(source.uid, 100),
            provenance(alternative.uid, 300),
            provenance(target.uid, 200),
            correspondence(source.uid, target.uid, 0.8),
            correspondence(alternative.uid, target.uid, 0.5),
        )

        class Peer:
            candidate_budget = 16

            def __init__(self) -> None:
                self.transfer = TransferValidator()
                self.calls: list[dict[str, object]] = []

            def record_transfer_trial(self, uid, **kwargs):
                self.calls.append({"uid": uid, **kwargs})
                return self.transfer.record_trial(uid, **kwargs)

        peer = Peer()
        _auto_transfer_trials(peer, (source, alternative, target), edges)
        self.assertEqual(peer.calls, [])
        self.assertEqual(peer.transfer.trials(source.uid), ())


class OutcomeHoldoutTests(unittest.TestCase):
    def test_structural_outcome_holdout_no_longer_emits_validated_evidence(self) -> None:
        a = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 1), support=8)
        b = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 2), support=8)
        edges = (provenance(a.uid, 100), provenance(b.uid, 200))

        class Peer:
            candidate_budget = 16

            def __init__(self) -> None:
                self.outcomes = OutcomeEquivalenceEstimator()
                self.rows: list[tuple[str, object, float, dict[str, object]]] = []

            def _fresh(self, *_args, **_kwargs):
                return True

            def _append_evidence(self, kind, row, value, **kwargs):
                self.rows.append((kind, row, value, kwargs))

        peer = Peer()
        _auto_outcome_holdout(peer, (a, b), edges)
        self.assertEqual(peer.rows, [])


class TrajectoryReplanningTests(unittest.TestCase):
    def setUp(self) -> None:
        import v8.hypothesis_validation_v054 as module

        module._PENDING_REPLAN_TRIALS.clear()
        module._LAST_OBSERVATION_TERMINAL = False
        self._prior_capture = primary._CAPTURE_ACTIVE
        primary._CAPTURE_ACTIVE = True

    def tearDown(self) -> None:
        import v8.hypothesis_validation_v054 as module

        module._PENDING_REPLAN_TRIALS.clear()
        module._LAST_OBSERVATION_TERMINAL = False
        primary._CAPTURE_ACTIVE = self._prior_capture

    def test_replanning_recovery_can_succeed_after_more_than_one_action(self) -> None:
        import v8.hypothesis_validation_v054 as module

        target = MemoryUid(9, 9)
        trial = TrajectoryReplanningTrialResult(MemoryUid(1, 1), MemoryUid(2, 2), target, False)
        self.assertFalse(trial.resolved)
        with patch.object(module, "_BASE_ACTOR_OBSERVED", return_value=(target, target)):
            _observed_with_trajectory_replanning(terminal_polarity=0)
        self.assertTrue(trial.resolved)
        self.assertTrue(trial.recovery_succeeded)
        self.assertEqual(trial.actions, 2)

    def test_terminal_boundary_resolves_unrecovered_replan_as_failure(self) -> None:
        import v8.hypothesis_validation_v054 as module

        trial = TrajectoryReplanningTrialResult(
            MemoryUid(1, 1), MemoryUid(2, 2), MemoryUid(9, 9), False
        )
        with patch.object(module, "_BASE_ACTOR_OBSERVED", return_value=(MemoryUid.zero(), MemoryUid.zero())):
            _observed_with_trajectory_replanning(terminal_polarity=-1)
        self.assertTrue(trial.resolved)
        self.assertFalse(trial.recovery_succeeded)


class PreferenceAggregationTests(unittest.TestCase):
    def test_v054_helper_can_still_be_called_for_legacy_replay(self) -> None:
        import v8.hypothesis_validation_v054 as module

        preferred = MemoryUid(10, 10)
        other = MemoryUid(20, 20)
        preference = primary.PrimaryValencePreference(preferred, other, 123456, 0.9)
        result = SimpleNamespace(
            actor_id=1,
            game_id="tt01",
            replanning_trials=(),
            primary_valence_preferences=(preference,),
        )

        class Peers:
            def __init__(self) -> None:
                self.calls = []

            def record_preference_probe(self, **kwargs):
                self.calls.append(kwargs)
                return True

        runtime = SimpleNamespace(peers=Peers())
        with patch.object(module, "_BASE_RUNTIME_RECORD_RESULTS", return_value=None):
            _record_actor_results_with_validation(runtime, (result,))
        self.assertEqual(len(runtime.peers.calls), 1)
        call = runtime.peers.calls[0]
        self.assertEqual(call["context_bucket"], 0)
        self.assertEqual(call["chosen_outcome"], preferred)
        self.assertFalse(call["preference_influenced"])


if __name__ == "__main__":
    unittest.main()