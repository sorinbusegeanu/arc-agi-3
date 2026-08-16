from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    stable_u64,
)
from v8.publication import ActionScore, LiveReadView
from v8.restart_memory_v815 import (
    _CONTROL_SCOPE_ENV,
    _augment_scores,
    _build_restart_indexes,
    _credit_session,
    _phase_variant,
)
from v8.restart_memory_v815_fixups import _reporting_worker_after_first_progress
from v8.runtime import V8RuntimeConfig
from v8.structural_events import NormalizedPrimitive, StructuralFact
from v8.trajectory_optimizer_v814 import ReplayAnchor, TrajectoryTarget, ValidatedTrajectory


def _m1(
    key,
    *,
    support=5,
    valence=1.0,
    valence_weight=5.0,
) -> NodeRecord:
    uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, tuple(key))
    return NodeRecord(
        uid=uid,
        fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, tuple(key)),
        level=int(MemoryLevel.M1),
        memory_type=int(MemoryType.CONTINGENCY),
        key_parts=tuple(key),
        support_count=int(support),
        significance_sum=float(support),
        prediction_error_sum=0.0,
        learning_value_sum=float(support),
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=0.0,
        score_weight=float(support),
        updated_watermark=1,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.VALIDATED),
        primary_valence_sum=float(valence) * float(valence_weight),
        primary_valence_sq_sum=float(valence * valence) * float(valence_weight),
        primary_valence_weight=float(valence_weight),
        positive_valence_count=float(valence_weight if valence > 0 else 0.0),
        negative_valence_count=float(valence_weight if valence < 0 else 0.0),
    )


class _FakeView:
    def __init__(self, nodes, edges=(), strategies=(), dependencies=None):
        self._strategy_version = (2, 2)
        self._node_by_uid = {row.uid: row for row in nodes}
        self._strategy_by_context = {1: list(strategies)} if strategies else {}
        self._strategy_fallback = []
        self._behavior_strategy_dependencies = dependencies or {}
        self._v815_restart_index_key = None
        self._v815_same_game_action_priors = {}
        self._v815_normalized_action_priors = {}
        self._v815_same_game_strategies = ()
        self._v815_session_action_priors = {}
        self._v815_session_trajectory = []
        self._v815_score_origins = {}
        self._edges = tuple(edges)

    def edge_records(self):
        return self._edges


class RestartPolicyIndexTests(unittest.TestCase):
    def test_same_game_success_memory_drives_action_when_exact_context_misses(self) -> None:
        game = "restart-game"
        game_hash = stable_u64(game, person=b"v8-game")
        remembered = _m1((111, 1, 222, 333))
        provenance = EdgeRecord(
            remembered.uid,
            int(RelationType.GAME_PROVENANCE),
            MemoryUid(0, game_hash),
            5,
            1,
        )
        view = _FakeView((remembered,), (provenance,))
        with patch.dict(os.environ, {_CONTROL_SCOPE_ENV: game}, clear=False):
            rows = _augment_scores(
                view,
                (
                    ActionScore(1, 0, 0.0, 0),
                    ActionScore(2, 0, 0.0, 0),
                ),
            )
        self.assertGreater(rows[0].support_count, 0)
        self.assertGreater(rows[0].score, 0.0)
        self.assertEqual(rows[1].support_count, 0)
        self.assertEqual(view._v815_score_origins[1], "same_game_m1")

    def test_normalized_memory_contributes_generic_action_prior(self) -> None:
        grounded = _m1((101, 3, 202, 303))
        token = StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, 55).token
        normalized = _m1((token,), support=8, valence=0.8, valence_weight=8.0)
        explains = EdgeRecord(
            normalized.uid,
            int(RelationType.EXPLAINS),
            grounded.uid,
            4,
            1,
        )
        view = _FakeView((grounded, normalized), (explains,))
        with patch.dict(os.environ, {_CONTROL_SCOPE_ENV: "different-game"}, clear=False):
            rows = _augment_scores(view, (ActionScore(3, 0, 0.0, 0),))
        self.assertGreater(rows[0].support_count, 0)
        self.assertGreater(rows[0].score, 0.0)
        self.assertEqual(view._v815_score_origins[3], "normalized_m1")

    def test_forced_random_is_not_overridden_by_restored_memory(self) -> None:
        view = _FakeView((_m1((1, 1, 2, 3)),))
        rows = _augment_scores(view, (ActionScore(1, 0, 0.0, 0),), force_random=True)
        self.assertEqual(rows[0].support_count, 0)
        self.assertEqual(rows[0].score, 0.0)

    def test_same_game_m7_is_indexed_without_transferability_requirement(self) -> None:
        game = "same-game"
        game_hash = stable_u64(game, person=b"v8-game")
        dependency = _m1((10, 2, 20, 30))
        strategy_uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (2, 8, 9, 10))
        strategy_node = NodeRecord(
            uid=strategy_uid,
            fingerprint=1,
            level=int(MemoryLevel.M7),
            memory_type=int(MemoryType.STRATEGY),
            key_parts=(2, 8, 9, 10),
            support_count=10,
            significance_sum=5.0,
            prediction_error_sum=0.0,
            learning_value_sum=5.0,
            transfer_prior_sum=0.0,
            explanatory_sum=0.0,
            future_option_sum=0.0,
            score_weight=10.0,
            updated_watermark=1,
            cognitive_state=int(CognitiveState.ACTIVE),
            validation_state=int(ValidationState.TESTED),
            success_sum=9.0,
            cost_sum=18.0,
            attempt_weight=10.0,
        )
        strategy = SimpleNamespace(
            action_id=2,
            outcome_uid=MemoryUid(8, 9),
            strategy_uid=strategy_uid,
            support=10,
            reliability=0.9,
            mean_cost=1.8,
            context_bucket=10,
            probationary=False,
            transferable=False,
        )
        provenance = EdgeRecord(
            dependency.uid,
            int(RelationType.GAME_PROVENANCE),
            MemoryUid(0, game_hash),
            5,
            1,
        )
        view = _FakeView(
            (dependency, strategy_node),
            (provenance,),
            (strategy,),
            {strategy_uid: {dependency.uid}},
        )
        with patch.dict(os.environ, {_CONTROL_SCOPE_ENV: game}, clear=False):
            _build_restart_indexes(view)
        self.assertEqual(tuple(row.strategy_uid for row in view._v815_same_game_strategies), (strategy_uid,))
        self.assertFalse(strategy.transferable)


class SessionAndTrajectoryReuseTests(unittest.TestCase):
    def test_success_credit_persists_as_session_action_memory(self) -> None:
        view = SimpleNamespace(
            _v815_session_trajectory=[(10, 1), (11, 2), (12, 1)],
            _v815_session_action_priors={},
        )
        _credit_session(view, success=True, failure=False)
        self.assertEqual(view._v815_session_trajectory, [])
        self.assertGreater(view._v815_session_action_priors[1][0], 0)
        self.assertGreater(view._v815_session_action_priors[1][1], 0.0)
        retained = dict(view._v815_session_action_priors)
        _credit_session(view, success=False, failure=True)
        self.assertEqual(view._v815_session_action_priors, retained)

    def test_validated_trajectory_reuses_level_phase_despite_prefix_and_seed_change(self) -> None:
        from v8 import restart_memory_v815 as restart
        from v8 import trajectory_optimizer_v814 as optimizer

        uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3, 4))
        row = ValidatedTrajectory(
            "variant",
            ReplayAnchor("world", 7, (1, 2, 3, 4), None),
            TrajectoryTarget(2, "LEVEL"),
            (5, 6),
            uid,
            MemoryUid.zero(),
            MemoryUid.zero(),
            8,
            "DELETE_SEGMENT",
        )
        view = SimpleNamespace(
            _v814_variants=(row,),
            _v814_attempted_variants=set(),
        )
        prior_level = restart._CURRENT_LEVELS_COMPLETED
        prior_source = optimizer._CAPTURE_SOURCE_ID
        prior_seed = optimizer._CAPTURE_SEED
        prior_history = list(optimizer._ACTOR_ACTION_HISTORY)
        try:
            restart._CURRENT_LEVELS_COMPLETED = 1
            optimizer._CAPTURE_SOURCE_ID = "world"
            optimizer._CAPTURE_SEED = 999
            optimizer._ACTOR_ACTION_HISTORY[:] = [99]
            with patch.object(optimizer, "_refresh_view_variants", lambda _view: None):
                self.assertIs(_phase_variant(view, (5, 6, 7)), row)
                restart._CURRENT_LEVELS_COMPLETED = 0
                self.assertIsNone(_phase_variant(view, (5, 6, 7)))
        finally:
            restart._CURRENT_LEVELS_COMPLETED = prior_level
            optimizer._CAPTURE_SOURCE_ID = prior_source
            optimizer._CAPTURE_SEED = prior_seed
            optimizer._ACTOR_ACTION_HISTORY[:] = prior_history


class ReporterStartupTests(unittest.TestCase):
    def test_reporter_suppresses_only_preprogress_zero_line(self) -> None:
        from v8.actor import ActorProgress

        events = queue.Queue()
        output = queue.Queue()
        stop = threading.Event()
        thread = threading.Thread(
            target=_reporting_worker_after_first_progress,
            kwargs={
                "event_queue": events,
                "stop_event": stop,
                "watermark": object(),
                "actors": ((1, "a"), (2, "b")),
                "interval_seconds": 0.05,
                "output_queue": output,
            },
            daemon=True,
        )
        thread.start()
        try:
            time.sleep(0.08)
            self.assertTrue(output.empty())
            events.put(ActorProgress(1, "a", 10, 0, 0, 0))
            deadline = time.monotonic() + 0.3
            while output.empty() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(output.empty())
        finally:
            stop.set()
            thread.join(timeout=1.0)


class RestartSnapshotIntegrationTests(unittest.TestCase):
    def test_snapshot_memory_is_usable_after_context_hash_changes(self) -> None:
        game = "restart-integration"
        game_hash = stable_u64(game, person=b"v8-game")
        with tempfile.TemporaryDirectory() as tmp:
            config = V8RuntimeConfig.from_path(
                tmp,
                shards=1,
                stage_workers=1,
                enable_snapshots=True,
                restore=True,
                enable_peers=False,
                snapshot_interval_seconds=3600,
                node_capacity_per_shard=256,
                edge_capacity_per_shard=512,
                action_capacity_per_shard=64,
            )
            runtime = v8.ContinuousMemoryRuntime(config)
            runtime.start()
            key = (1001, 4, 2002, 3003)
            uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
            runtime.submit_proposal(
                MemoryProposal(
                    uid=uid,
                    fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
                    event_id=EventId.from_producer(77, 1),
                    watermark=1,
                    level=MemoryLevel.M1,
                    memory_type=MemoryType.CONTINGENCY,
                    key_parts=key,
                    support_delta=8,
                    significance_sum=8.0,
                    learning_value_sum=8.0,
                    score_weight=8.0,
                    source_game_hash=game_hash,
                    cognitive_state=int(CognitiveState.ACTIVE),
                    validation_state=int(ValidationState.VALIDATED),
                    primary_valence_sum=8.0,
                    primary_valence_sq_sum=8.0,
                    primary_valence_weight=8.0,
                    positive_valence_count=8.0,
                )
            )
            runtime.wait_quiescent(timeout=20)
            runtime.close(normal=True, timeout=30)

            restored = v8.ContinuousMemoryRuntime(config)
            restored.start()
            try:
                with patch.dict(os.environ, {_CONTROL_SCOPE_ENV: game}, clear=False):
                    view = LiveReadView(restored.shard_descriptors)
                    try:
                        scores = view.score_actions(999999, (4, 5))
                    finally:
                        view.close()
                by_action = {row.action_id: row for row in scores}
                self.assertGreater(by_action[4].support_count, 0)
                self.assertGreater(by_action[4].score, 0.0)
                self.assertEqual(by_action[5].support_count, 0)
            finally:
                restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()
