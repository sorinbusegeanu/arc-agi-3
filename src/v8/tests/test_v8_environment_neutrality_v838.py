from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import v8  # noqa: F401
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import environment_neutrality_v837 as v837
from v8 import environment_neutrality_v838 as v838
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_optimizer_convergence_v836 as convergence
from v8 import trajectory_optimizer_v814 as optimizer
from v8.environment_contract import BoundaryEvent, BoundaryScope, EnvironmentTransition
from v8.model import MemoryUid
from v8.publication import ActionScore


class EnvironmentNeutralityV838Tests(unittest.TestCase):
    def test_m6_participates_in_trajectory_candidate_and_frontier_identity(self):
        anchor = optimizer.ReplayAnchor("env", 0, (), None)
        uid_a = MemoryUid(10, 11)
        uid_b = MemoryUid(20, 21)
        target_a = optimizer.TrajectoryTarget(
            0, "BOUNDARY", BoundaryScope.EPISODE.value, +1, False, uid_a.hi, uid_a.lo
        )
        target_b = optimizer.TrajectoryTarget(
            0, "BOUNDARY", BoundaryScope.EPISODE.value, +1, False, uid_b.hi, uid_b.lo
        )
        actions = (1, 2)
        source_a = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target_a, actions), anchor, target_a, actions,
            target_outcome_uid=uid_a,
        )
        source_b = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target_b, actions), anchor, target_b, actions,
            target_outcome_uid=uid_b,
        )
        self.assertNotEqual(source_a.trajectory_id, source_b.trajectory_id)
        self.assertNotEqual(
            optimizer._candidate_id(source_a, "DELETE_ACTION", (2,)),
            optimizer._candidate_id(source_b, "DELETE_ACTION", (2,)),
        )
        self.assertNotEqual(
            optimizer._frontier_key(anchor, target_a),
            optimizer._frontier_key(anchor, target_b),
        )

    def test_submit_normalizes_separate_target_outcome_into_identity(self):
        root = Path(tempfile.mkdtemp())
        service = optimizer.TrajectoryOptimizationService(
            root, validator=lambda _candidate: SimpleNamespace(success=False)
        )
        uid = MemoryUid(100, 200)
        anchor = optimizer.ReplayAnchor("env", 0, (), None)
        plain = optimizer.TrajectoryTarget(
            0, "BOUNDARY", BoundaryScope.EPISODE.value, +1, False
        )
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, plain, (1, 2)),
            anchor, plain, (1, 2), target_outcome_uid=uid,
        )
        self.assertTrue(service.submit_trajectory(source))
        queued = service._sources.get_nowait()
        self.assertEqual((queued.target.outcome_hi, queued.target.outcome_lo), (uid.hi, uid.lo))
        self.assertEqual(queued.target_outcome_uid, uid)
        self.assertEqual(
            queued.trajectory_id,
            optimizer._trajectory_id(queued.anchor, queued.target, queued.actions),
        )
        service._sources.task_done()

    def test_m6_replay_requires_outcome_and_boundary_not_boundary_alone(self):
        uid = MemoryUid(300, 400)

        class SymbolicEnv:
            def __init__(self):
                self.event = BoundaryEvent()
                self.last_action = 0

            def reset(self):
                self.event = BoundaryEvent()
                self.last_action = 0

            def observe(self):
                return self.last_action

            def available_actions(self):
                return (2, 3)

            def step(self, action):
                self.last_action = int(action)
                self.event = BoundaryEvent(BoundaryScope.EPISODE, +1, False)
                return self.last_action

            def cognitive_boundary_event(self):
                return self.event

            def cognitive_context_signature(self):
                return 77

            def cognitive_transition_signature(self, _before, after):
                return 1000 + int(after)

        service = SimpleNamespace(
            _v818_prefix_for=lambda _candidate: (),
            _v838_outcome_matcher=lambda _env, _ctx, action, _outcome, target: (
                int(action) == 2 and target == uid
            ),
        )
        target = optimizer.TrajectoryTarget(
            0, "BOUNDARY", BoundaryScope.EPISODE.value, +1, False, uid.hi, uid.lo
        )
        anchor = optimizer.ReplayAnchor("symbolic", 0, (), None)
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (2,)),
            anchor, target, (2,), target_outcome_uid=uid,
        )
        validator = v837._EnvironmentReplayValidator(service, "symbolic")
        validator._environment = lambda _seed, _root: SymbolicEnv()

        good = optimizer.TrajectoryCandidate(
            optimizer._candidate_id(source, "DIRECT", (2,)), source, "DIRECT", (2,), 0, 0
        )
        bad = optimizer.TrajectoryCandidate(
            optimizer._candidate_id(source, "DIRECT", (3,)), source, "DIRECT", (3,), 0, 0
        )
        self.assertTrue(validator._trial(good, 0, ())[0])
        bad_result = validator._trial(bad, 0, ())
        self.assertFalse(bad_result[0])
        self.assertEqual(bad_result[2], "outcome_not_preserved")

    def test_target_minimization_anchor_check_passes_trajectory_source(self):
        from v8 import trajectory_target_minimization_v820 as minimization

        uid = MemoryUid(500, 600)

        class SymbolicEnv:
            def reset(self):
                return None

            def available_actions(self):
                return (2, 3)

            def cognitive_boundary_event(self):
                return BoundaryEvent()

        service = SimpleNamespace(_v818_prefix_for=lambda _candidate: ())
        target = optimizer.TrajectoryTarget(
            0, "BOUNDARY", BoundaryScope.EPISODE.value, +1, False, uid.hi, uid.lo
        )
        anchor = optimizer.ReplayAnchor("symbolic", 0, (), None)
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (2, 3)),
            anchor,
            target,
            (2, 3),
            target_outcome_uid=MemoryUid.zero(),
        )
        candidate = minimization._candidate(
            optimizer, source, minimization._TARGET_MINIMIZE, source.actions
        )
        validator = v837._EnvironmentReplayValidator(service, "symbolic")
        validator._environment = lambda _seed, _root: SymbolicEnv()

        self.assertEqual(
            minimization._available_actions_at_anchor(validator, candidate),
            (2, 3),
        )

    def test_adapter_transition_overrides_conflicting_legacy_kwargs(self):
        transition = EnvironmentTransition(
            0, 0, 9, (9,), (9,), None,
            BoundaryEvent(BoundaryScope.EPISODE, -1, False),
            12, 12, False,
        )
        semantics = v837._transition_semantics(
            {
                "environment_transition": transition,
                "action": 9,
                "terminal_state": "WIN",
                "changed_cells": 100,
                "before_context": 1,
                "after_context": 2,
            }
        )
        self.assertTrue(semantics.terminal_failure)
        self.assertFalse(semantics.successful_boundary)
        self.assertFalse(semantics.productive)

    def test_generic_optimized_solution_does_not_manufacture_win(self):
        root = Path(tempfile.mkdtemp())
        service = optimizer.TrajectoryOptimizationService(
            root, validator=lambda _candidate: SimpleNamespace(success=False)
        )
        uid = MemoryUid(500, 600)
        anchor = optimizer.ReplayAnchor("symbolic", 0, (), None)
        target = optimizer.TrajectoryTarget(
            0, "BOUNDARY", BoundaryScope.EPISODE.value, +1, False, uid.hi, uid.lo
        )
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, (7, 8, 9)),
            anchor, target, (7, 8, 9), target_outcome_uid=uid,
        )
        candidate = optimizer.TrajectoryCandidate(
            optimizer._candidate_id(source, "DELETE_ACTION", (7, 8)),
            source, "DELETE_ACTION", (7, 8), 2, 1,
        )
        validated = optimizer.ValidatedTrajectory(
            candidate.candidate_id, anchor, target, candidate.actions,
            MemoryUid.zero(), uid, MemoryUid.zero(), source.cost,
            candidate.edit_kind, 2, 2,
        )
        result = SimpleNamespace(attempts=2, successes=2)
        with mock.patch.object(convergence, "_replay_full_win_levels", return_value=((7, 8),)):
            self.assertTrue(inspection._publish_optimized_solution(service, candidate, result, validated))

        payload = json.loads(service.best_successful_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["target_schema_version"], 2)
        record = payload["environments"]["symbolic"]
        self.assertNotIn("terminal_state", record)
        self.assertEqual(record["target"]["kind"], "OUTCOME")
        self.assertEqual(record["target"]["outcome_uid"], [uid.hi, uid.lo])
        self.assertEqual(record["total_cost"], 2)

    def test_legacy_full_win_scope_is_migrated_on_restore(self):
        coordinator = v819.AdaptiveLearningCoordinator()
        coordinator.load_state(
            {
                "version": 1,
                "games": ["env"],
                "game_won": {"env": True},
                "game_level_states": [
                    {
                        "game_id": "env",
                        "level": 1_000_000_000,
                        "state": "SOLVED_OPTIMIZING",
                        "first_success_generation": 2,
                    }
                ],
            }
        )
        new_key = v838._episode_scope_key()
        self.assertIn(("env", new_key), coordinator._records)
        self.assertNotIn(("env", 1_000_000_000), coordinator._records)
        self.assertEqual(coordinator.game_state("env").value, "SOLVED_OPTIMIZING")
        self.assertEqual(coordinator.state_dict()["scope_schema_version"], 2)

    def test_old_optimizer_identity_state_releases_seen_and_attempted_ids(self):
        root = Path(tempfile.mkdtemp())
        service = optimizer.TrajectoryOptimizationService(
            root, validator=lambda _candidate: SimpleNamespace(success=False)
        )
        service.load_state(
            {
                "version": 2,
                "seen_sources": ["old-source"],
                "attempted": ["old-candidate"],
                "validated": [],
                "metrics": {},
                "best_prefixes": {},
            }
        )
        self.assertNotIn("old-source", service._seen_sources)
        self.assertNotIn("old-candidate", service._attempted)
        self.assertEqual(service.state_dict()["identity_schema_version"], 2)

    def test_explicit_transfer_score_can_use_grounded_m1n(self):
        fake_view = SimpleNamespace()
        base_rows = (ActionScore(1, 0, 0.0, 0), ActionScore(2, 0, 0.0, 0))
        prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.TRANSFER.value
        try:
            with mock.patch.object(v838, "_BASE_V829_SCORE", return_value=base_rows), mock.patch.object(
                v837, "_current_game_id", return_value="target"
            ), mock.patch.object(
                v837, "_grounded_transfer_index",
                return_value=({}, {1: ((0.8, None, "M1N_GROUNDED"),)}),
            ):
                rows = v838._score_grounded_transfer_v838(fake_view, 10, (1, 2))
        finally:
            if prior_mode is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior_mode
        by_action = {row.action_id: row for row in rows}
        self.assertGreater(by_action[1].support_count, 0)
        self.assertGreater(by_action[1].score, 0.0)
        self.assertEqual(by_action[2].support_count, 0)


if __name__ == "__main__":
    unittest.main()
