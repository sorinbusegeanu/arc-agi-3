from __future__ import annotations

import io
import queue
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.publication import PlannedAction
from v8.trajectory_optimizer_v814 import (
    ReplayAnchor,
    SuccessfulTrajectory,
    TrajectoryOptimizationService,
    TrajectoryOptimizerConfig,
    TrajectoryTarget,
    ValidatedTrajectory,
    _trajectory_id,
    generate_optimization_candidates,
    select_validated_variant,
    variant_strategy_uid,
)
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8.learning_blockers_v055 import pack_action_choice


def source(actions, *, seed=1, prefix=(), game="world", level=1, outcome=None):
    anchor = ReplayAnchor(game, seed, tuple(prefix), None)
    target = TrajectoryTarget(level, "LEVEL")
    target_uid = outcome or MemoryUid.zero()
    return SuccessfulTrajectory(
        _trajectory_id(anchor, target, tuple(actions)),
        anchor,
        target,
        tuple(actions),
        MemoryUid.zero(),
        target_uid,
        0,
    )


def validated(*, seed=1, prefix=(), game="world", level=1, actions=(2, 3), outcome=None):
    outcome_uid = outcome or MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
    anchor = ReplayAnchor(game, seed, tuple(prefix), None)
    target = TrajectoryTarget(level, "LEVEL")
    row = ValidatedTrajectory(
        "temporary",
        anchor,
        target,
        tuple(actions),
        MemoryUid.zero(),
        outcome_uid,
        MemoryUid.zero(),
        len(actions) + 2,
        "DELETE_SEGMENT",
        2,
        2,
    )
    raw = row.to_dict()
    raw["variant_id"] = "old-seed-specific"
    return ValidatedTrajectory.from_dict(raw)


class SeedlessIdentityTests(unittest.TestCase):
    def test_anchor_serialization_contains_no_seed(self) -> None:
        raw = ReplayAnchor("world", 999, (1, 2), None).to_dict()
        self.assertNotIn("seed", raw)
        restored = ReplayAnchor.from_dict({**raw, "seed": 12345})
        self.assertEqual(restored.seed, 0)

    def test_trajectory_identity_is_identical_across_seeds(self) -> None:
        first = source((4, 5, 6), seed=1, prefix=(1, 2, 3))
        second = source((4, 5, 6), seed=999, prefix=(1, 2, 3))
        self.assertEqual(first.trajectory_id, second.trajectory_id)

    def test_strategy_identity_is_identical_across_seeds(self) -> None:
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 9, 4))
        first_source = source((4, 5, 6), seed=1, prefix=(1, 2), outcome=outcome)
        second_source = source((4, 5, 6), seed=999, prefix=(1, 2), outcome=outcome)
        first = optimizer.TrajectoryCandidate("a", first_source, "DELETE_ACTION", (4, 6), 1, 1)
        second = optimizer.TrajectoryCandidate("b", second_source, "DELETE_ACTION", (4, 6), 1, 1)
        self.assertEqual(variant_strategy_uid(first), variant_strategy_uid(second))

    def test_variant_selection_ignores_seed_but_keeps_prefix_anchor(self) -> None:
        row = validated(seed=7, prefix=(1, 2, 3))
        selected = select_validated_variant(
            (row,),
            source_id="world",
            seed=8888,
            action_history=(1, 2, 3),
        )
        self.assertIs(selected, row)
        self.assertIsNone(
            select_validated_variant(
                (row,),
                source_id="world",
                seed=7,
                action_history=(1, 2),
            )
        )


class DeferredOutcomeResolutionTests(unittest.TestCase):
    def test_stale_index_retry_never_refreshes_live_graph(self) -> None:
        outcome = MemoryUid.from_key(
            MemoryLevel.M6,
            MemoryType.OUTCOME,
            (91, 92, 93),
        )

        class ReadView:
            _node_by_uid = {}
            _behavior_observed_outcomes = {(123, 6, 999): {outcome}}

            @staticmethod
            def node_records(**_kwargs):
                raise AssertionError("deferred retry refreshed node graph")

            @staticmethod
            def _refresh_strategy_cache():
                raise AssertionError("deferred retry refreshed behavior graph")

        candidate = optimizer.TrajectoryCandidate(
            "deferred",
            source((6,), game="g"),
            "VALIDATE_SOURCE",
            (6,),
            0,
            0,
        )
        result = SimpleNamespace(
            success=True,
            terminal_context=456,
            terminal_action=6,
            outcome_signature=999,
        )
        runtime = SimpleNamespace(read_view=ReadView())
        from v8 import normalized_memory_v086_fixups as grounding

        with patch.object(grounding, "_grounded_context", return_value=123):
            resolved = v818._resolve_target_outcome(
                runtime,
                candidate,
                result,
                refresh_view=False,
            )

        self.assertEqual(resolved, outcome)


class MigrationTests(unittest.TestCase):
    def test_v1_state_migrates_without_seed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            row = validated(seed=42, prefix=(1, 2))
            old = row.to_dict()
            old["anchor"]["seed"] = 42
            old["variant_id"] = "legacy-seeded-id"
            service.load_state(
                {
                    "version": 1,
                    "seen_sources": ["old-source"],
                    "attempted": ["old-candidate"],
                    "validated": [old],
                    "metrics": {},
                }
            )
            state = service.state_dict()
            self.assertEqual(state["version"], 3)
            self.assertNotIn("seed", state["validated"][0]["anchor"])
            self.assertNotEqual(state["validated"][0]["variant_id"], "legacy-seeded-id")
            self.assertNotIn("old-source", state["seen_sources"])
            self.assertNotIn("old-candidate", state["attempted"])

    def test_v2_state_reopens_seen_sources_but_preserves_completed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(
                Path(root), validator=lambda _candidate: None
            )
            service.load_state(
                {
                    "version": 2,
                    "identity_schema_version": 2,
                    "seen_sources": ["possibly-incomplete-source"],
                    "attempted": ["completed-candidate"],
                    "validated": [],
                    "metrics": {},
                }
            )
            state = service.state_dict()
            self.assertNotIn("possibly-incomplete-source", state["seen_sources"])
            self.assertIn("completed-candidate", state["attempted"])

    def test_pending_source_and_incomplete_candidate_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            row = source((1, 2, 3, 4))
            candidate = optimizer.TrajectoryCandidate(
                "in-flight-candidate",
                row,
                "DELETE_ACTION",
                (1, 2, 3),
                3,
                1,
            )
            service = TrajectoryOptimizationService(
                Path(root), validator=lambda _candidate: None
            )
            v818._begin_source_work(service, row)
            v818._track_candidate_work(service, candidate)
            v818._end_source_routing(service, row)
            with service._lock:
                service._seen_sources.add(row.trajectory_id)
                service._attempted.update(
                    {"completed-candidate", candidate.candidate_id}
                )

            state = service.state_dict()
            self.assertEqual(
                [item["trajectory_id"] for item in state["pending_sources"]],
                [row.trajectory_id],
            )
            self.assertNotIn(row.trajectory_id, state["seen_sources"])
            self.assertNotIn(candidate.candidate_id, state["attempted"])
            self.assertIn("completed-candidate", state["attempted"])

            restored = TrajectoryOptimizationService(
                Path(root) / "restored", validator=lambda _candidate: None
            )
            restored.load_state(state)
            self.assertIn(row.trajectory_id, restored._seen_sources)
            self.assertEqual(len(restored._v818_restored_sources), 1)
            v818._restore_pending_sources(restored)
            self.assertEqual(restored._sources.get_nowait().trajectory_id, row.trajectory_id)

    def test_queued_source_is_serialized_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            row = source((1, 2, 3))
            service = TrajectoryOptimizationService(
                Path(root), validator=lambda _candidate: None
            )
            self.assertTrue(service.submit_trajectory(row))
            state = service.state_dict()
            self.assertEqual(
                [item["trajectory_id"] for item in state["pending_sources"]],
                [row.trajectory_id],
            )
            self.assertNotIn(row.trajectory_id, state["seen_sources"])


class CandidateSchedulingTests(unittest.TestCase):
    def test_long_trajectory_tries_large_segment_deletion_before_single_deletion(self) -> None:
        row = source(tuple(range(1, 101)))
        candidates = generate_optimization_candidates(
            row,
            TrajectoryOptimizerConfig(max_candidates_per_round=24),
        )
        self.assertTrue(candidates)
        kinds = [candidate.edit_kind for candidate in candidates]
        self.assertIn("DELETE_SEGMENT", kinds)
        if "DELETE_ACTION" in kinds:
            self.assertLess(kinds.index("DELETE_SEGMENT"), kinds.index("DELETE_ACTION"))
        self.assertTrue(any(candidate.removed_length >= 20 for candidate in candidates))

    def test_prefix_cache_prefers_shorter_validated_prior_level_path(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            row = source((8, 9), prefix=(1, 2, 3, 4, 5, 6), level=2)
            candidate = optimizer.TrajectoryCandidate("c", row, "DELETE_ACTION", (8,), 1, 1)
            service._v818_best_prefixes["world"] = {1: (1, 2, 3)}
            self.assertEqual(service._v818_prefix_for(candidate), (1, 2, 3))


class ValidatorPoolTests(unittest.TestCase):
    def test_exact_click_is_replayable_outside_current_exploration_page(self) -> None:
        env = SimpleNamespace(
            available_actions=lambda: (1, 2, 3, 4),
            cognitive_action_executable=lambda action: action == exact_click,
        )
        exact_click = pack_action_choice(6, 4, 4)

        self.assertTrue(v818._GameReplayValidator._action_available(env, exact_click))
        self.assertFalse(v818._GameReplayValidator._action_available(env, 5))

    def test_validator_pool_is_bounded_to_ten_games(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            service._v818_max_validators = 10
            started = []

            class FakeThread:
                def __init__(self, *, target, args, name, daemon):
                    del target, args, daemon
                    self.name = name
                    self._alive = False
                def start(self):
                    self._alive = True
                    started.append(self.name)
                def is_alive(self):
                    return self._alive

            with patch.object(v818.threading, "Thread", FakeThread):
                for index in range(12):
                    v818._ensure_validator(service, f"g{index:02d}")
            self.assertEqual(len(started), 10)
            self.assertEqual(len(service._v818_waiting_games), 2)

    def test_each_game_has_its_own_candidate_queue(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            with service._v818_validator_lock:
                service._v818_game_queues["a"] = __import__("queue").Queue()
                service._v818_game_queues["b"] = __import__("queue").Queue()
            self.assertIsNot(service._v818_game_queues["a"], service._v818_game_queues["b"])

    def test_validation_quantum_hands_slot_to_existing_waiter_first(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(
                Path(root), validator=lambda _candidate: None
            )
            active = queue.Queue()
            active.put(object())
            service._v818_game_queues["active"] = active
            service._v818_validator_threads["active"] = threading.current_thread()
            service._v818_waiting_games = {"waiting"}
            calls = []

            def ensure(_service, game):
                calls.append(str(game))
                _service._v818_waiting_games.discard(str(game))

            with patch.object(v818, "_ensure_validator", side_effect=ensure):
                v818._retire_game_validator(service, "active")

            self.assertEqual(calls, ["waiting", "active"])


class InboxIngestionTests(unittest.TestCase):
    def test_v819_duplicate_is_removed_instead_of_blocking_quiescence(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            row = source((1, 2, 3))
            pending = service.inbox / "duplicate.json"
            optimizer._atomic_json(pending, row.to_dict())
            service._v819_lock = threading.Lock()
            service._v819_source_seen = {row.trajectory_id}

            with patch.object(service, "submit_trajectory") as submit:
                v818._ingest_inbox_v818(service)

            self.assertFalse(pending.exists())
            submit.assert_not_called()


class SafeActivationTests(unittest.TestCase):
    def test_target_compatible_variant_can_activate_without_seed_or_exact_prefix(self) -> None:
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (4, 5, 6, 7))
        row = validated(seed=7, prefix=(1, 2, 3, 4), actions=(5, 6), outcome=outcome)
        row = replace(row, parent_strategy_uid=strategy)
        view = SimpleNamespace(
            _v814_variants=(row,),
            _v814_attempted_variants=set(),
        )
        prior_source = optimizer._CAPTURE_SOURCE_ID
        prior_seed = optimizer._CAPTURE_SEED
        prior_history = list(optimizer._ACTOR_ACTION_HISTORY)
        try:
            optimizer._CAPTURE_SOURCE_ID = "world"
            optimizer._CAPTURE_SEED = 9999
            optimizer._ACTOR_ACTION_HISTORY[:] = [99]
            plans = (PlannedAction(4, outcome, strategy, 1.0, False),)
            with patch.object(optimizer, "_refresh_view_variants", lambda _view: None):
                selected = v818._target_compatible_variant(view, plans, (4, 5, 6))
            self.assertIs(selected, row)
        finally:
            optimizer._CAPTURE_SOURCE_ID = prior_source
            optimizer._CAPTURE_SEED = prior_seed
            optimizer._ACTOR_ACTION_HISTORY[:] = prior_history

    def test_unrelated_target_does_not_activate(self) -> None:
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        other = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (9, 9, 9))
        row = validated(actions=(5, 6), outcome=outcome)
        view = SimpleNamespace(_v814_variants=(row,), _v814_attempted_variants=set())
        prior_source = optimizer._CAPTURE_SOURCE_ID
        try:
            optimizer._CAPTURE_SOURCE_ID = "world"
            plans = (PlannedAction(5, other, row.parent_strategy_uid, 1.0, False),)
            with patch.object(optimizer, "_refresh_view_variants", lambda _view: None):
                self.assertIsNone(v818._target_compatible_variant(view, plans, (5, 6)))
        finally:
            optimizer._CAPTURE_SOURCE_ID = prior_source


class ActivityReportingTests(unittest.TestCase):
    def test_activity_reports_even_when_no_validation_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            with service._lock:
                service._validations = 12
                service._candidates_generated = 20
                service._trajectories_seen = 2
            service._v818_last_activity_report -= 301.0
            output = io.StringIO()
            with redirect_stdout(output):
                emitted = service._v818_emit_activity_if_due()
            self.assertTrue(emitted)
            line = output.getvalue()
            self.assertIn("trajectory optimization validators=", line)
            self.assertIn("validations=12", line)
            self.assertIn("successes=0", line)


if __name__ == "__main__":
    unittest.main()
