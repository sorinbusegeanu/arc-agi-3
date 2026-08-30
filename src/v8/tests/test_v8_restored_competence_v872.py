from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v8 import mixed_environment_v859 as mixed
from v8.actor import ActorJob, ActorProgress
from v8.restored_competence_v872 import (
    capture_startup_restored_competence_v872,
    generic_replay_candidate_v872,
    restored_competence_snapshot_v872,
)
from v8.verified_success_metrics_v866 import (
    SUCCESS_ROOT_ENV,
    _VerifiedAdapterProxy,
    _periodic_progress_line_v866,
    record_verified_success_v866,
    verified_success_snapshot_v866,
)


class _Stop:
    def is_set(self):
        return False


class _View:
    def plan_candidates(self, context, actions):
        del context, actions
        return ()

    def score_actions(self, context, actions):
        del context
        return tuple(
            SimpleNamespace(action_id=action, support_count=0, score=0.0)
            for action in actions
        )

    def outcome_distribution(self, context, action):
        del context, action
        return {}


class _Runtime:
    def __init__(self):
        self.read_view = _View()
        self._stop = _Stop()
        self.watermark = 0
        self.events = []

    def make_experience(self, **kwargs):
        return dict(kwargs)

    def submit(self, event):
        self.events.append(event)
        self.watermark += 1


class _ReplayAdapter:
    def __init__(self):
        self.identity = SimpleNamespace(source_hash=12345)
        self.observation_schema = SimpleNamespace(schema_id=101)
        self.action_schema = SimpleNamespace(schema_id=202)
        self.actions = []
        self.closed = False

    def observe(self):
        return tuple(self.actions)

    def available_actions(self):
        return (1, 2)

    def observation_signature(self, observation):
        return 100 + len(observation)

    def cognitive_transition_signature(self, before, after):
        return 200 + len(after) - len(before)

    def cognitive_family_signature(self, before, after):
        del before, after
        return 300

    def cognitive_changed_extent(self, before, after):
        return abs(len(after) - len(before))

    def step(self, action):
        self.actions.append(int(action))
        return tuple(self.actions)

    def cognitive_boundary_event(self):
        won = self.actions == [2, 1]
        return SimpleNamespace(continuation=not won, primary_valence=1 if won else 0)

    def reset(self):
        self.actions = []

    def close(self):
        self.closed = True


class RestoredCompetenceV872Tests(unittest.TestCase):
    def setUp(self):
        self.previous_root = os.environ.get(SUCCESS_ROOT_ENV)
        os.environ.pop(SUCCESS_ROOT_ENV, None)

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop(SUCCESS_ROOT_ENV, None)
        else:
            os.environ[SUCCESS_ROOT_ENV] = self.previous_root

    @staticmethod
    def _write_arc_solution(root: Path) -> None:
        path = root / "trajectory_optimizer" / "best_successful.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "games": {
                        "tp02": {
                            "trajectory_id": "tp02-complete",
                            "successes": 1,
                            "levels": [
                                {"level": level, "actions": [level + 1]}
                                for level in range(5)
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _record_historical_generic(root: Path) -> Path:
        old_run = root / "verified_success" / "run-old"
        old_run.mkdir(parents=True, exist_ok=True)
        assert record_verified_success_v866(
            game_id="FrozenLake-v1",
            seed=17,
            terminal_state="WIN",
            levels_completed=1,
            actions=(2, 1),
            capture_step=2,
            root=old_run,
        )
        return old_run

    def test_startup_metrics_separate_arc_and_generic_durable_competence(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp)
            self._write_arc_solution(runtime_root)
            self._record_historical_generic(runtime_root)
            current = runtime_root / "verified_success" / "run-current"
            current.mkdir(parents=True)

            capture_startup_restored_competence_v872(runtime_root, current)
            games = (
                "gp03",
                "tp02",
                "FrozenLake-v1",
                "ArcAgi/Chess-v0",
                "ArcAgi/Sudoku-v0",
            )
            restored = restored_competence_snapshot_v872(current, games)

            self.assertEqual(restored["level_target_count"], 13)
            self.assertEqual(restored["restored_levels_solved"], 6)
            self.assertEqual(restored["restored_games_solved"], 2)
            self.assertAlmostEqual(
                restored["restored_level_solve_rate_pct"], 600.0 / 13.0
            )
            self.assertEqual(restored["restored_game_solve_rate_pct"], 40.0)
            self.assertEqual(
                generic_replay_candidate_v872(current, "FrozenLake-v1")["actions"],
                [2, 1],
            )

    def test_generic_restart_replays_recorded_seed_and_requires_positive_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp)
            self._record_historical_generic(runtime_root)
            current = runtime_root / "verified_success" / "run-current"
            current.mkdir(parents=True)
            capture_startup_restored_competence_v872(runtime_root, current)
            os.environ[SUCCESS_ROOT_ENV] = str(current)

            runtime = _Runtime()
            seeds = []

            def make_adapter(game_id, *, seed=0):
                seeds.append(int(seed))
                return _VerifiedAdapterProxy(_ReplayAdapter(), game_id, seed)

            with patch.object(mixed, "make_adapter", side_effect=make_adapter):
                result = mixed.run_generic_actor_job(
                    runtime,
                    ActorJob(7, "FrozenLake-v1", 2, 999, epsilon=1.0),
                )

            self.assertEqual(seeds, [17])
            self.assertEqual([row["action_id"] for row in runtime.events], [2, 1])
            self.assertEqual(result.wins, 1)
            verified = verified_success_snapshot_v866(current, ("FrozenLake-v1",))
            self.assertEqual(verified["current_run_games_won"], 1)

    def test_stdout_keeps_current_and_restored_rates_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp)
            self._record_historical_generic(runtime_root)
            current = runtime_root / "verified_success" / "run-current"
            current.mkdir(parents=True)
            capture_startup_restored_competence_v872(runtime_root, current)
            os.environ[SUCCESS_ROOT_ENV] = str(current)

            rows = tuple(
                ActorProgress(index, game, 0, 0, 0, 0)
                for index, game in enumerate(
                    (
                        "gp03",
                        "tp02",
                        "FrozenLake-v1",
                        "ArcAgi/Chess-v0",
                        "ArcAgi/Sudoku-v0",
                    ),
                    start=1,
                )
            )
            line = _periodic_progress_line_v866(rows, 100)

            self.assertIn("L=0.0% G=0.0%", line)
            self.assertIn("RestL=7.7% RestG=20.0%", line)


if __name__ == "__main__":
    unittest.main()
