from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v9.hgt import training as hgt_training
from v9.hgt.epoch_dataset import (
    EpochTransitionDataset,
    action_ranking_pairs,
    iter_epoch_transitions,
    transition_training_rows,
    transition_training_rows_from_records,
)
from v9.hgt.matched_evaluation import matched_jobs, select_matched_branch
from v9.runtime.multiprocess import EncodedTransition
from v9.runtime.runtime import ContinuousMemoryRuntime
from v9.runtime import epoch_runner


def transition(step: int, action: int, valence: int = 0, success: bool = False) -> EncodedTransition:
    return EncodedTransition(
        actor_id=1, producer_sequence=step + 1, global_step=step,
        environment_identity=("synthetic", "test", "1", "test"),
        episode_id=1, observation_schema_id=1, before_signature=7,
        action_id=action, after_signature=8 + step, available_actions_after=2,
        primary_valence=valence, symbols=(), curriculum_step=None,
        game_scenario="test", task_success=success,
    )


class HGTIterativeTrainingTests(unittest.TestCase):
    def test_every_oom_retry_preserves_the_promotion_contract(self):
        source = Path(inspect.getsourcefile(hgt_training) or "").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        training_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_hgt_epoch"
        )
        retry_calls = [
            node
            for node in ast.walk(training_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_retry_after_oom"
        ]
        self.assertGreaterEqual(len(retry_calls), 2)
        for call in retry_calls:
            self.assertIn("allow_promotion", {row.arg for row in call.keywords})

    def test_oom_retry_forwards_false_allow_promotion(self):
        gauges = {}
        runtime = SimpleNamespace(
            config=SimpleNamespace(
                scientific=SimpleNamespace(hgt_oom_retry_limit=2)
            ),
            set_telemetry_gauge=gauges.__setitem__,
        )
        expected = object()
        with patch.object(hgt_training, "train_hgt_epoch", return_value=expected) as retry:
            result = hgt_training._retry_after_oom(
                runtime,
                epoch=3,
                training_epochs=4,
                learning_rate=0.01,
                root="run",
                allow_promotion=False,
                budget_scale=1.0,
                oom_retry=0,
                exc=RuntimeError("CUDA out of memory"),
            )
        self.assertIs(result, expected)
        retry.assert_called_once_with(
            runtime,
            epoch=3,
            training_epochs=4,
            learning_rate=0.01,
            root="run",
            allow_promotion=False,
            _budget_scale=0.5,
            _oom_retry=1,
        )
        self.assertEqual(gauges["hgt_oom_retry_count"], 1)
        self.assertEqual(gauges["hgt_oom_shedding_factor"], 0.5)

    def test_epoch_dataset_captures_every_transition(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "epoch.jsonl"
            with EpochTransitionDataset(path, epoch=1, branch="bootstrap", model_version=None) as dataset:
                for i in range(25):
                    dataset.append(transition(i, i % 2))
                self.assertEqual(dataset.count, 25)
            self.assertEqual(len(list(iter_epoch_transitions(path))), 25)
            self.assertEqual(len(transition_training_rows(path)), 25)

    def test_return_targets_and_ranking_use_sampled_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "epoch.jsonl"
            with EpochTransitionDataset(path, epoch=2, branch="hgt_off", model_version="hgt-000001") as dataset:
                dataset.append(transition(0, 0, -1))
                dataset.append(transition(1, 1, 1, True))
            rows = transition_training_rows(path)
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[1]["target_return"], rows[0]["target_return"])
            self.assertEqual(len(action_ranking_pairs(rows)), 1)

    def test_wal_records_reconstruct_the_same_bounded_training_rows(self):
        raw = [
            {
                "environment_identity": ["synthetic", "test", "1", "test"],
                "game_scenario": "test",
                "actor_id": 1,
                "episode_id": 1,
                "global_step": 0,
                "before_signature": 7,
                "after_signature": 8,
                "action_id": 0,
                "primary_valence": -1,
                "task_success": False,
                "task_failure": False,
                "task_truncated": False,
            },
            {
                "environment_identity": ["synthetic", "test", "1", "test"],
                "game_scenario": "test",
                "actor_id": 1,
                "episode_id": 1,
                "global_step": 1,
                "before_signature": 8,
                "after_signature": 9,
                "action_id": 1,
                "primary_valence": 1,
                "task_success": True,
                "task_failure": False,
                "task_truncated": False,
            },
        ]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "epoch.jsonl"
            path.write_text("".join(__import__("json").dumps(row) + "\n" for row in raw))
            legacy = transition_training_rows(path)
        self.assertEqual(transition_training_rows_from_records(iter(raw)), legacy)

    def test_matched_jobs_preserve_seeds_and_budgets(self):
        jobs = [(1, object(), 500, 123), (2, object(), 700, 456)]
        on, off = matched_jobs(jobs)
        self.assertEqual([(x[0], x[2], x[3]) for x in on], [(x[0], x[2], x[3]) for x in off])

    def test_matched_jobs_are_independent_lists(self):
        jobs = [(1, object(), 500, 123)]
        on, off = matched_jobs(jobs)
        on.clear()
        self.assertEqual(len(off), 1)



    def test_experiment_state_restore_reinstates_branch_base(self):
        class Fake:
            capture_experiment_state = ContinuousMemoryRuntime.capture_experiment_state
            restore_experiment_state = ContinuousMemoryRuntime.restore_experiment_state
            def __init__(self):
                import threading
                self._lock = threading.RLock()
                self.value = 1
            def wait_quiescent(self): pass
            def flush_deferred_memory_updates(self): pass
            def state_dict(self): return {"value": self.value}
            def _restore(self, snapshot): self.value = int(snapshot["state"]["value"])
        runtime = Fake()
        base = runtime.capture_experiment_state()
        runtime.value = 9
        runtime.restore_experiment_state(base)
        self.assertEqual(runtime.value, 1)

    def test_matched_epoch_avoids_redundant_off_state_snapshot(self):
        source = inspect.getsource(epoch_runner.run_epochs)
        self.assertEqual(source.count("capture_experiment_state()"), 2)
        self.assertNotIn("off_state = runtime.capture_experiment_state()", source)

    def test_branch_selection_uses_macro_success_then_progress(self):
        self.assertEqual(select_matched_branch(.30, .20).selected_branch, "hgt_on")
        self.assertEqual(select_matched_branch(.20, .30).selected_branch, "hgt_off")
        self.assertEqual(select_matched_branch(.20, .20, on_levels=5, off_levels=3).selected_branch, "hgt_on")


if __name__ == "__main__":
    unittest.main()
