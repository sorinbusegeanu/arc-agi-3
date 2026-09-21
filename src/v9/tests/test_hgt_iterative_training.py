from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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

    def test_branch_selection_uses_macro_success_then_progress(self):
        self.assertEqual(select_matched_branch(.30, .20).selected_branch, "hgt_on")
        self.assertEqual(select_matched_branch(.20, .30).selected_branch, "hgt_off")
        self.assertEqual(select_matched_branch(.20, .20, on_levels=5, off_levels=3).selected_branch, "hgt_on")


if __name__ == "__main__":
    unittest.main()
