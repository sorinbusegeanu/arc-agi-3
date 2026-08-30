from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install chronological runtime stack
from v8 import research_integrity_v863 as v863


class ResearchIntegrityV863Tests(unittest.TestCase):
    def _optimizer_artifacts(self, root: Path) -> tuple[Path, ...]:
        inbox = root / "trajectory_optimizer" / "inbox"
        solutions = root / "trajectory_optimizer" / "solutions_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        solutions.mkdir(parents=True, exist_ok=True)
        paths = (
            inbox / "old.json",
            solutions / "old-solution.json",
            root / "trajectory_optimizer" / "validated.json",
            root / "trajectory_optimizer" / "best_successful.json",
        )
        for path in paths:
            path.write_text("{}", encoding="utf-8")
        return paths

    def test_clean_run_purges_orphan_optimizer_behavioral_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._optimizer_artifacts(root)
            log = root / "trajectory_optimizer" / "trajectory_optimizer.log"
            log.write_text("keep diagnostics", encoding="utf-8")

            removed = v863._purge_orphan_optimizer_state(root)

            self.assertEqual(len(removed), 4)
            self.assertTrue(all(not path.exists() for path in paths))
            self.assertTrue(log.exists())

    def test_runtime_init_purges_when_no_snapshot_is_restorable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._optimizer_artifacts(root)
            config = SimpleNamespace(root=root, restore=True)
            runtime = SimpleNamespace()

            with patch("v8.snapshot.latest_complete_snapshot", return_value=None), patch.object(
                v863, "_BASE_RUNTIME_INIT", lambda self, config, *args, **kwargs: None
            ):
                v863._runtime_init_v863(runtime, config)

            self.assertTrue(all(not path.exists() for path in paths))
            self.assertEqual(len(runtime._v863_clean_run_purged_optimizer), 4)

    def test_runtime_init_preserves_optimizer_state_with_restorable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._optimizer_artifacts(root)
            config = SimpleNamespace(root=root, restore=True)
            runtime = SimpleNamespace()

            with patch("v8.snapshot.latest_complete_snapshot", return_value=root / "snapshot"), patch.object(
                v863, "_BASE_RUNTIME_INIT", lambda self, config, *args, **kwargs: None
            ):
                v863._runtime_init_v863(runtime, config)

            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(runtime._v863_clean_run_purged_optimizer, ())

    def test_mixed_sampling_defers_all_early_completion_paths(self) -> None:
        runtime = SimpleNamespace(_v863_mixed_sampling_active=True)
        mark_calls = []
        drain_calls = []
        with patch.object(v863, "_BASE_MARK_SAMPLING_COMPLETE", lambda row: mark_calls.append(row)), patch.object(
            v863, "_BASE_REQUEST_FINAL_PEER_DRAIN", lambda row: drain_calls.append(row)
        ):
            v863._mark_sampling_complete_v863(runtime)
            v863._request_final_peer_drain_v863(runtime)

        self.assertEqual(mark_calls, [])
        self.assertEqual(drain_calls, [])
        self.assertTrue(runtime._v863_mixed_final_drain_pending)

    def test_mixed_wrapper_declares_completion_once_after_all_environments_return(self) -> None:
        runtime = SimpleNamespace()
        events = []

        def fake_mixed(row, jobs, **kwargs):
            self.assertTrue(row._v863_mixed_sampling_active)
            v863._mark_sampling_complete_v863(row)
            v863._request_final_peer_drain_v863(row)
            events.append("all-environments-returned")
            return ("done",)

        def final_drain(row):
            self.assertFalse(row._v863_mixed_sampling_active)
            events.append("final-drain")

        with patch.object(v863, "_BASE_RUN_MIXED_ACTOR_JOBS", fake_mixed), patch.object(
            v863, "_BASE_MARK_SAMPLING_COMPLETE", lambda row: events.append("early-mark")
        ), patch.object(v863, "_BASE_REQUEST_FINAL_PEER_DRAIN", final_drain):
            result = v863._run_mixed_actor_jobs_v863(runtime, ())

        self.assertEqual(result, ("done",))
        self.assertEqual(events, ["all-environments-returned", "final-drain"])
        self.assertFalse(runtime._v863_mixed_final_drain_pending)

    def test_mixed_wrapper_does_not_finalize_failed_sampling(self) -> None:
        runtime = SimpleNamespace()
        drains = []

        def fail(row, jobs, **kwargs):
            raise RuntimeError("sampling failed")

        with patch.object(v863, "_BASE_RUN_MIXED_ACTOR_JOBS", fail), patch.object(
            v863, "_BASE_REQUEST_FINAL_PEER_DRAIN", lambda row: drains.append(row)
        ):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                v863._run_mixed_actor_jobs_v863(runtime, ())

        self.assertEqual(drains, [])
        self.assertFalse(runtime._v863_mixed_sampling_active)


if __name__ == "__main__":
    unittest.main()
