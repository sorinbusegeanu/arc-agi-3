from __future__ import annotations

import io
import os
import queue
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_inspection_v819_fixups as inspection_fix
from v8 import trajectory_optimizer_v814 as optimizer
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


def _fake_full_lease_worker(
    *,
    worker_id,
    assignment_queue,
    event_queue,
    ready_event,
    experience_ring_args,
    read_descriptors,
    watermark,
    stop_event,
    actor_throttle,
    snapshot_freeze,
    trajectory_root,
):
    del (
        experience_ring_args,
        read_descriptors,
        watermark,
        actor_throttle,
        snapshot_freeze,
        trajectory_root,
    )
    ready_event.set()
    while not stop_event.is_set():
        try:
            lease = assignment_queue.get(timeout=0.10)
        except queue.Empty:
            continue
        if lease is None:
            return
        result = actor_module.ActorResult(
            int(worker_id),
            str(lease.game_id),
            int(lease.steps),
            0,
            0,
            0,
            0,
            0,
            0,
            (),
            (),
            (),
            None,
        )
        event_queue.put(v819._LeaseResult(int(worker_id), lease, result))


class LearningPerformanceAndTrajectoryVisibilityV819Tests(unittest.TestCase):
    @staticmethod
    def runtime_config(root: Path) -> V8RuntimeConfig:
        return V8RuntimeConfig(
            root=root,
            shards=1,
            stage_workers=1,
            stage_ring_capacity=512,
            shard_ring_capacity=512,
            node_capacity_per_shard=12000,
            edge_capacity_per_shard=24000,
            action_capacity_per_shard=2048,
            snapshot_interval_seconds=9999.0,
            enable_snapshots=False,
            restore=False,
            enable_peers=False,
        )

    @staticmethod
    def row(game: str, trajectory_id: str, *, prefix=(), actions=(), levels=1, state="WIN"):
        return optimizer.SuccessfulTrajectory(
            trajectory_id,
            optimizer.ReplayAnchor(game, 0, tuple(prefix), None),
            optimizer.TrajectoryTarget(int(levels), str(state)),
            tuple(actions),
        )

    def test_default_discovery_lease_preserves_full_original_horizon(self) -> None:
        root = Path(tempfile.mkdtemp())
        runtime = ContinuousMemoryRuntime(self.runtime_config(root))
        job = actor_module.ActorJob(1, "synthetic-game", 130, 7, None, 0.10, 0.02, 256, 1000)
        try:
            with patch.object(perf, "_worker_until_win", _fake_full_lease_worker):
                results = actor_module.run_actor_jobs(
                    runtime,
                    (job,),
                    timeout=60.0,
                    progress_interval_seconds=5.0,
                )
            self.assertEqual(sum(int(row.steps) for row in results), 130)
            run = runtime._v819_adaptive_learning._run["synthetic-game"]
            self.assertEqual(run.sample_steps, 130)
            self.assertEqual(run.leases, 1)
        finally:
            runtime.close(normal=False)

    def test_live_allocation_steps_include_active_progress(self) -> None:
        lease = v819.ActorLease(
            1,
            1,
            "g",
            100,
            0,
            None,
            0.1,
            1000,
            v819.SamplingMode.DISCOVERY,
        )
        progress = actor_module.ActorProgress(1, "g", 37, 0, 0, 0)
        totals = perf._live_steps_by_game(
            {"g": {"steps": 11}},
            {1: progress},
            {1: lease},
        )
        self.assertEqual(totals["g"], 48)

    def test_runtime_sets_trajectory_root_before_base_start(self) -> None:
        root = Path(tempfile.mkdtemp())
        expected = str(root / "trajectory_optimizer")
        dummy = SimpleNamespace(
            _v814_trajectory_optimizer=SimpleNamespace(root=root / "trajectory_optimizer")
        )
        prior = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")
        observed = []
        try:
            with patch.object(
                perf,
                "_BASE_RUNTIME_START",
                side_effect=lambda _self: observed.append(
                    os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")
                ),
            ):
                perf._runtime_start_perf(dummy)
            self.assertEqual(observed, [expected])
        finally:
            if prior is None:
                os.environ.pop("ARC_AGI3_V8_TRAJECTORY_ROOT", None)
            else:
                os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = prior

    def test_show_best_can_read_single_level_win_from_normal_optimizer_inbox(self) -> None:
        root = Path(tempfile.mkdtemp())
        optimizer_root = root / "trajectory_optimizer"
        optimizer._atomic_json(
            optimizer_root / "inbox" / "win.json",
            self.row("ic02", "win", actions=(1, 2, 3), levels=1, state="WIN").to_dict(),
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(root, "ic02")
        self.assertEqual(code, 0)
        self.assertIn("game=ic02 cost=3 source=observed reliability=1.000", stream.getvalue())
        self.assertIn("L0: A1,A2,A3", stream.getvalue())

    def test_show_best_reconstructs_only_exact_nested_optimizer_chain(self) -> None:
        root = Path(tempfile.mkdtemp())
        optimizer_root = root / "trajectory_optimizer"
        optimizer._atomic_json(
            optimizer_root / "inbox" / "level.json",
            self.row("g", "level", actions=(1, 2), levels=1, state="LEVEL").to_dict(),
        )
        optimizer._atomic_json(
            optimizer_root / "inbox" / "win.json",
            self.row("g", "win", prefix=(1, 2), actions=(3, 4), levels=2, state="WIN").to_dict(),
        )
        record = inspection_fix._best_visible_solution(root, "g")
        self.assertIsNotNone(record)
        self.assertEqual(
            [tuple(level["actions"]) for level in record["levels"]],
            [(1, 2), (3, 4)],
        )


if __name__ == "__main__":
    unittest.main()
