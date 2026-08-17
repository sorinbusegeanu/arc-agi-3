from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import v8
from v7.environment.arc_adapter import ArcGridEnvironment
from v8 import complete_win_trajectory_repair_v825 as repair
from v8 import learning_fixes_v088 as learning
from v8 import runtime_repair_v822 as v822
from v8 import solved_game_recovery_v821 as recovery
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_inspection_v819_fixups as visibility
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import MemoryUid


def _validated(identifier: str, *, level: int, terminal: str, actions):
    anchor = optimizer.ReplayAnchor("ic02", 0, (), None)
    target = optimizer.TrajectoryTarget(level, terminal)
    values = tuple(int(value) for value in actions)
    return optimizer.ValidatedTrajectory(
        identifier,
        anchor,
        target,
        values,
        MemoryUid.zero(),
        MemoryUid.zero(),
        MemoryUid.zero(),
        len(values) + 1,
        "VALIDATE_SOURCE",
        1,
        1,
    )


def _write_snapshot_aux(root: Path, state: dict[str, object]) -> None:
    from v8 import snapshot

    directory = root / "snapshots" / "snapshot-00000000000000000001"
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, sort_keys=True).encode("utf-8")
    (directory / "auxiliary_state.json").write_bytes(payload)
    manifest = {
        "format_version": 3,
        "snapshot_id": 1,
        "watermark": 1,
        "generation": 1,
        "final": True,
        "shards": [],
        "auxiliary_state": {
            "file": "auxiliary_state.json",
            "sha256": snapshot._sha(payload),
            "bytes": len(payload),
        },
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (directory / "manifest.json").write_bytes(manifest_payload)
    (directory / "COMPLETE").write_text(snapshot._sha(manifest_payload) + "\n", encoding="ascii")


class CompleteWinTrajectoryRepairV825Tests(unittest.TestCase):
    def test_v822_composes_through_v821_level_recovery(self) -> None:
        self.assertIs(v822._BASE_ENV_STEP, recovery._tracked_env_step)
        self.assertIs(v822._BASE_ENV_RESET, recovery._tracked_env_reset)
        self.assertIs(ArcGridEnvironment.step, v822._runtime_env_step)
        self.assertIs(ArcGridEnvironment.reset, v822._runtime_env_reset)

    def test_complete_win_cost_replaces_final_level_fragment_metric(self) -> None:
        prior_best = learning._BEST_WIN_STEPS
        prior_last = learning._LAST_WIN_STEPS
        prior_complete_best = repair._COMPLETE_BEST_WIN_STEPS
        prior_complete_last = repair._COMPLETE_LAST_WIN_STEPS
        try:
            learning._BEST_WIN_STEPS = 2
            learning._LAST_WIN_STEPS = 2
            repair._COMPLETE_BEST_WIN_STEPS = 0
            repair._COMPLETE_LAST_WIN_STEPS = 0
            base = Mock()
            with patch.object(repair, "_BASE_PUBLISH_RUNTIME_LEVELS", base):
                repair._publish_runtime_levels_v825("ic01", ((1, 2, 3), (4, 5)))
            base.assert_called_once_with("ic01", ((1, 2, 3), (4, 5)))
            self.assertEqual(learning._BEST_WIN_STEPS, 5)
            self.assertEqual(learning._LAST_WIN_STEPS, 5)
        finally:
            learning._BEST_WIN_STEPS = prior_best
            learning._LAST_WIN_STEPS = prior_last
            repair._COMPLETE_BEST_WIN_STEPS = prior_complete_best
            repair._COMPLETE_LAST_WIN_STEPS = prior_complete_last

    def test_existing_validated_sidecar_is_loaded_before_service_start_can_overwrite_it(self) -> None:
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            service = optimizer.TrajectoryOptimizationService(
                root / "trajectory_optimizer",
                validator=lambda _candidate: None,
            )
            row = _validated("level-1", level=1, terminal="LEVEL", actions=(1, 2))
            optimizer._atomic_json(
                service.validated_path,
                {"version": 1, "validated": [row.to_dict()]},
            )
            self.assertEqual(repair._restore_persisted_validated_v825(service), 1)
            restored = tuple(service._validated.values())
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].anchor.source_id, "ic02")
            self.assertEqual(restored[0].target.levels_completed, 1)
            self.assertEqual(tuple(restored[0].actions), (1, 2))

    def test_show_best_recovers_complete_ic02_from_snapshot_when_live_sidecar_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            rows = (
                _validated("level-1", level=1, terminal="LEVEL", actions=(1, 2)),
                _validated("win", level=2, terminal="WIN", actions=(3,)),
            )
            _write_snapshot_aux(
                root,
                {
                    "trajectory_optimizer": {
                        "version": 1,
                        "validated": [row.to_dict() for row in rows],
                    }
                },
            )

            self.assertFalse((root / "trajectory_optimizer" / "validated.json").exists())
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = inspection.show_best_trajectory(root, "ic02")

            self.assertEqual(code, 0)
            self.assertIn("game=ic02 cost=3", stream.getvalue())
            self.assertIn("L0: A1,A2", stream.getvalue())
            self.assertIn("L1: A3", stream.getvalue())

    def test_final_installed_visibility_and_service_start_authorities(self) -> None:
        self.assertIs(visibility._best_visible_solution, repair._best_visible_solution_v825)
        self.assertIs(optimizer.TrajectoryOptimizationService.start, repair._service_start_v825)
        self.assertIs(recovery._publish_runtime_levels, repair._publish_runtime_levels_v825)
        self.assertIs(v822._reset_solve_metrics, repair._reset_complete_solve_metrics_v825)


if __name__ == "__main__":
    unittest.main()
