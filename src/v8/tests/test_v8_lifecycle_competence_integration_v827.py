from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as adaptive
from v8 import lifecycle_competence_integration_v827 as integration
from v8 import lifecycle_switch_v827 as switch
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import CognitiveState, MemoryUid, ValidationState
from v8.peers import DevelopmentalPeerSupervisor


class _FakeReadView:
    def __init__(self, index) -> None:
        self._node_by_uid = index

    def _refresh_strategy_cache(self) -> None:
        return None


class LifecycleCompetenceIntegrationV827Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._prior_root = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")
        self._prior_lifecycle = os.environ.get(switch.LIFECYCLE_ENV)
        inspection._reset_observed_capture()

    def tearDown(self) -> None:
        inspection._reset_observed_capture()
        if self._prior_root is None:
            os.environ.pop("ARC_AGI3_V8_TRAJECTORY_ROOT", None)
        else:
            os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = self._prior_root
        if self._prior_lifecycle is None:
            os.environ.pop(switch.LIFECYCLE_ENV, None)
        else:
            os.environ[switch.LIFECYCLE_ENV] = self._prior_lifecycle

    @staticmethod
    def _coordinator(strategy_state: CognitiveState, outcome_state: CognitiveState):
        strategy_uid = MemoryUid(1, 11)
        outcome_uid = MemoryUid(2, 22)
        coordinator = adaptive.AdaptiveLearningCoordinator()
        coordinator.register_games(("g",))
        coordinator._game_won["g"] = True
        coordinator._record("g", 1).state = adaptive.GameLearningState.SOLVED_STABLE
        scope = adaptive.FrontierScope("g", 1, 0, outcome_uid.hi, outcome_uid.lo)
        candidate = adaptive.FrontierCandidate(
            strategy_uid,
            "trajectory",
            123,
            5,
            2,
            2,
            int(ValidationState.TESTED),
            adaptive.FrontierSource.TRAJECTORY_OPTIMIZER,
            10,
        )
        coordinator.frontier.add(scope, candidate)
        coordinator._v827_read_view = _FakeReadView(
            {
                strategy_uid: SimpleNamespace(cognitive_state=int(strategy_state)),
                outcome_uid: SimpleNamespace(cognitive_state=int(outcome_state)),
            }
        )
        return coordinator, strategy_uid

    def test_quarantined_frontier_is_verification_only_not_stable(self) -> None:
        coordinator, strategy_uid = self._coordinator(
            CognitiveState.QUARANTINED,
            CognitiveState.ACTIVE,
        )
        self.assertEqual(
            coordinator.game_state("g"),
            adaptive.GameLearningState.SOLVED_OPTIMIZING,
        )
        self.assertEqual(coordinator.choose_mode("g"), adaptive.SamplingMode.VERIFY)
        self.assertTrue(coordinator.alternative_exclusion("g").is_zero)

        index = coordinator._v827_read_view._node_by_uid
        index[strategy_uid].cognitive_state = int(CognitiveState.REACTIVATED)
        self.assertEqual(
            coordinator.game_state("g"),
            adaptive.GameLearningState.SOLVED_STABLE,
        )
        self.assertEqual(coordinator.alternative_exclusion("g"), strategy_uid)

    def test_retired_or_pending_frontier_is_unsolved(self) -> None:
        for state in (CognitiveState.RETIRE_PENDING, CognitiveState.RETIRED):
            coordinator, _strategy_uid = self._coordinator(state, CognitiveState.ACTIVE)
            self.assertEqual(
                coordinator.game_state("g"),
                adaptive.GameLearningState.UNSOLVED,
            )

    def test_sidecar_lifecycle_pair_blocks_retired_and_allows_quarantine_only_explicitly(self) -> None:
        strategy_uid = MemoryUid(1, 1)
        outcome_uid = MemoryUid(2, 2)
        index = {
            strategy_uid: SimpleNamespace(cognitive_state=int(CognitiveState.QUARANTINED)),
            outcome_uid: SimpleNamespace(cognitive_state=int(CognitiveState.ACTIVE)),
        }
        self.assertEqual(
            integration._pair_class(index, strategy_uid, outcome_uid),
            "QUARANTINED",
        )
        index[strategy_uid].cognitive_state = int(CognitiveState.RETIRED)
        self.assertEqual(
            integration._pair_class(index, strategy_uid, outcome_uid),
            "BLOCKED",
        )
        self.assertEqual(
            integration._pair_class(index, MemoryUid.zero(), outcome_uid),
            "BLOCKED",
        )

    def test_lifecycle_off_bypasses_only_dedicated_lifecycle_start(self) -> None:
        calls = []
        prior_base = switch._BASE_SUPERVISOR_START
        switch._BASE_SUPERVISOR_START = lambda _self: calls.append("lifecycle")
        try:
            with patch.object(
                DevelopmentalPeerSupervisor,
                "start",
                new=lambda _self: calls.append("peers"),
            ):
                os.environ[switch.LIFECYCLE_ENV] = "off"
                switch._supervisor_start_v827(object())
                os.environ[switch.LIFECYCLE_ENV] = "on"
                switch._supervisor_start_v827(object())
        finally:
            switch._BASE_SUPERVISOR_START = prior_base
        self.assertEqual(calls, ["peers", "lifecycle"])

    def test_cli_accepts_lifecycle_off(self) -> None:
        from v8 import cli

        observed = {}

        def fake_run(args):
            observed["lifecycle"] = args.lifecycle
            return 0

        with patch.object(cli, "run_continuous", side_effect=fake_run):
            code = cli.main(["continuous-run", "--lifecycle", "off"])
        self.assertEqual(code, 0)
        self.assertEqual(observed["lifecycle"], "off")

    @staticmethod
    def _row(game, trajectory_id, *, prefix=(), actions=(), levels_completed=1, terminal="LEVEL"):
        return optimizer.SuccessfulTrajectory(
            trajectory_id,
            optimizer.ReplayAnchor(game, 0, tuple(prefix), None),
            optimizer.TrajectoryTarget(levels_completed, terminal),
            tuple(actions),
            MemoryUid.zero(),
            MemoryUid.zero(),
            0,
        )

    def test_complete_win_history_survives_inbox_consumption_and_is_visible(self) -> None:
        run_root = Path(tempfile.mkdtemp())
        trajectory_root = run_root / "trajectory_optimizer"
        os.environ["ARC_AGI3_V8_TRAJECTORY_ROOT"] = str(trajectory_root)

        optimizer._write_successful_trajectory(
            self._row("g", "l0", actions=(1, 2), levels_completed=1)
        )
        optimizer._write_successful_trajectory(
            self._row(
                "g",
                "win",
                prefix=(1, 2),
                actions=(3,),
                levels_completed=2,
                terminal="WIN",
            )
        )

        history = tuple((trajectory_root / "solutions_history").glob("*.json"))
        self.assertEqual(len(history), 1)
        for directory in ("inbox", "solutions_inbox"):
            for path in (trajectory_root / directory).glob("*.json"):
                path.unlink()

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = inspection.show_best_trajectory(run_root, "g")
        self.assertEqual(code, 0)
        self.assertIn("game=g cost=3 source=observed reliability=1.000", stream.getvalue())
        self.assertIn("L0: A1,A2", stream.getvalue())
        self.assertIn("L1: A3", stream.getvalue())

        missing = io.StringIO()
        with redirect_stdout(missing):
            code = inspection.show_best_trajectory(run_root, "other")
        self.assertEqual(code, 1)
        self.assertEqual(
            missing.getvalue().strip(),
            "game=other no successful trajectory found; available=g",
        )


if __name__ == "__main__":
    unittest.main()
