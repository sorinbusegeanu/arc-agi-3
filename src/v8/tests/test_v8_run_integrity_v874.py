from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import run_integrity_v874 as v874
from v8 import runtime_win_optimization_v834 as v834
from v8 import runtime_win_scope_v835 as v835
from v8 import trajectory_optimizer_v814 as optimizer
from v8.evaluation_v82 import V82HypothesisDecision


def decision(hypothesis_id: str, final: str) -> V82HypothesisDecision:
    return V82HypothesisDecision(
        hypothesis_id=hypothesis_id,
        raw_decision=final,
        quality_gate="PASS",
        dependency_gate="PASS",
        final_decision=final,
        evidence_count=1,
        blocker="",
        paper_claim="test",
        required_measurements=(),
        falsification_measurements=(),
        ordering_gate="PASS",
    )


class FormationTelemetryRepairTests(unittest.TestCase):
    def test_runtime_metrics_reads_m2_from_production_promotion_estimator(self) -> None:
        runtime = SimpleNamespace(
            peers=SimpleNamespace(
                compression=SimpleNamespace(),
                promotion=SimpleNamespace(
                    generative_compression=SimpleNamespace(
                        _v870_formation_telemetry={
                            "m1n_count": 17,
                            "eligible_m2_groups": 2,
                            "m2_candidates_emitted": 2,
                        }
                    )
                ),
                roles=SimpleNamespace(
                    _v870_formation_telemetry={
                        "m3_carrier_count": 3,
                        "role_candidates": 1,
                    }
                ),
            )
        )
        with patch.object(
            v874,
            "_BASE_RUNTIME_METRICS",
            lambda _runtime: {"formation_telemetry": {}},
        ):
            metrics = v874._formation_metrics_v874(runtime)

        telemetry = metrics["formation_telemetry"]
        self.assertEqual(telemetry["m1n_count"], 17)
        self.assertEqual(telemetry["m2_candidates_emitted"], 2)
        self.assertEqual(telemetry["m3_carrier_count"], 3)
        self.assertEqual(telemetry["role_candidates"], 1)


class DevelopmentalOrderingRepairTests(unittest.TestCase):
    def test_h05_cannot_validate_after_family_before_carrier_failed(self) -> None:
        rows = v874.enforce_role_developmental_order_v874(
            (decision("H04", "INVALID"), decision("H05", "VALID")),
            {
                "family_before_carrier": "FAIL",
                "carrier_before_role": "PASS",
            },
        )
        by_id = {row.hypothesis_id: row for row in rows}

        self.assertEqual(by_id["H04"].final_decision, "INVALID")
        self.assertEqual(by_id["H05"].final_decision, "INVALID")
        self.assertEqual(by_id["H05"].ordering_gate, "FAIL")
        self.assertIn("family_before_carrier", by_id["H05"].blocker)

    def test_independent_causal_hypothesis_is_not_status_gated(self) -> None:
        rows = v874.enforce_role_developmental_order_v874(
            (decision("H05", "INVALID"), decision("H06", "VALID")),
            {"family_before_carrier": "FAIL"},
        )
        by_id = {row.hypothesis_id: row for row in rows}
        self.assertEqual(by_id["H06"].final_decision, "VALID")


class VerifiedWinAllocationRepairTests(unittest.TestCase):
    def test_verified_worker_win_uses_existing_v834_runtime_win_channel(self) -> None:
        self.assertIs(v819.AdaptiveLearningCoordinator.game_state, v834._game_state_v834)
        with tempfile.TemporaryDirectory() as directory:
            verified_root = os.path.join(directory, "verified")
            trajectory_root = os.path.join(directory, "trajectory_optimizer")
            with patch.dict(
                os.environ,
                {
                    "ARC_AGI3_V8_VERIFIED_SUCCESS_ROOT": verified_root,
                    optimizer._TRAJECTORY_ROOT_ENV: trajectory_root,
                    v835._RUN_SESSION_ENV: "v874-test-run",
                },
                clear=False,
            ):
                coordinator = v819.AdaptiveLearningCoordinator()
                coordinator.register_games(("tp01",))
                coordinator._v834_runtime = SimpleNamespace(generation=123)
                self.assertEqual(
                    coordinator.game_state("tp01"),
                    v819.GameLearningState.UNSOLVED,
                )

                persisted = v874._record_verified_success_v874(
                    game_id="tp01",
                    seed=7,
                    terminal_state="WIN",
                    levels_completed=5,
                    actions=(1, 2, 3),
                    capture_step=3,
                )
                self.assertTrue(persisted)

                self.assertEqual(
                    coordinator.game_state("tp01"),
                    v819.GameLearningState.SOLVED_OPTIMIZING,
                )
                self.assertEqual(
                    coordinator.choose_mode("tp01"),
                    v819.SamplingMode.VERIFY,
                )
                self.assertTrue(coordinator._game_won["tp01"])
                full_win = coordinator._record("tp01", v835._FULL_WIN_SCOPE_LEVEL)
                self.assertEqual(
                    full_win.state,
                    v819.GameLearningState.SOLVED_OPTIMIZING,
                )


if __name__ == "__main__":
    unittest.main()
