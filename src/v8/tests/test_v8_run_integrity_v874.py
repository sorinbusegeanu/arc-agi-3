from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8.adaptive_learning_allocation_v819 import (
    AdaptiveLearningCoordinator,
    GameLearningState,
    SamplingMode,
)
from v8.evaluation_v82 import V82HypothesisDecision
from v8 import run_integrity_v874 as v874


def decision(
    hypothesis_id: str,
    final: str,
    *,
    evidence_count: int = 1,
) -> V82HypothesisDecision:
    return V82HypothesisDecision(
        hypothesis_id=hypothesis_id,
        raw_decision=final,
        quality_gate="PASS",
        dependency_gate="PASS",
        final_decision=final,
        evidence_count=evidence_count,
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


class TraceabilityDependencyRepairTests(unittest.TestCase):
    def test_h05_cannot_remain_valid_when_h04_is_invalid(self) -> None:
        rows = v874.enforce_traceability_dependencies_v874(
            (
                decision("H04", "INVALID"),
                decision("H05", "VALID"),
            )
        )
        by_id = {row.hypothesis_id: row for row in rows}

        self.assertEqual(by_id["H04"].final_decision, "INVALID")
        self.assertEqual(by_id["H05"].dependency_gate, "BLOCKED")
        self.assertEqual(by_id["H05"].final_decision, "PARTIALLY_VALID")
        self.assertIn("H04=INVALID", by_id["H05"].blocker)

    def test_forward_h12_dependency_on_h13_is_enforced(self) -> None:
        rows = v874.enforce_traceability_dependencies_v874(
            (
                decision("H12", "VALID"),
                decision("H13", "INSUFFICIENT_EVIDENCE", evidence_count=0),
            )
        )
        by_id = {row.hypothesis_id: row for row in rows}

        self.assertEqual(by_id["H12"].dependency_gate, "BLOCKED")
        self.assertEqual(by_id["H12"].final_decision, "PARTIALLY_VALID")
        self.assertIn("H13=INSUFFICIENT_EVIDENCE", by_id["H12"].blocker)


class VerifiedWinAllocationRepairTests(unittest.TestCase):
    def test_worker_verified_win_changes_live_allocator_state_without_frontier(self) -> None:
        old_root = os.environ.get("ARC_AGI3_V8_VERIFIED_SUCCESS_ROOT")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["ARC_AGI3_V8_VERIFIED_SUCCESS_ROOT"] = directory
                coordinator = AdaptiveLearningCoordinator()
                coordinator.register_games(("tp01",))
                self.assertEqual(
                    coordinator.game_state("tp01"),
                    GameLearningState.UNSOLVED,
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
                self.assertTrue(v874._has_verified_win_v874("tp01"))
                self.assertEqual(
                    coordinator.game_state("tp01"),
                    GameLearningState.SOLVED_OPTIMIZING,
                )
                self.assertNotEqual(
                    coordinator.choose_mode("tp01"),
                    SamplingMode.DISCOVERY,
                )

                payload = coordinator.state_dict()
                self.assertTrue(payload["game_won"]["tp01"])
                self.assertAlmostEqual(
                    payload["sampling_weight"]["tp01"],
                    coordinator.config.optimizing_weight,
                )

                restored = AdaptiveLearningCoordinator()
                restored.load_state(payload)
                self.assertEqual(
                    restored.game_state("tp01"),
                    GameLearningState.SOLVED_OPTIMIZING,
                )
        finally:
            if old_root is None:
                os.environ.pop("ARC_AGI3_V8_VERIFIED_SUCCESS_ROOT", None)
            else:
                os.environ["ARC_AGI3_V8_VERIFIED_SUCCESS_ROOT"] = old_root


if __name__ == "__main__":
    unittest.main()
