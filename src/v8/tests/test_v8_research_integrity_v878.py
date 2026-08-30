from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as allocation
from v8 import evaluation
from v8 import intelligence_loop_v087 as intelligence
from v8 import lease_dispatch_lifecycle_v843 as dispatch
from v8 import research_integrity_v878 as v878
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.research import experiment_artifacts


class ResearchIntegrityV878Tests(unittest.TestCase):
    @staticmethod
    def _compression_proposal():
        return intelligence.CompressionProposal(
            uid=MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, (1, 2)),
            key_parts=(1, 2),
            parents=(
                MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (1,)),
                MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (2,)),
            ),
            support=8,
            compression_benefit=6.0,
            explanatory_reach=2.0,
            contradiction=0.0,
            future_option_delta=0.0,
        )

    def test_normalized_m2_uses_one_canonical_family_compression_kind(self):
        candidate = intelligence._compression_to_candidate(self._compression_proposal())
        self.assertEqual(candidate.evidence_kind, "family_compression")
        h03 = next(row for row in evaluation.CONTRACTS if row.hypothesis_id == "H03")
        self.assertEqual(h03.required_kinds, ("family_compression",))

    def test_current_run_win_is_visible_to_telemetry_without_graph_refresh(self):
        coordinator = allocation.AdaptiveLearningCoordinator()
        coordinator.register_games(("tp01",))

        def promote(inner, game):
            inner._game_won[str(game)] = True
            row = inner._record(str(game), 1_000_000_000)
            row.state = allocation.GameLearningState.SOLVED_OPTIMIZING
            return True

        with patch.object(v878, "_promote_current_run_win", side_effect=promote):
            state = v878._telemetry_game_state_v878(coordinator, "tp01")
        self.assertEqual(state, allocation.GameLearningState.SOLVED_OPTIMIZING)

    def test_current_run_win_is_promoted_before_dispatch_raw_state_gate(self):
        coordinator = allocation.AdaptiveLearningCoordinator()
        coordinator.register_games(("tp01",))

        def promote(inner, game):
            inner._game_won[str(game)] = True
            row = inner._record(str(game), 1_000_000_000)
            row.state = allocation.GameLearningState.SOLVED_OPTIMIZING
            return True

        def base_choose(inner, game):
            return (
                allocation.SamplingMode.VERIFY
                if inner._game_won.get(str(game), False)
                else allocation.SamplingMode.DISCOVERY
            )

        with (
            patch.object(v878, "_promote_current_run_win", side_effect=promote),
            patch.object(v878, "_BASE_CHOOSE_MODE", side_effect=base_choose),
        ):
            mode = v878._choose_mode_v878(coordinator, "tp01")
        self.assertEqual(mode, allocation.SamplingMode.VERIFY)
        self.assertIs(allocation.AdaptiveLearningCoordinator.choose_mode, dispatch._choose_mode_v843)

    def test_evidence_digest_counts_worlds_not_selected_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"evidence_kind": "a", "source_game_hash": 11},
                        {"evidence_kind": "b", "source_game_hash": 12},
                        {"evidence_kind": "c", "source_game_hash": 12, "target_game_hash": 21},
                    )
                ),
                encoding="utf-8",
            )
            digest = experiment_artifacts._evidence_digest(path)
        self.assertEqual(digest["distinct_source_worlds"], 2)
        self.assertEqual(digest["distinct_target_worlds"], 1)
        self.assertNotIn("distinct_source_games", digest)
        self.assertIn("instance/seed-scoped", digest["provenance_scope_note"])

    def test_optimizer_metrics_explain_source_validation_scope(self):
        payload = v878._optimizer_with_scope(
            {
                "candidates_generated": 0,
                "trajectories_seen": 0,
                "validated_variants": 0,
                "validations": 18,
                "validation_successes": 18,
            }
        )
        self.assertEqual(payload["validations"], 18)
        self.assertIn("source-validation replay", payload["counter_scope_note"])

    def test_runtime_stack_installs_v878_as_final_research_authority(self):
        self.assertIs(
            allocation.AdaptiveLearningCoordinator._v819_telemetry_game_state,
            v878._telemetry_game_state_v878,
        )
        self.assertIs(experiment_artifacts._evidence_digest, v878._evidence_digest_v878)


if __name__ == "__main__":
    unittest.main()
