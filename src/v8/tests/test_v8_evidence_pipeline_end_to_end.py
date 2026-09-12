from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v8 import ContinuousMemoryRuntime, V8RuntimeConfig
from v8.actor import (
    PreferenceProbeResult,
    ReplanningTrialResult,
    StrategyRunStat,
)
from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    stable_u64,
)
from v8 import strategy_statistics_persistence_fix_v828 as strategy_persistence


class EvidencePipelineEndToEndTests(unittest.TestCase):
    def test_eligible_evidence_reaches_h02_h10_and_h12_through_h15(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prior_root = os.environ.get("ARC_AGI3_V8_ROOT")
            os.environ["ARC_AGI3_V8_ROOT"] = tmp
            config = V8RuntimeConfig.from_path(
                Path(tmp),
                shards=1,
                stage_workers=1,
                enable_snapshots=True,
                restore=True,
                peer_interval_seconds=3600.0,
                node_capacity_per_shard=2000,
                edge_capacity_per_shard=6000,
                action_capacity_per_shard=256,
            )
            runtime = ContinuousMemoryRuntime(config)
            closed = False
            assert runtime.peers is not None
            runtime.peers.pause()
            runtime.start()
            sequence = 0

            def submit(
                level: MemoryLevel,
                memory_type: MemoryType,
                key: tuple[int, ...],
                *,
                support: int = 1,
                prediction_error: float = 0.0,
                parent: MemoryUid = MemoryUid.zero(),
                relation: RelationType = RelationType.PROVENANCE,
                game: int = 0,
                attempts: float = 0.0,
                successes: float = 0.0,
                cost: float = 0.0,
            ) -> MemoryUid:
                nonlocal sequence
                sequence += 1
                uid = MemoryUid.from_key(level, memory_type, key)
                runtime.submit_proposal(MemoryProposal(
                    uid=uid,
                    fingerprint=proposal_fingerprint(level, memory_type, key),
                    event_id=EventId.from_producer(901, sequence),
                    watermark=sequence,
                    level=level,
                    memory_type=memory_type,
                    key_parts=key,
                    support_delta=support,
                    significance_sum=float(support),
                    prediction_error_sum=float(prediction_error),
                    learning_value_sum=float(support),
                    score_weight=float(max(1, support)),
                    success_sum=successes,
                    cost_sum=cost,
                    attempt_weight=attempts,
                    parent_uid=parent,
                    relation_type=relation,
                    source_game_hash=game,
                    cognitive_state=int(
                        CognitiveState.VALIDATED
                        if level == MemoryLevel.M7 else CognitiveState.ACTIVE
                    ),
                    validation_state=int(
                        ValidationState.VALIDATED
                        if level == MemoryLevel.M7 else ValidationState.STRUCTURAL
                    ),
                ))
                return uid

            try:
                expected = MemoryUid.from_key(
                    MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 1, 100, 20)
                )
                contradiction = MemoryUid.from_key(
                    MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 1, 101, 21)
                )
                for game in (10, 20):
                    for _ in range(4):
                        submit(MemoryLevel.M1, MemoryType.CONTINGENCY,
                               (10, 1, 100, 20), game=game)
                    for _ in range(2):
                        submit(MemoryLevel.M1, MemoryType.CONTINGENCY,
                               (10, 1, 101, 21), prediction_error=1.0, game=game)

                outcome_a = submit(
                    MemoryLevel.M6, MemoryType.OUTCOME, (500, 500, 1),
                    support=4, parent=expected, relation=RelationType.LEADS_TO,
                )
                outcome_b = submit(
                    MemoryLevel.M6, MemoryType.OUTCOME, (500, 500, 2),
                    support=4, parent=contradiction, relation=RelationType.LEADS_TO,
                )
                context_bucket = stable_u64(10, person=b"v8-context")
                strategy_a = submit(
                    MemoryLevel.M7, MemoryType.STRATEGY,
                    (1, outcome_a.hi, outcome_a.lo, context_bucket),
                    support=4, parent=outcome_a, relation=RelationType.DEPENDS_ON,
                )
                strategy_b = submit(
                    MemoryLevel.M7, MemoryType.STRATEGY,
                    (2, outcome_a.hi, outcome_a.lo, context_bucket),
                    support=4, parent=outcome_a, relation=RelationType.DEPENDS_ON,
                )
                runtime.wait_quiescent(timeout=20)
                runtime.read_view.node_records()

                probes = tuple(
                    PreferenceProbeResult(
                        outcome_a, outcome_b, context_bucket, outcome_a, False, True
                    )
                    for _ in range(6)
                )
                runtime.record_actor_results((SimpleNamespace(
                    actor_id=7,
                    game_id="synthetic-evidence-world",
                    strategy_stats=(
                        StrategyRunStat(strategy_a, 3, 2, 3.0),
                        StrategyRunStat(strategy_b, 3, 1, 6.0),
                    ),
                    preference_probes=probes,
                    replanning_trials=(ReplanningTrialResult(
                        strategy_a, strategy_b, outcome_a, True
                    ),),
                ),))
                runtime.wait_quiescent(timeout=20)
                committed = {
                    row.uid: row for row in runtime.read_view.node_records()
                }
                self.assertEqual(committed[strategy_a].attempt_weight, 3.0)
                self.assertEqual(committed[strategy_b].attempt_weight, 3.0)
                self.assertIn(committed[strategy_a].cognitive_state, (
                    int(CognitiveState.ACTIVE), int(CognitiveState.VALIDATED),
                    int(CognitiveState.REACTIVATED),
                ))
                runtime.read_view.invalidate_strategy_cache()
                runtime.read_view._refresh_strategy_cache()
                self.assertEqual(
                    len(runtime.read_view._strategy_by_context.get(context_bucket, ())), 2,
                    runtime.read_view._strategy_by_context,
                )
                with patch.object(
                    strategy_persistence,
                    "_emit_committed_strategy_efficiency",
                    wraps=strategy_persistence._emit_committed_strategy_efficiency,
                ) as emit_efficiency:
                    runtime.peers.run_once()
                self.assertEqual(emit_efficiency.call_count, 1)
                runtime.wait_quiescent(timeout=20)

                kinds = {
                    row.evidence_kind
                    for row in runtime.peers.ledger.cut(2**63 - 1)
                }
                self.assertIn("prediction_violation", kinds)
                self.assertIn("context_refinement_gain", kinds)
                self.assertIn("strategy_efficiency", kinds)
                self.assertIn("outcome_consistency_holdout", kinds)
                self.assertIn("replanning_recovery_trial", kinds)
                self.assertIn("stable_preference_probe", kinds)
                statuses = runtime.scientific_statuses()
                for hypothesis in ("H02", "H10", "H12", "H13", "H14", "H15"):
                    self.assertEqual(statuses[hypothesis], "VALID", hypothesis)
                runtime.write_scientific_report()
                self.assertTrue((Path(tmp) / "reports" / "h01_h15.json").is_file())
                final = runtime.close(normal=True, timeout=30)
                closed = True
                self.assertIsNotNone(final)

                restored = ContinuousMemoryRuntime(config)
                try:
                    restored.start()
                    restored_statuses = restored.scientific_statuses()
                    for hypothesis in ("H02", "H10", "H12", "H13", "H14", "H15"):
                        self.assertEqual(restored_statuses[hypothesis], "VALID", hypothesis)
                finally:
                    restored.close(normal=False)
            finally:
                if not closed:
                    runtime.close(normal=False)
                if prior_root is None:
                    os.environ.pop("ARC_AGI3_V8_ROOT", None)
                else:
                    os.environ["ARC_AGI3_V8_ROOT"] = prior_root


if __name__ == "__main__":
    unittest.main()
