from __future__ import annotations

import json

import numpy as np

from v7.derivation.scientific import EpisodeEvidence, ScientificDerivationKernels, world_transition_signature
from v7.environment.ablation import CognitionAblation, ablation_names
from v7.environment.encoding import carrier_signature, transition_signature
from v7.experiment import _append_hypothesis_log
from v7.hypotheses import _Snapshot, _h09, _h10, _h11, evaluate_hypothesis_suite
from v7.memory.evidence_store import EvidenceRecord
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryLevel
from v7.runtime import V7Runtime, V7RuntimeConfig


def test_transition_and_carrier_signatures_ignore_absolute_position() -> None:
    before_a=np.zeros((5,5),dtype=np.int64); after_a=before_a.copy(); before_a[1,1]=3; after_a[1,1]=4
    before_b=np.zeros((5,5),dtype=np.int64); after_b=before_b.copy(); before_b[3,2]=3; after_b[3,2]=4
    assert transition_signature(before_a,after_a)==transition_signature(before_b,after_b)
    assert carrier_signature(before_a,after_a)==carrier_signature(before_b,after_b)


def test_higher_order_canonical_keys_ignore_support_population_but_m3_keeps_context() -> None:
    m2a=ScientificDerivationKernels.m2_family(action_id=4,member_ids=(1,2),outcome_class=9); m2b=ScientificDerivationKernels.m2_family(action_id=4,member_ids=(1,2,3),outcome_class=9); assert m2a.key==m2b.key
    m3a=ScientificDerivationKernels.m3_role(family_id=10,context_class=100,action_id=4,member_ids=(1,2)); m3b=ScientificDerivationKernels.m3_role(family_id=10,context_class=100,action_id=4,member_ids=(1,2,3)); m3c=ScientificDerivationKernels.m3_role(family_id=10,context_class=200,action_id=4,member_ids=(1,2,3)); assert m3a.key==m3b.key; assert m3a.key!=m3c.key
    m4a=ScientificDerivationKernels.m4_concept(role_ids=(11,12),relation_signature=99); m4b=ScientificDerivationKernels.m4_concept(role_ids=(11,12,13),relation_signature=99); assert m4a.key==m4b.key


def test_epoch_finalization_derives_m2_m3_m4_from_direct_evidence(tmp_path) -> None:
    runtime=V7Runtime(V7RuntimeConfig.from_path(tmp_path,restore=False))
    try:
        runtime.observe_batch((EpisodeEvidence(1,1,100,True,source_game='g1',source_context='1',source_global_step=1,carrier_signature=77),EpisodeEvidence(2,1,100,True,source_game='g1',source_context='2',source_global_step=2,carrier_signature=77),EpisodeEvidence(3,2,200,True,source_game='g2',source_context='3',source_global_step=3,carrier_signature=77),EpisodeEvidence(4,2,200,True,source_game='g2',source_context='4',source_global_step=4,carrier_signature=77)))
        result=runtime.commit(run_lifecycle=False,derive_hierarchy=True); levels=[node.level for node in result.view.nodes.values()]; assert MemoryLevel.M2 in levels; assert MemoryLevel.M3 in levels; assert MemoryLevel.M4 in levels
    finally: runtime.close()


def test_hypothesis_reports_include_real_metrics_and_issue_diagnostics(tmp_path) -> None:
    runtime=V7Runtime(V7RuntimeConfig.from_path(tmp_path,restore=False))
    try:
        runtime.observe_batch((EpisodeEvidence(1,1,100,True,prediction_error=1.0,source_game='g1',source_context='1',source_global_step=1,carrier_signature=77),EpisodeEvidence(2,1,100,True,source_game='g1',source_context='2',source_global_step=2,carrier_signature=77),EpisodeEvidence(3,2,200,True,source_game='g2',source_context='3',source_global_step=3,carrier_signature=77),EpisodeEvidence(4,2,200,True,source_game='g2',source_context='4',source_global_step=4,carrier_signature=77)))
        runtime.commit(run_lifecycle=True,derive_hierarchy=True); reports=evaluate_hypothesis_suite(runtime,epoch=0,output_root=tmp_path,workers=2); assert set(reports)=={f'H{i:02d}' for i in range(1,13)}; assert reports['H03']['evidence']['measurement']['family_count']>=2; assert reports['H04']['evidence']['measurement']['carrier_candidate_count']>=1; assert reports['H04']['potential_issues']; assert (tmp_path/'reports'/'epoch_0001'/'hypotheses.json').exists()
        blocker_path=tmp_path/'reports'/'epoch_0001'/'hypothesis_blockers.jsonl'; rows=[json.loads(line) for line in blocker_path.read_text(encoding='utf-8').splitlines()]; assert [row['hypothesis_id'] for row in rows]==[f'H{i:02d}' for i in range(1,13)]; assert all(row['blockers'] for row in rows if not row['valid']); assert all(not row['blockers'] and not row['dependency_blockers'] for row in rows if row['valid']); h06=rows[5]; assert any('0/4' in blocker for blocker in h06['blockers']); h08=rows[7]; assert h08['dependency_blockers'] and len(h08['blockers'])>len(h08['dependency_blockers'])
        first=blocker_path.read_text(encoding='utf-8'); evaluate_hypothesis_suite(runtime,epoch=0,output_root=tmp_path,workers=2); assert blocker_path.read_text(encoding='utf-8')==first; assert len(blocker_path.read_text(encoding='utf-8').splitlines())==12
    finally: runtime.close()


def test_hypothesis_blockers_are_written_to_dedicated_log_without_stdout(tmp_path, capsys) -> None:
    reports = {
        'H01': {'final_decision': 'VALID', 'blockers': []},
        'H05': {'final_decision': 'PARTIALLY_VALID', 'blockers': ['usable roles 0/1']},
    }
    _append_hypothesis_log(tmp_path, 2, reports)
    assert capsys.readouterr().out == ''
    assert (tmp_path / 'hypotheis.log').read_text(encoding='utf-8').endswith(
        'E0003] H05 blockers: usable roles 0/1\n'
    )



def test_world_transition_identity_and_future_option_ablation_contract() -> None:
    assert world_transition_signature((1, 2), 3, (4, 5)) == world_transition_signature((1, 2), 3, (4, 5))
    assert world_transition_signature((1, 2), 3, (4, 5)) != world_transition_signature((1, 2), 4, (4, 5))
    assert ablation_names(int(CognitionAblation.FUTURE_OPTION)) == ("future_option",)


def test_h09_uses_direct_option_change_without_proxy(tmp_path) -> None:
    runtime = V7Runtime(V7RuntimeConfig.from_path(tmp_path, restore=False))
    try:
        rows = []
        for step, game, context in ((1, "g1", "a"), (2, "g1", "b"), (3, "g2", "c"), (4, "g2", "d")):
            rows.append(EvidenceRecord(memory_id=None, evidence_type=int(EvidenceType.EPISODE), generation_id=1, payload={"raw_action_option_delta": 1.0, "carrier_signature": 77, "action_id": 2}, source_game=game, source_context=context, source_global_step=step))
        runtime.evidence.append_evidence_batch(rows)
        result = _h09(_Snapshot(runtime))
        assert result["raw_decision"] == "VALID"
        assert result["evidence"]["proxy_only"] is False
    finally:
        runtime.close()


def test_h10_uses_paired_same_state_scorer_ablation(tmp_path) -> None:
    runtime = V7Runtime(V7RuntimeConfig.from_path(tmp_path, restore=False))
    try:
        rows = []
        for step in range(20):
            high = step < 10
            rows.append(EvidenceRecord(memory_id=None, evidence_type=int(EvidenceType.EPISODE), generation_id=1, payload={"raw_action_option_delta": 2.0 if high else 0.0, "future_option_ablation_available": True, "future_option_ablation_score_delta": 0.20 if high else 0.01, "future_option_ablation_rank_lift": 1 if high else 0}, source_game="g1", source_context=str(step), source_global_step=step + 1))
        runtime.evidence.append_evidence_batch(rows)
        result = _h10(_Snapshot(runtime))
        assert result["raw_decision"] == "VALID"
        assert result["evidence"]["measurement"]["causal_attention_lift"] >= 1.25
    finally:
        runtime.close()


def test_h11_uses_validation_scope_not_single_source_provenance() -> None:
    class Stub:
        concept_validations = [{"memory_id": 10, "validated": True, "generation_id": 5, "validation_source_games": ["g1", "g0"]}]
        transfer_trials = [
            {"memory_id": 10, "generation_id": 6, "target_game": "g2", "source_game": "g0", "source_game_count": 2, "success": True, "attribution": "trajectory_usage", "raw_action_option_delta": 1.0},
            {"memory_id": 10, "generation_id": 7, "target_game": "g3", "source_game": "g1", "source_game_count": 3, "success": True, "attribution": "trajectory_usage", "raw_action_option_delta": -1.0},
        ]
        def nodes_at(self, level, type_id=None):
            return [(10, object())] if level == MemoryLevel.M4 else []
    result = _h11(Stub())
    assert result["raw_decision"] == "VALID"
    assert result["evidence"]["measurement"]["distinct_post_validation_target_games"] == 2
