from __future__ import annotations

from types import SimpleNamespace

from v9.hgt.grounding_objectives import GROUNDING_OBJECTIVES, GroundingObjectiveEvidence, validate_objective_evidence
from v9.memory.identity import MemoryUid
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.m2_family import M2TransformationFamily
from v9.memory.m3_role import M3FunctionalRole
from v9.memory.v978_descriptors import modality_neutral_family_signature
from v9.research.h16_reproducible import materialize_h16_matched_state


def _relation(channel: NormalizedChannel, low: int, family: int, *, support: float = 1.0, contradiction: float = 0.0):
    parent = MemoryUid(1, low)
    return M1NormalizedRelation.from_provenance(
        f"REL-{low}",
        channel,
        parents=(parent,),
        evidence=(parent,),
        structural_key=("REL", channel.value, low),
        family_signature=family,
        support=support,
        contradiction=contradiction,
        causal_watermark=low,
    )


def test_modality_neutral_descriptor_ignores_action_and_environment_identity() -> None:
    left = SimpleNamespace(
        observable_relation="ACTION:10:env-a:4:SEM_ACTION:1:SEM_DELTA:2:FAMILY:3:OUTCOME:4:OPTIONS:5:BOUNDARY:TASK:SUCCESS:1:FAILURE:0:TRUNCATED:0:LEVEL:2:LEVELS_COMPLETED:1"
    )
    right = SimpleNamespace(
        observable_relation="ACTION:99:env-b:123:SEM_ACTION:500:SEM_DELTA:900:FAMILY:700:OUTCOME:800:OPTIONS:600:BOUNDARY:TASK:SUCCESS:1:FAILURE:0:TRUNCATED:0:LEVEL:20:LEVELS_COMPLETED:1"
    )
    payload_a = {"semantic_effects": [[5, 100, 2, 200, 1.0]]}
    payload_b = {"semantic_effects": [[5, 999, 2, 888, 1.0]]}
    assert modality_neutral_family_signature(left, payload_a) == modality_neutral_family_signature(right, payload_b)


def test_world_symbol_cross_modal_evidence_forms_one_m2_and_m3() -> None:
    family_signature = 123456
    rows = (
        _relation(NormalizedChannel.WORLD, 10, family_signature),
        _relation(NormalizedChannel.SYMBOL, 11, family_signature),
        _relation(NormalizedChannel.CROSS_MODAL, 12, family_signature),
    )
    family = M2TransformationFamily.form(rows)
    support = dict(family.support_decomposition)
    assert support == {
        "aligned_cross_modal": 1,
        "heldout_transfer": 0,
        "interaction_only": 1,
        "symbol_only": 1,
    }
    role = M3FunctionalRole.form((family,), consequence_signature=77)
    assert dict(role.support_decomposition) == support


def test_contradictory_cross_modal_evidence_remains_identifiable() -> None:
    row = _relation(NormalizedChannel.CROSS_MODAL, 20, 42, support=0.0, contradiction=1.0)
    assert row.support <= row.contradiction
    assert row.causal_watermark == 20


def test_every_grounding_objective_has_positive_negative_control_contract() -> None:
    rows = []
    for objective in GROUNDING_OBJECTIVES:
        rows.append(GroundingObjectiveEvidence(objective, 1.0, 0.9, 10, 11, f"{objective}:positive", "aligned"))
        rows.append(GroundingObjectiveEvidence(objective, 0.0, 0.1, 10, 11, f"{objective}:negative", "shuffled"))
    result = validate_objective_evidence(rows, require_complete=True)
    assert result["objectives"] == len(GROUNDING_OBJECTIVES)
    assert result["positive_examples"] == len(GROUNDING_OBJECTIVES)
    assert result["negative_examples"] == len(GROUNDING_OBJECTIVES)
    assert result["control_groups"] == 2


def test_matched_h16_source_state_is_persisted_and_reproducible(tmp_path) -> None:
    first_path, first_digest = materialize_h16_matched_state(
        tmp_path,
        seed=7,
        environment_config_id=3,
        interaction_budget=24,
        evaluation_id=2,
    )
    second_path, second_digest = materialize_h16_matched_state(
        tmp_path,
        seed=7,
        environment_config_id=3,
        interaction_budget=24,
        evaluation_id=2,
    )
    assert first_path == second_path
    assert first_digest == second_digest
    assert first_path.exists()


def test_public_runtime_is_final_conformance_runtime() -> None:
    import v9
    import v9.runtime
    from v9.runtime.v978_conformant import V978ContinuousMemoryRuntime

    assert v9.ContinuousMemoryRuntime is V978ContinuousMemoryRuntime
    assert v9.runtime.ContinuousMemoryRuntime is V978ContinuousMemoryRuntime
