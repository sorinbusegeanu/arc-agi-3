from __future__ import annotations

from typing import Dict, List

from codex_baseline_v2.shared.config import SequenceMiningConfigV2
from codex_baseline_v2.shared.schemas import ChangeEventV2, EffectSignatureV2, TopologyDeltaV2, SCHEMA_VERSION


def _delay_bucket(step_idx: int, cfg: SequenceMiningConfigV2) -> str:
    edges = list(cfg.delay_bucket_edges)
    if step_idx <= edges[1]:
        return "immediate"
    if step_idx <= edges[2]:
        return "delayed"
    return "post_transition"


def build_effect_signatures(events: List[ChangeEventV2], topology_deltas: List[TopologyDeltaV2], cfg: SequenceMiningConfigV2) -> list[EffectSignatureV2]:
    topo_by_event: Dict[str, TopologyDeltaV2] = {delta.event_id: delta for delta in topology_deltas}
    signatures: Dict[str, EffectSignatureV2] = {}
    for event in events:
        topo = topo_by_event.get(event.event_id)
        area_relation = "same_area" if event.pre_area_id == event.post_area_id else "cross_area"
        delta_types = sorted({delta.delta_type for delta in event.object_state_deltas})
        delay_bucket = _delay_bucket(max(0, event.end_step_idx - event.start_step_idx), cfg)
        signature_id = f"effect_signature:{event.event_type}:{event.locality}:{area_relation}:{delay_bucket}:{topo.delta_id if topo is not None else 'none'}"
        signatures[signature_id] = EffectSignatureV2(
            schema_version=SCHEMA_VERSION,
            game_id=event.game_id,
            effect_signature_id=signature_id,
            event_type=event.event_type,
            locality=event.locality,
            area_relation=area_relation,
            topology_effect_type=topo.delta_id if topo is not None else None,
            object_delta_types=delta_types,
            delay_bucket=delay_bucket,
            confidence=max(signatures[signature_id].confidence, event.confidence) if signature_id in signatures else event.confidence,
        )
    return list(signatures.values())
