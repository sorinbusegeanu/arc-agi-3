from __future__ import annotations

from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleIndexMutation
from v7.memory.models import NodeMutation
from v7.memory.state import CognitiveState, GateId, GateValidationState
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter


def test_probe_only_state_and_index_survive_mmap_transport(tmp_path) -> None:
    writer = CanonicalMemoryWriter(gate_candidates=True)
    memory_id = MemoryId(5)
    writer.apply_mutation_batch(
        (
            NodeMutation(
                memory_id,
                MemoryLevel.M3,
                300,
                support_delta=3,
                cognitive_state=int(CognitiveState.PROBE_ONLY),
                validation_state=int(GateValidationState.PROBE_ELIGIBLE),
                gate_id=int(GateId.G23R),
            ),
        )
    )
    writer.apply_role_index_batch((RoleIndexMutation(77, 4, memory_id, None),))
    _state, view, _delta = writer.commit_generation()

    transport = SegmentedMmapReadViewTransport(tmp_path / "segments")
    handle = transport.publish(view)
    attached = transport.attach(handle)

    node = attached.nodes[memory_id]
    assert node.cognitive_state == int(CognitiveState.PROBE_ONLY)
    assert node.validation_state == int(GateValidationState.PROBE_ELIGIBLE)
    assert node.gate_id == int(GateId.G23R)
    assert memory_id not in attached.score_inputs(
        context_signature=77, action_ids=(4,)
    )[0].role_ids
    assert memory_id in attached.probe_score_inputs(
        context_signature=77, action_ids=(4,)
    )[0].role_ids
