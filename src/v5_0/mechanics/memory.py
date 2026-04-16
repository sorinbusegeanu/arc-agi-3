from __future__ import annotations

from v5_0.contracts.avatar_types import MechanicMemory, POIMechanicState


def build_mechanic_memory(
    classified_poi_states,
    previous_memory_state=None,
) -> MechanicMemory:
    previous_by_id = {}
    retired = set()
    selected_prev = None
    if previous_memory_state is not None:
        previous_by_id = {str(item.poi_id): item for item in tuple(getattr(previous_memory_state, "poi_states", ()))}
        retired = set(str(v) for v in tuple(getattr(previous_memory_state, "retired_poi_ids", ())))
        selected_prev = getattr(previous_memory_state, "selected_poi_id", None)

    merged: list[POIMechanicState] = []
    for item in sorted(tuple(classified_poi_states or ()), key=lambda v: str(v.poi_id)):
        poi_id = str(item.poi_id)
        prev = previous_by_id.get(poi_id)
        attempt_count = int(item.attempt_count) + int(getattr(prev, "attempt_count", 0))
        success_count = int(item.success_count) + int(getattr(prev, "success_count", 0))
        failure_count = int(item.failure_count) + int(getattr(prev, "failure_count", 0))
        last_outcome = item.last_outcome_type if item.last_outcome_type is not None else getattr(prev, "last_outcome_type", None)
        active = bool(item.active)

        if item.mechanic_label == "decoy" and failure_count >= 3 and success_count == 0:
            retired.add(poi_id)
            active = False
        elif item.mechanic_label == "unknown":
            active = True

        merged.append(
            POIMechanicState(
                poi_id=poi_id,
                mechanic_label=str(item.mechanic_label),
                confidence=float(item.confidence),
                priority_score=float(item.priority_score),
                attempt_count=int(attempt_count),
                success_count=int(success_count),
                failure_count=int(failure_count),
                last_outcome_type=last_outcome,
                active=bool(active),
            )
        )

    selected_poi_id = selected_prev
    active_ids = {item.poi_id for item in merged if item.active and item.poi_id not in retired}
    if selected_poi_id not in active_ids:
        selected_poi_id = None

    return MechanicMemory(
        poi_states=tuple(merged),
        selected_poi_id=selected_poi_id,
        retired_poi_ids=tuple(sorted(retired)),
    )
