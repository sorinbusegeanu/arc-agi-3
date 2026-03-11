from __future__ import annotations

import ray


def export_postrun_artifact(storage_agent, *, session_id: str, round_id: int, name: str, payload) -> str:
    persist = getattr(storage_agent, "persist", None)
    if persist is not None and hasattr(persist, "remote"):
        return ray.get(persist.remote(session_id=session_id, round_id=round_id, kind="report", name=name, payload=payload))
    return storage_agent.persist(session_id=session_id, round_id=round_id, kind="report", name=name, payload=payload)
