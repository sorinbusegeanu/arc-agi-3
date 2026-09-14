from __future__ import annotations

from typing import Any

from .memory_pipeline import DerivationTask


def derivation_candidates(runtime: Any, signatures: set[int]) -> tuple[DerivationTask, ...]:
    candidates: list[DerivationTask] = []
    for signature in sorted(signatures):
        rows = tuple(runtime._m1n_occurrences.get(int(signature), ()))
        support = int(runtime._m1n_supports.get(int(signature), len(rows)))
        evidence = {uid for row in rows for uid in row.provenance.evidence}
        if len(rows) < 2 or support < 2 or len(evidence) < 2:
            continue

        environments: set[int] = set()
        for uid in evidence:
            payload = runtime.graph.payloads.get(uid)
            if payload is None:
                deferred = runtime._deferred_base_nodes.get(uid)
                payload = None if deferred is None else deferred[1]
            if payload is not None and payload.get("environment_instance_id") is not None:
                environments.add(int(payload["environment_instance_id"]))
        scope = tuple(sorted(environments or runtime._formation_environments))
        candidates.append(
            DerivationTask(
                0,
                int(signature),
                rows,
                support,
                scope,
                int(runtime._watermark),
            )
        )
    return tuple(candidates)
