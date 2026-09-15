from __future__ import annotations

from typing import Any

from .memory_pipeline import DerivationTask


def derivation_candidates(runtime: Any, signatures: set[int]) -> tuple[DerivationTask, ...]:
    touched_rows = [
        row
        for signature in sorted(signatures)
        for row in runtime._m1n_occurrences.get(int(signature), ())
    ]
    by_family: dict[int, list[Any]] = {}
    for row in touched_rows:
        family = int(row.family_signature or row.structural_signature)
        bucket = by_family.setdefault(family, [])
        identity = (row.uid, row.channel.value, tuple(row.provenance.evidence))
        if all((existing.uid, existing.channel.value, tuple(existing.provenance.evidence)) != identity for existing in bucket):
            bucket.append(row)

    # Untouched recurrent rows may belong to a family touched through another
    # channel. Add them so WORLD/SYMBOL/CROSS_MODAL evidence can actually meet.
    touched_families = set(by_family)
    for rows in runtime._m1n_occurrences.values():
        for row in rows:
            family = int(row.family_signature or row.structural_signature)
            if family not in touched_families:
                continue
            bucket = by_family.setdefault(family, [])
            identity = (row.uid, row.channel.value, tuple(row.provenance.evidence))
            if all((existing.uid, existing.channel.value, tuple(existing.provenance.evidence)) != identity for existing in bucket):
                bucket.append(row)

    candidates: list[DerivationTask] = []
    for family_signature in sorted(by_family):
        rows = tuple(by_family[family_signature])
        evidence = {uid for row in rows for uid in row.provenance.evidence}
        if len(rows) < 2 or len(evidence) < 2:
            continue
        support = len(rows)
        environments: set[int] = set()
        confidence_values: list[float] = []
        for uid in evidence:
            payload = runtime.graph.payloads.get(uid)
            if payload is None:
                deferred = runtime._deferred_base_nodes.get(uid)
                payload = None if deferred is None else deferred[1]
            if payload is None:
                continue
            if payload.get("environment_instance_id") is not None:
                environments.add(int(payload["environment_instance_id"]))
            confidence_values.append(float(payload.get("evidence_confidence", 1.0)))
        scope = tuple(sorted(environments or runtime._formation_environments))
        evidence_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 1.0
        candidates.append(
            DerivationTask(
                0,
                int(family_signature),
                rows,
                support,
                scope,
                int(runtime._watermark),
                evidence_confidence,
            )
        )
    return tuple(candidates)
