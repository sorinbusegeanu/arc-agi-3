from __future__ import annotations

from typing import Any

from .memory_pipeline import DerivationTask


def _formation_eligible(row: Any) -> bool:
    support = float(getattr(row, "support", 1.0))
    contradiction = float(getattr(row, "contradiction", 0.0))
    return support > contradiction and support > 0.0


def _family_rows(runtime: Any, family_signature: int) -> tuple[Any, ...]:
    index = getattr(runtime, "_m1n_family_occurrences", None)
    if isinstance(index, dict):
        return tuple(index.get(int(family_signature), ()))

    # Compatibility fallback for runtimes/tests that have not initialized the
    # incremental family index yet. The production canonical path initializes
    # the index in record_normalized_fast before derivation candidates form.
    rows: list[Any] = []
    for occurrences in runtime._m1n_occurrences.values():
        for row in occurrences:
            if int(row.family_signature or row.structural_signature) == int(family_signature):
                rows.append(row)
    return tuple(rows)


def derivation_candidates(runtime: Any, signatures: set[int]) -> tuple[DerivationTask, ...]:
    touched_rows = [
        row
        for signature in sorted(signatures)
        for row in runtime._m1n_occurrences.get(int(signature), ())
        if _formation_eligible(row)
    ]
    touched_families = {
        int(row.family_signature or row.structural_signature)
        for row in touched_rows
    }

    # Only inspect indexed rows for families touched by this canonical batch.
    # The old implementation scanned every retained M1N occurrence on every
    # batch, making canonical commit cost grow with total accumulated memory.
    by_family: dict[int, list[Any]] = {}
    for family_signature in sorted(touched_families):
        bucket: list[Any] = []
        seen: set[tuple[Any, str, tuple[Any, ...]]] = set()
        for row in _family_rows(runtime, family_signature):
            if not _formation_eligible(row):
                continue
            identity = (row.uid, row.channel.value, tuple(row.provenance.evidence))
            if identity in seen:
                continue
            seen.add(identity)
            bucket.append(row)
        if bucket:
            by_family[family_signature] = bucket

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
