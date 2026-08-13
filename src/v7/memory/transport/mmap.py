from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ActionAggregate, CognitionIndexes
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.base import ReadViewHandle

_FORMAT_VERSION = 1


def _encode_view(view: MemoryReadView) -> bytes:
    indexes = view.cognition_indexes
    payload = {
        "format_version": _FORMAT_VERSION,
        "generation_id": int(view.generation_id),
        "nodes": [
            [
                int(node.memory_id),
                int(node.level),
                int(node.type_id),
                int(node.created_generation),
                int(node.updated_generation),
                int(node.status_flags),
                int(node.support_count),
            ]
            for node in sorted(view.nodes.values(), key=lambda item: int(item.memory_id))
        ],
        "scores": [
            [
                int(score.memory_id),
                score.significance,
                score.prediction_error,
                score.learning_value,
                score.transfer_prior,
                score.explanatory_potential,
                score.future_option_delta,
            ]
            for score in sorted(view.scores.values(), key=lambda item: int(item.memory_id))
        ],
        "adjacency": [
            [int(source_id), int(relation_type), [int(value) for value in targets]]
            for (source_id, relation_type), targets in sorted(
                view.adjacency.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
            )
        ],
        "contingencies": [
            [context, action, [int(value) for value in values]]
            for (context, action), values in sorted(indexes.contingency_by_context_action.items())
        ],
        "roles_exact": [
            [context, action, int(family), [int(value) for value in values]]
            for (context, action, family), values in sorted(
                indexes.role_by_context_action_family.items(),
                key=lambda item: (item[0][0], item[0][1], int(item[0][2])),
            )
        ],
        "roles_fallback": [
            [context, action, [int(value) for value in values]]
            for (context, action), values in sorted(indexes.role_by_context_action.items())
        ],
        "concepts": [
            [int(role_id), [int(value) for value in values]]
            for role_id, values in sorted(indexes.concepts_by_role.items(), key=lambda item: int(item[0]))
        ],
        "aggregates": [
            [
                int(action_id),
                aggregate.future_option_sum,
                aggregate.future_option_count,
                aggregate.positive_count,
                aggregate.negative_count,
                aggregate.failure_count,
                aggregate.contradiction_count,
            ]
            for action_id, aggregate in sorted(indexes.action_aggregates.items())
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_view(payload: bytes) -> MemoryReadView:
    raw = json.loads(payload)
    if int(raw.get("format_version", -1)) != _FORMAT_VERSION:
        raise ValueError("unsupported mmap read-view format")
    generation_id = GenerationId(int(raw["generation_id"]))
    nodes = {
        MemoryId(row[0]): MemoryNode(
            memory_id=MemoryId(row[0]),
            level=MemoryLevel(row[1]),
            type_id=row[2],
            created_generation=GenerationId(row[3]),
            updated_generation=GenerationId(row[4]),
            status_flags=row[5],
            support_count=row[6],
        )
        for row in raw["nodes"]
    }
    scores = {
        MemoryId(row[0]): MemoryScore(
            memory_id=MemoryId(row[0]),
            significance=row[1],
            prediction_error=row[2],
            learning_value=row[3],
            transfer_prior=row[4],
            explanatory_potential=row[5],
            future_option_delta=row[6],
        )
        for row in raw["scores"]
    }
    adjacency = {
        (MemoryId(source), int(relation)): tuple(MemoryId(value) for value in targets)
        for source, relation, targets in raw["adjacency"]
    }
    indexes = CognitionIndexes.freeze(
        contingency_by_context_action={
            (int(context), int(action)): tuple(MemoryId(value) for value in values)
            for context, action, values in raw["contingencies"]
        },
        role_by_context_action_family={
            (int(context), int(action), MemoryId(family)): tuple(MemoryId(value) for value in values)
            for context, action, family, values in raw["roles_exact"]
        },
        role_by_context_action={
            (int(context), int(action)): tuple(MemoryId(value) for value in values)
            for context, action, values in raw["roles_fallback"]
        },
        concepts_by_role={
            MemoryId(role_id): tuple(MemoryId(value) for value in values)
            for role_id, values in raw["concepts"]
        },
        action_aggregates={
            int(row[0]): ActionAggregate(
                future_option_sum=row[1],
                future_option_count=row[2],
                positive_count=row[3],
                negative_count=row[4],
                failure_count=row[5],
                contradiction_count=row[6],
            )
            for row in raw["aggregates"]
        },
    )
    return MemoryReadView.freeze(
        generation_id=generation_id,
        nodes=nodes,
        scores=scores,
        adjacency=adjacency,
        cognition_indexes=indexes,
    )


class MmapReadViewTransport:
    """Cross-process immutable read-view transport backed by memory-mapped files.

    Publication writes a complete immutable generation file and atomically renames it
    into place. Attachment memory-maps that file, verifies its digest and reconstructs
    the immutable Python read view without any manager proxy or read RPC.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish(self, view: MemoryReadView) -> ReadViewHandle:
        payload = _encode_view(view)
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"generation-{int(view.generation_id)}-{digest[:16]}.rv"
        final_path = self.directory / filename
        if not final_path.exists():
            temporary = self.directory / f".{filename}.{os.getpid()}.tmp"
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        return ReadViewHandle(
            generation_id=view.generation_id,
            transport_key=f"{filename}:{digest}",
        )

    def attach(self, handle: ReadViewHandle) -> MemoryReadView:
        filename, separator, expected_digest = handle.transport_key.partition(":")
        if not separator or not filename or not expected_digest:
            raise ValueError("invalid mmap read-view handle")
        path = self.directory / filename
        if not path.exists():
            raise KeyError(handle.transport_key)
        with path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
                payload = mapped[:]
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_digest:
            raise ValueError("mmap read-view digest mismatch")
        view = _decode_view(payload)
        if view.generation_id != handle.generation_id:
            raise ValueError("read-view handle generation mismatch")
        return view

    def release(self, handle: ReadViewHandle) -> None:
        filename = handle.transport_key.partition(":")[0]
        if filename:
            (self.directory / filename).unlink(missing_ok=True)

    @property
    def retained_generations(self) -> tuple[int, ...]:
        generations: list[int] = []
        for path in self.directory.glob("generation-*.rv"):
            parts = path.name.split("-")
            if len(parts) >= 3:
                try:
                    generations.append(int(parts[1]))
                except ValueError:
                    continue
        return tuple(sorted(generations))
