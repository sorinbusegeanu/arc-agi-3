from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_TRANSITION_SAMPLE_ROWS = 8192
DEFAULT_ACTIVE_EPISODE_LIMIT = 128
DEFAULT_ACTION_RANKING_PAIRS = 4096


class EpochTransitionDataset:
    """Append-only, epoch-scoped HGT training evidence."""

    def __init__(self, path: str | Path, *, epoch: int, branch: str, model_version: str | None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.count = 0
        self.bytes_written = 0
        self._handle = self.path.open("w", encoding="utf-8")
        self._meta = {
            "schema_version": 2,
            "epoch": int(epoch),
            "branch": str(branch),
            "model_version": model_version,
        }

    def append(self, transition: Any) -> None:
        row = asdict(transition)
        encoded = json.dumps(row, separators=(",", ":"), sort_keys=True)
        self._handle.write(encoded + "\n")
        self.count += 1
        self.bytes_written += len(encoded.encode("utf-8")) + 1

    def append_batch(self, transitions: Iterable[Any]) -> int:
        encoded_rows = [
            json.dumps(asdict(transition), separators=(",", ":"), sort_keys=True)
            for transition in transitions
        ]
        if not encoded_rows:
            return 0
        self._handle.writelines(encoded + "\n" for encoded in encoded_rows)
        self.count += len(encoded_rows)
        self.bytes_written += sum(len(encoded.encode("utf-8")) + 1 for encoded in encoded_rows)
        return len(encoded_rows)

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        self._handle.close()
        meta = {**self._meta, "transitions": self.count, "bytes": self.bytes_written}
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def __enter__(self) -> "EpochTransitionDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_epoch_transitions(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dataset_path(root: str | Path, *, epoch: int, branch: str) -> Path:
    return Path(root) / "hgt" / "datasets" / f"epoch-{int(epoch):04d}" / f"{branch.lower()}.jsonl"


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("game_scenario", "")), int(row.get("actor_id", 0)), int(row.get("episode_id", 0))


def _terminal(row: dict[str, Any]) -> bool:
    return bool(
        row.get("task_success", False)
        or row.get("task_failure", False)
        or row.get("task_truncated", False)
        or row.get("done", False)
    )


def _episode_rows(raw_rows: list[dict[str, Any]], *, discount: float = 0.97) -> list[dict[str, Any]]:
    running = 0.0
    result: list[dict[str, Any]] = []
    for row in reversed(raw_rows):
        immediate = float(int(row.get("primary_valence", 0)))
        immediate += 1.0 if bool(row.get("task_success", False)) else 0.0
        immediate -= 1.0 if bool(row.get("task_failure", False)) else 0.0
        immediate -= 0.10 if bool(row.get("task_truncated", False)) else 0.0
        running = max(-1.0, min(1.0, immediate + float(discount) * running))
        result.append({
            "environment_identity": tuple(row.get("environment_identity") or ()),
            "game_scenario": str(row.get("game_scenario", "")),
            "actor_id": int(row.get("actor_id", 0)),
            "episode_id": int(row.get("episode_id", 0)),
            "global_step": int(row.get("global_step", 0)),
            "context_signature": int(row.get("before_signature", 0)),
            "next_context_signature": int(row.get("after_signature", 0)),
            "action_id": int(row.get("action_id", 0)),
            "target_return": running,
            "primary_valence": int(row.get("primary_valence", 0)),
            "task_success": bool(row.get("task_success", False)),
            "task_failure": bool(row.get("task_failure", False)),
            "task_truncated": bool(row.get("task_truncated", False)),
        })
    result.reverse()
    return result


def iter_episode_training_rows(
    path: str | Path,
    *,
    active_episode_limit: int | None = None,
    discount: float = 0.97,
) -> Iterator[dict[str, Any]]:
    """Stream returns while retaining only bounded active episode buffers."""
    limit = max(1, int(active_episode_limit or DEFAULT_ACTIVE_EPISODE_LIMIT))
    active: OrderedDict[tuple[str, int, int], list[dict[str, Any]]] = OrderedDict()
    current_by_actor: dict[tuple[str, int], tuple[str, int, int]] = {}

    def flush(key: tuple[str, int, int]) -> list[dict[str, Any]]:
        raw_rows = active.pop(key, None)
        if not raw_rows:
            return []
        actor_key = (key[0], key[1])
        if current_by_actor.get(actor_key) == key:
            current_by_actor.pop(actor_key, None)
        return _episode_rows(raw_rows, discount=discount)

    for row in iter_epoch_transitions(path):
        key = _episode_key(row)
        actor_key = (key[0], key[1])
        previous = current_by_actor.get(actor_key)
        if previous is not None and previous != key and previous in active:
            yield from flush(previous)
        current_by_actor[actor_key] = key
        active.setdefault(key, []).append(row)
        active.move_to_end(key)
        if _terminal(row):
            yield from flush(key)
        while len(active) > limit:
            yield from flush(next(iter(active)))

    for key in tuple(active):
        yield from flush(key)


def iter_training_chunks(
    path: str | Path,
    *,
    chunk_rows: int | None = None,
    active_episode_limit: int | None = None,
) -> Iterator[tuple[dict[str, Any], ...]]:
    maximum = max(1, int(chunk_rows or DEFAULT_TRANSITION_SAMPLE_ROWS))
    chunk: list[dict[str, Any]] = []
    for row in iter_episode_training_rows(path, active_episode_limit=active_episode_limit):
        chunk.append(row)
        if len(chunk) >= maximum:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def _sample_priority(row: dict[str, Any]) -> int:
    raw = (
        f"{row.get('game_scenario','')}:{row.get('actor_id',0)}:{row.get('episode_id',0)}:"
        f"{row.get('global_step',0)}:{row.get('action_id',0)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8, person=b"v9-hgt-row").digest(), "big")


def transition_training_rows(
    path: str | Path,
    *,
    max_rows: int | None = None,
    active_episode_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic bounded sample without materializing the raw epoch."""
    maximum = max(1, int(max_rows or DEFAULT_TRANSITION_SAMPLE_ROWS))
    heap: list[tuple[int, int, dict[str, Any]]] = []
    serial = 0
    for row in iter_episode_training_rows(path, active_episode_limit=active_episode_limit):
        priority = _sample_priority(row)
        item = (-priority, -serial, row)
        if len(heap) < maximum:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)
        serial += 1
    rows = [item[2] for item in heap]
    rows.sort(key=lambda row: (
        str(row["game_scenario"]), int(row["actor_id"]), int(row["episode_id"]), int(row["global_step"])
    ))
    return rows


def iter_action_ranking_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_pairs: int = DEFAULT_ACTION_RANKING_PAIRS,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[int, tuple[float, int, dict[str, Any]]]] = {}
    context_limit = max(1024, int(max_pairs) * 4)
    for row in rows:
        context = str(row["game_scenario"]), int(row["context_signature"])
        action = int(row["action_id"])
        actions = grouped.setdefault(context, {})
        total, count, example = actions.get(action, (0.0, 0, row))
        actions[action] = total + float(row["target_return"]), count + 1, example
        if len(grouped) > context_limit:
            for key in sorted(grouped, reverse=True)[: len(grouped) - context_limit]:
                grouped.pop(key, None)
    emitted = 0
    for context in sorted(grouped):
        actions = grouped[context]
        means = {action: total / count for action, (total, count, _example) in actions.items()}
        if len(means) < 2:
            continue
        ordered = sorted(means, key=lambda action: (-means[action], action))
        best, worst = ordered[0], ordered[-1]
        if means[best] <= means[worst]:
            continue
        yield actions[best][2], actions[worst][2]
        emitted += 1
        if emitted >= max(1, int(max_pairs)):
            break


def action_ranking_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return list(iter_action_ranking_pairs(rows))
