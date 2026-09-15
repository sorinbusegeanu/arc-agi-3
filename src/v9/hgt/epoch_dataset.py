from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterator


class EpochTransitionDataset:
    """Append-only, epoch-scoped HGT training evidence.

    This dataset is independent of Hydra retention. Every sampled transition is
    written exactly once and can be streamed back without keeping the epoch in RAM.
    """

    def __init__(self, path: str | Path, *, epoch: int, branch: str, model_version: str | None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.count = 0
        self.bytes_written = 0
        self._handle = self.path.open("w", encoding="utf-8")
        self._meta = {
            "schema_version": 1,
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


def transition_training_rows(path: str | Path) -> list[dict[str, Any]]:
    """Materialize normalized action supervision from every sampled transition."""
    rows: list[dict[str, Any]] = []
    episode_returns: dict[tuple[str, int, int], float] = {}
    raw = list(iter_epoch_transitions(path))
    for row in reversed(raw):
        env = tuple(row.get("environment_identity") or ())
        game = str(row.get("game_scenario", ""))
        actor = int(row.get("actor_id", 0))
        episode = int(row.get("episode_id", 0))
        key = (game, actor, episode)
        immediate = float(int(row.get("primary_valence", 0)))
        immediate += 1.0 if bool(row.get("task_success", False)) else 0.0
        immediate -= 1.0 if bool(row.get("task_failure", False)) else 0.0
        immediate -= 0.10 if bool(row.get("task_truncated", False)) else 0.0
        running = max(-1.0, min(1.0, immediate + 0.97 * episode_returns.get(key, 0.0)))
        episode_returns[key] = running
        rows.append({
            "environment_identity": env,
            "game_scenario": game,
            "actor_id": actor,
            "episode_id": episode,
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
    rows.reverse()
    return rows


def action_ranking_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pairs actions observed in the same game/context when their returns differ."""
    grouped: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault((str(row["game_scenario"]), int(row["context_signature"])), {}).setdefault(int(row["action_id"]), []).append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for actions in grouped.values():
        means = {action: sum(float(r["target_return"]) for r in rs) / len(rs) for action, rs in actions.items()}
        ordered = sorted(means, key=means.get, reverse=True)
        if len(ordered) < 2 or means[ordered[0]] <= means[ordered[-1]]:
            continue
        pairs.append((actions[ordered[0]][0], actions[ordered[-1]][0]))
    return pairs
