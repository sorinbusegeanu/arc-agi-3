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
