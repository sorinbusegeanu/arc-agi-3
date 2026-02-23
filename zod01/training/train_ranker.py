from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from zod01.src.learned_models import ActionRankerModel, sigmoid


def load_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_by_game(rows: list[dict[str, object]], train_ratio: float = 0.8) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    games = sorted({str(r.get("game_id", "")) for r in rows})
    if not games:
        return rows, []
    cut = max(1, int(len(games) * train_ratio))
    train_games = set(games[:cut])
    train = [r for r in rows if str(r.get("game_id", "")) in train_games]
    valid = [r for r in rows if str(r.get("game_id", "")) not in train_games]
    return train, valid


def train_sgd(rows: list[dict[str, object]], epochs: int = 5, lr: float = 0.05) -> ActionRankerModel:
    model = ActionRankerModel(weights=[0.0] * 6, action_bias={})
    random.shuffle(rows)

    for _ in range(epochs):
        for r in rows:
            x = [float(v) for v in r.get("features", [0.0] * 6)]
            action = str(r.get("action", ""))
            y = float(int(r.get("label", 0)))
            z = sum(w * xv for w, xv in zip(model.weights, x)) + model.action_bias.get(action, 0.0)
            p = sigmoid(z)
            grad = p - y
            for i in range(len(model.weights)):
                model.weights[i] -= lr * grad * x[i]
            model.action_bias[action] = model.action_bias.get(action, 0.0) - lr * grad

    return model


def accuracy(model: ActionRankerModel, rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    ok = 0
    for r in rows:
        x = [float(v) for v in r.get("features", [0.0] * 6)]
        action = str(r.get("action", ""))
        y = int(r.get("label", 0))
        p = model.score(x, action)
        pred = 1 if p >= 0.5 else 0
        ok += int(pred == y)
    return ok / len(rows)


def rank_quality(model: ActionRankerModel, rows: list[dict[str, object]]) -> float:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r.get("episode_id", "")), int(r.get("step_idx", 0)))].append(r)
    if not grouped:
        return 0.0

    hits = 0
    total = 0
    for cand_rows in grouped.values():
        pos = [r for r in cand_rows if int(r.get("label", 0)) == 1]
        if not pos:
            continue
        total += 1
        top = max(cand_rows, key=lambda r: model.score([float(v) for v in r.get("features", [0.0] * 6)], str(r.get("action", ""))))
        if int(top.get("label", 0)) == 1:
            hits += 1
    return 0.0 if total == 0 else hits / total


def main() -> None:
    p = argparse.ArgumentParser(description="Train zod01 action ranker")
    p.add_argument("--dataset", type=str, default="zod01/datasets/ranker.jsonl")
    p.add_argument("--out", type=str, default="zod01/models/ranker.json")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--allow-no-positive", action="store_true")
    args = p.parse_args()

    rows = load_rows(args.dataset)
    positives = sum(int(r.get("label", 0)) for r in rows)
    if positives == 0 and not args.allow_no_positive:
        print(
            json.dumps(
                {
                    "error": "ranker dataset has zero positive labels (no successful episodes). collect better logs first.",
                    "rows_total": len(rows),
                },
                indent=2,
                sort_keys=True,
            )
        )
        sys.exit(2)

    train, valid = split_by_game(rows)
    model = train_sgd(train, epochs=args.epochs, lr=args.lr)
    model.save(args.out)

    metrics = {
        "rows_total": len(rows),
        "rows_train": len(train),
        "rows_valid": len(valid),
        "train_acc": round(accuracy(model, train), 6),
        "valid_acc": round(accuracy(model, valid), 6),
        "train_rank_quality": round(rank_quality(model, train), 6),
        "valid_rank_quality": round(rank_quality(model, valid), 6),
        "model_out": args.out,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
