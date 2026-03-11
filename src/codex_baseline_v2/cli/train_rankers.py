from __future__ import annotations

import argparse
import json
import os

from codex_baseline_v2.learning.ranking_dataset import build_ranking_dataset
from codex_baseline_v2.shared.config import load_config
from codex_baseline_v2.shared.storage import StoragePathsV2


def _write_weights(path: str, weights: dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": "v2.3.4", "weights": weights}, handle, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight ranking layers")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = load_config(json.load(handle))
    storage = StoragePathsV2(cfg.memory.storage_dir)
    samples = build_ranking_dataset(storage, cfg.game_id)
    game_root = storage.game_root(cfg.game_id)
    with open(os.path.join(game_root, "ranking_samples.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema_version": "v2.3.4", "ranking_samples": [sample.to_dict() for sample in samples]}, handle, sort_keys=True)
    mean_label = sum(sample.label_value for sample in samples) / float(max(1, len(samples)))
    _write_weights(os.path.join(game_root, "option_ranker_weights.json"), {"success_rate": 1.0 + mean_label, "duration_cost": 0.05, "effect_count": 0.2, "active_latent": 0.1})
    _write_weights(os.path.join(game_root, "mechanic_ranker_weights.json"), {"bias": mean_label, "length": 0.01})


if __name__ == "__main__":
    main()
