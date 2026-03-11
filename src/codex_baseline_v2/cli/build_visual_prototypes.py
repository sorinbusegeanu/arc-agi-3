from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from codex_baseline_v2.memory.store import load_blackboard
from codex_baseline_v2.shared.config import load_config
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.vision.event_crop_encoder import encode_event_records
from codex_baseline_v2.vision.object_crop_encoder import encode_object_records
from codex_baseline_v2.vision.prototype_memory import build_prototype_memory, save_prototype_memory
from codex_baseline_v2.vision.reid_matcher import build_reid_links
from codex_baseline_v2.vision.training_pairs import build_training_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual prototype memory")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = load_config(json.load(handle))
    storage = StoragePathsV2(cfg.memory.storage_dir)
    blackboard = load_blackboard(storage, cfg.game_id)
    if blackboard is None:
        raise SystemExit("No blackboard state found.")
    game_root = Path(storage.game_root(cfg.game_id))
    vision_root = game_root / "vision"
    vision_root.mkdir(parents=True, exist_ok=True)

    object_embeddings = encode_object_records(blackboard, vision_root)
    event_embeddings = encode_event_records(blackboard, vision_root)
    training_pairs = build_training_pairs(blackboard, object_embeddings, event_embeddings)
    prototypes = build_prototype_memory(object_embeddings, event_embeddings, vision_root)
    reid_links = build_reid_links(blackboard, object_embeddings, event_embeddings, prototypes)
    save_prototype_memory(vision_root, object_embeddings, event_embeddings, prototypes, reid_links, training_pairs)


if __name__ == "__main__":
    main()
