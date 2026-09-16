from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYMBOL_FEATURE_SCHEMA_VERSION = 1
SYMBOL_FEATURE_NAMES = (
    "vocabulary_id",
    "symbol_id",
    "occurrence_count",
    "mean_position",
    "min_position",
    "max_position",
    "mean_micro_step",
    "temporal_span",
    "context_diversity",
    "before_action_ratio",
    "between_actions_ratio",
    "after_action_ratio",
    "after_outcome_ratio",
    "support",
    "contradiction",
    "grounding_confidence",
    "grounding_maturity",
)


def _phase_bucket(phase: str) -> int:
    return {
        "BEFORE_ACTION": 0,
        "BETWEEN_ACTIONS": 1,
        "AFTER_ACTION": 2,
        "AFTER_OUTCOME": 3,
        "COINCIDENT": 4,
    }.get(str(phase), 4)


def _patch_checkpoint_metadata(training_module: Any, result: Any, root: str | Path) -> None:
    checkpoint = getattr(result, "checkpoint", None)
    if not checkpoint:
        return
    torch, _, _ = training_module._require_torch()
    root = Path(root)
    path = root / str(checkpoint)
    if not path.exists():
        return
    data = torch.load(path, map_location="cpu")
    data["symbol_feature_schema_version"] = SYMBOL_FEATURE_SCHEMA_VERSION
    data["symbol_feature_names"] = list(SYMBOL_FEATURE_NAMES)
    torch.save(data, path)
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["symbol_feature_schema_version"] = SYMBOL_FEATURE_SCHEMA_VERSION
        payload["symbol_feature_names"] = list(SYMBOL_FEATURE_NAMES)
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "models" / "hgt_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["symbol_feature_schema_version"] = SYMBOL_FEATURE_SCHEMA_VERSION
        manifest["symbol_feature_names"] = list(SYMBOL_FEATURE_NAMES)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def install(training_module: Any) -> None:
    if getattr(training_module, "_v978_symbol_features_installed", False):
        return
    original_graph = training_module.build_hgt_graph
    original_train = training_module.train_hgt_epoch

    def build_hgt_graph(read_view: Any, **kwargs: Any):
        result = original_graph(read_view, **kwargs)
        x_dict = result[0]
        if "SYMBOL" not in x_dict:
            return result
        torch, _, _ = training_module._require_torch()
        input_dim = int(kwargs.get("input_dim", 64))
        max_nodes = int(kwargs.get("max_nodes", 800))
        selected = training_module._select_connected_nodes(read_view, max_nodes)
        ordered = sorted(selected, key=lambda uid: (int(read_view.nodes[uid].created_watermark), uid))

        order: list[tuple[int, int]] = []
        stats: dict[tuple[int, int], dict[str, Any]] = {}
        for uid in ordered:
            payload = dict(read_view.payloads.get(uid, {}))
            identity = payload.get("symbol_identity")
            if not (isinstance(identity, (list, tuple)) and len(identity) == 4):
                continue
            key = (int(identity[0]), int(identity[2]))
            row = stats.get(key)
            if row is None:
                row = {
                    "count": 0,
                    "positions": [],
                    "micro": [],
                    "watermarks": [],
                    "contexts": set(),
                    "phases": [0, 0, 0, 0, 0],
                    "support": 0.0,
                    "contradiction": 0.0,
                    "confidence": 0.0,
                    "maturity": 0,
                }
                stats[key] = row
                order.append(key)
            row["count"] += 1
            row["positions"].append(int(payload.get("symbol_position", identity[3])))
            row["micro"].append(int(payload.get("symbol_micro_step", identity[3])))
            row["watermarks"].append(int(payload.get("symbol_causal_watermark", payload.get("causal_watermark", read_view.nodes[uid].created_watermark))))
            context = payload.get("nearby_context_signature", payload.get("context_signature"))
            if context is not None:
                row["contexts"].add(int(context))
            row["phases"][_phase_bucket(str(payload.get("symbol_temporal_phase", "COINCIDENT")))] += 1
            row["support"] += float(payload.get("support", 0.0))
            row["contradiction"] += float(payload.get("contradiction", 0.0))
            row["confidence"] = max(float(row["confidence"]), float(payload.get("grounding_confidence", 0.0)))
            row["maturity"] = max(int(row["maturity"]), int(payload.get("grounding_maturity", 0)))

        if len(order) != int(x_dict["SYMBOL"].shape[0]):
            return result

        features = []
        for vocabulary, symbol in order:
            row = stats[(vocabulary, symbol)]
            count = max(1, int(row["count"]))
            positions = row["positions"] or [0]
            micro = row["micro"] or [0]
            watermarks = row["watermarks"] or [0]
            values = [0.0] * input_dim
            declared = (
                float(vocabulary & 0xFFFF) / 65535.0,
                float(symbol & 0xFFFF) / 65535.0,
                min(1.0, count / 32.0),
                min(1.0, (sum(positions) / len(positions)) / 64.0),
                min(1.0, min(positions) / 64.0),
                min(1.0, max(positions) / 64.0),
                min(1.0, (sum(micro) / len(micro)) / 64.0),
                min(1.0, (max(watermarks) - min(watermarks)) / 256.0),
                min(1.0, len(row["contexts"]) / 16.0),
                float(row["phases"][0]) / count,
                float(row["phases"][1]) / count,
                float(row["phases"][2]) / count,
                float(row["phases"][3]) / count,
                min(1.0, float(row["support"]) / max(1.0, count)),
                min(1.0, float(row["contradiction"]) / max(1.0, count)),
                max(0.0, min(1.0, float(row["confidence"]))),
                max(0.0, min(1.0, float(row["maturity"]) / 5.0)),
            )
            for index, value in enumerate(declared[:input_dim]):
                values[index] = float(value)
            features.append(torch.tensor(values, dtype=torch.float32))
        x_dict["SYMBOL"] = torch.stack(features, dim=0)
        training_module._v978_symbol_feature_schema_version = SYMBOL_FEATURE_SCHEMA_VERSION
        return result

    def train_hgt_epoch(runtime: Any, **kwargs: Any):
        result = original_train(runtime, **kwargs)
        root = kwargs.get("root")
        if root is not None:
            _patch_checkpoint_metadata(training_module, result, root)
        return result

    training_module.build_hgt_graph = build_hgt_graph
    training_module.train_hgt_epoch = train_hgt_epoch
    training_module._v978_symbol_feature_schema_version = SYMBOL_FEATURE_SCHEMA_VERSION
    training_module._v978_symbol_features_installed = True
