from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .grounding_objectives import _causal_bounds, _targets


HGT_V978_MODEL_SCHEMA_VERSION = 7
SYMBOL_SCHEMA_VERSION = 3


def _without_legacy_symbol_tuples(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("semantic_before", "semantic_action", "semantic_options", "semantic_after", "semantic_effects"):
        rows = result.get(key)
        if rows:
            result[key] = [row for row in rows if not (isinstance(row, (list, tuple)) and len(row) == 5 and int(row[0]) == 7)]
    # Delegated legacy construction must never synthesize SYMBOL nodes from kind 7
    # or symbol_identity. Canonical SYMBOL nodes are added explicitly below.
    result.pop("symbol_identity", None)
    return result


def _proxy(read_view: Any) -> Any:
    return SimpleNamespace(
        nodes=read_view.nodes,
        edges=read_view.edges,
        payloads={uid: _without_legacy_symbol_tuples(dict(payload)) for uid, payload in read_view.payloads.items()},
    )


def _relation_family(payload: dict[str, Any]) -> tuple[str, ...]:
    relation = str(payload.get("symbol_relation", ""))
    result: list[str] = ["PROVENANCE"]
    if relation in {
        "SYMBOL_INTERACTION_ALIGNMENT",
        "SYMBOL_PRECEDES_ACTION",
        "SYMBOL_FOLLOWS_ACTION",
        "SYMBOL_PRECEDES_NORMALIZED_CHANGE",
        "SYMBOL_FOLLOWS_NORMALIZED_CHANGE",
    }:
        result.append("TEMPORALLY_ALIGNED_WITH")
    if relation in {
        "CROSS_MODAL_CORRESPONDENCE",
        "SYMBOL_TO_INTERACTION_PREDICTION",
        "INTERACTION_TO_SYMBOL_GENERALIZATION",
        "CROSS_MODAL_HELDOUT_TRANSFER",
        "CROSS_MODAL_COMPOSITION",
    }:
        result.append("STRUCTURALLY_CORRESPONDS_TO")
    if relation:
        result.append("PARTICIPATES_IN")
    support = float(payload.get("support", 0.0))
    contradiction = float(payload.get("contradiction", 0.0))
    if support > contradiction and support > 0.0:
        result.append("SUPPORTS")
    if contradiction > 0.0 or str(payload.get("cross_modal_control", "aligned")) == "shuffled":
        result.append("CONTRADICTS")
    if bool(payload.get("grounding_active", False)):
        result.append("GROUNDS")
    if bool(payload.get("heldout_transfer", False)):
        result.append("TRANSFER_VALIDATES")
    return tuple(dict.fromkeys(result))


def _symbol_feature(key: tuple[int, int], occurrences: int, dim: int, torch: Any):
    values = [0.0] * int(dim)
    vocabulary, symbol = key
    values[0] = float(vocabulary & 0xFFFF) / 65535.0
    values[1] = float(symbol & 0xFFFF) / 65535.0
    values[2] = min(1.0, float(occurrences) / 32.0)
    return torch.tensor(values, dtype=torch.float32)


def _grounding_targets(payload: dict[str, Any]) -> dict[str, tuple[float, bool]]:
    return _targets(payload)


def _checkpoint_metadata(training_module: Any) -> dict[str, Any]:
    relation_type = getattr(training_module, "RelationType", None)
    relation_vocabulary = [] if relation_type is None else [str(row.value) for row in relation_type]
    return {
        "model_schema_version": HGT_V978_MODEL_SCHEMA_VERSION,
        "symbol_schema_version": SYMBOL_SCHEMA_VERSION,
        "objective_names": list(training_module.OBJECTIVE_NAMES),
        "grounding_objectives": list(training_module.GROUNDING_OBJECTIVES),
        "relation_vocabulary": relation_vocabulary,
        "symbol_node_type": "SYMBOL",
    }


def _rewrite_checkpoint_metadata(training_module: Any, result: Any, root: str | Path) -> None:
    checkpoint_rel = getattr(result, "checkpoint", None)
    if not checkpoint_rel:
        return
    checkpoint_path = Path(root) / str(checkpoint_rel)
    if not checkpoint_path.exists():
        return
    torch, _, _ = training_module._require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.update(_checkpoint_metadata(training_module))
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)

    sidecar = checkpoint_path.with_suffix(".metadata.json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload.update(_checkpoint_metadata(training_module))
        temporary_sidecar = sidecar.with_suffix(".json.tmp")
        temporary_sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_sidecar, sidecar)

    manifest = Path(root) / "models" / "hgt_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.update(_checkpoint_metadata(training_module))
        temporary_manifest = manifest.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest)


def install(training_module: Any) -> None:
    if getattr(training_module, "_v978_symbol_graph_installed", False):
        return

    # Schema 7 explicitly invalidates pre-v9.7.8 objective/relation tensor layouts.
    # Same-schema relation-count changes remain migratable by edge identity in
    # training._migrate_model_state.
    training_module.MODEL_SCHEMA_VERSION = HGT_V978_MODEL_SCHEMA_VERSION
    original_graph = training_module.build_hgt_graph
    original_loss = training_module._loss
    original_train = training_module.train_hgt_epoch

    def build_hgt_graph(read_view: Any, **kwargs: Any):
        sanitized = _proxy(read_view)
        result = original_graph(sanitized, **kwargs)
        include_objectives = bool(kwargs.get("include_objectives", False))
        if include_objectives:
            x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta, task_target_dict, task_mask_dict = result
        else:
            x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta = result
        torch, _, _ = training_module._require_torch()
        max_nodes = int(kwargs.get("max_nodes", 800))
        input_dim = int(kwargs.get("input_dim", 64))
        selected = training_module._select_connected_nodes(sanitized, max_nodes)
        ordered = sorted(selected, key=lambda uid: (int(sanitized.nodes[uid].created_watermark), uid))
        by_type: dict[str, list[Any]] = {node_type: [] for node_type in training_module.MEMORY_NODE_TYPES}
        for uid in ordered:
            by_type[training_module._memory_node_type(sanitized.nodes[uid])].append(uid)
        memory_index = {uid: (node_type, index) for node_type, values in by_type.items() for index, uid in enumerate(values)}

        contexts: dict[tuple[int, int], int] = {}
        context_features: list[Any] = []
        context_edges: list[tuple[str, int, int]] = []
        symbols: dict[tuple[int, int], int] = {}
        symbol_counts: dict[tuple[int, int], int] = {}
        symbol_occurrences: list[tuple[tuple[int, int], str, int, int, int, int, int, dict[str, Any]]] = []
        dataset_counts = {name: [0, 0, 0] for name in training_module.GROUNDING_OBJECTIVES}

        for uid in ordered:
            original_payload = dict(read_view.payloads.get(uid, {}))
            node_type, node_index = memory_index[uid]
            if original_payload.get("context_signature") is not None:
                context_key = (int(original_payload.get("environment_instance_id", 0)), int(original_payload["context_signature"]))
                context_index = contexts.get(context_key)
                if context_index is None:
                    context_index = len(contexts)
                    contexts[context_key] = context_index
                    values = [0.0] * input_dim
                    values[0] = float(context_key[0] & 0xFFFF) / 65535.0
                    values[1] = float(context_key[1] & 0xFFFF) / 65535.0
                    context_features.append(torch.tensor(values, dtype=torch.float32))
                context_edges.append((node_type, node_index, context_index))

            identity = original_payload.get("symbol_identity")
            if isinstance(identity, (list, tuple)) and len(identity) == 4:
                symbol_key = (int(identity[0]), int(identity[2]))
                if symbol_key not in symbols:
                    symbols[symbol_key] = len(symbols)
                symbol_counts[symbol_key] = symbol_counts.get(symbol_key, 0) + 1
                symbol_occurrences.append((
                    symbol_key,
                    node_type,
                    node_index,
                    int(original_payload.get("environment_instance_id", 0)),
                    int(original_payload.get("episode_id", 0)),
                    int(original_payload.get("symbol_causal_watermark", original_payload.get("causal_watermark", sanitized.nodes[uid].created_watermark))),
                    int(original_payload.get("symbol_micro_step", identity[3])),
                    original_payload,
                ))
                if include_objectives:
                    causal, evaluation = _causal_bounds(original_payload, sanitized.nodes[uid])
                    for objective, (target, eligible) in _grounding_targets(original_payload).items():
                        if objective not in task_target_dict or node_type not in task_target_dict[objective]:
                            continue
                        task_target_dict[objective][node_type][node_index] = float(target)
                        mask = bool(causal <= evaluation and eligible)
                        task_mask_dict[objective][node_type][node_index] = mask
                        if mask:
                            dataset_counts[objective][0] += 1
                            dataset_counts[objective][1 if float(target) >= 0.5 else 2] += 1

        if context_features:
            x_dict["CONTEXT"] = torch.stack(context_features, dim=0)
            count = len(context_features)
            y_dict["CONTEXT"] = torch.zeros(count, dtype=torch.long)
            action_target_dict["CONTEXT"] = torch.zeros(count, dtype=torch.float32)
            action_mask_dict["CONTEXT"] = torch.zeros(count, dtype=torch.bool)
            action_meta["CONTEXT"] = [None] * count
            if include_objectives:
                for objective in training_module.AUX_OBJECTIVES:
                    task_target_dict[objective]["CONTEXT"] = torch.zeros(count, dtype=torch.float32)
                    task_mask_dict[objective]["CONTEXT"] = torch.zeros(count, dtype=torch.bool)
            for node_type, node_index, context_index in context_edges:
                key = (node_type, "OBSERVED_IN", "CONTEXT")
                edge_index_dict[key] = _append_edge(torch, edge_index_dict.get(key), node_index, context_index)

        if symbols:
            ordered_symbols = sorted(symbols, key=lambda key: symbols[key])
            x_dict["SYMBOL"] = torch.stack([_symbol_feature(key, symbol_counts.get(key, 1), input_dim, torch) for key in ordered_symbols], dim=0)
            count = len(ordered_symbols)
            y_dict["SYMBOL"] = torch.zeros(count, dtype=torch.long)
            action_target_dict["SYMBOL"] = torch.zeros(count, dtype=torch.float32)
            action_mask_dict["SYMBOL"] = torch.zeros(count, dtype=torch.bool)
            action_meta["SYMBOL"] = [None] * count
            if include_objectives:
                for objective in training_module.AUX_OBJECTIVES:
                    task_target_dict[objective]["SYMBOL"] = torch.zeros(count, dtype=torch.float32)
                    task_mask_dict[objective]["SYMBOL"] = torch.zeros(count, dtype=torch.bool)

            symbol_index = {key: index for index, key in enumerate(ordered_symbols)}
            by_window: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
            family_members: dict[int, list[tuple[int, str, int]]] = {}
            for symbol_key, node_type, node_index, environment_id, episode_id, watermark, micro_step, payload in symbol_occurrences:
                symbol_idx = symbol_index[symbol_key]
                observed_key = ("SYMBOL", "OBSERVED_IN", node_type)
                edge_index_dict[observed_key] = _append_edge(torch, edge_index_dict.get(observed_key), symbol_idx, node_index)
                for relation in _relation_family(payload):
                    relation_key = ("SYMBOL", relation, node_type)
                    edge_index_dict[relation_key] = _append_edge(torch, edge_index_dict.get(relation_key), symbol_idx, node_index)
                window = (environment_id, episode_id, int(payload.get("symbol_macro_step", watermark)))
                by_window.setdefault(window, []).append((watermark * 1_000_000 + micro_step, symbol_idx, node_index))
                family = payload.get("family_signature")
                if family is not None:
                    family_members.setdefault(int(family), []).append((symbol_idx, node_type, node_index))

            for rows in by_window.values():
                ordered_rows = sorted(rows)
                for position, (_order, symbol_idx, _node_index) in enumerate(ordered_rows):
                    for _other_order, other_idx, _other_node in ordered_rows[position + 1:]:
                        key = ("SYMBOL", "CO_OCCURS", "SYMBOL")
                        edge_index_dict[key] = _append_edge(torch, edge_index_dict.get(key), symbol_idx, other_idx)
                    if position + 1 < len(ordered_rows):
                        _, next_idx, _ = ordered_rows[position + 1]
                        precedes = ("SYMBOL", "PRECEDES", "SYMBOL")
                        follows = ("SYMBOL", "FOLLOWS", "SYMBOL")
                        edge_index_dict[precedes] = _append_edge(torch, edge_index_dict.get(precedes), symbol_idx, next_idx)
                        edge_index_dict[follows] = _append_edge(torch, edge_index_dict.get(follows), next_idx, symbol_idx)

            for family, members in family_members.items():
                for uid, (node_type, node_index) in memory_index.items():
                    payload = read_view.payloads.get(uid, {})
                    if int(payload.get("family_signature", payload.get("structural_signature", -1))) != family:
                        continue
                    if int(read_view.nodes[uid].level) < 2:
                        continue
                    for symbol_idx, _source_type, _source_index in members:
                        key = ("SYMBOL", "PARTICIPATES_IN", node_type)
                        edge_index_dict[key] = _append_edge(torch, edge_index_dict.get(key), symbol_idx, node_index)

            for edge in read_view.edges:
                relation = str(edge.relation.value)
                if relation not in {"GROUNDS", "TRANSFER_VALIDATES"}:
                    continue
                source_payload = read_view.payloads.get(edge.source, {})
                identity = source_payload.get("symbol_identity")
                if not (isinstance(identity, (list, tuple)) and len(identity) == 4):
                    continue
                symbol_idx = symbol_index.get((int(identity[0]), int(identity[2])))
                target = memory_index.get(edge.target)
                if symbol_idx is None or target is None:
                    continue
                target_type, target_index = target
                key = ("SYMBOL", relation, target_type)
                edge_index_dict[key] = _append_edge(torch, edge_index_dict.get(key), symbol_idx, target_index)

        training_module._v978_last_symbol_nodes = len(symbols)
        training_module._v978_last_symbol_edges = sum(int(value.shape[1]) for key, value in edge_index_dict.items() if "SYMBOL" in (key[0], key[2]))
        training_module._v978_last_context_nodes = len(contexts)
        training_module._v978_grounding_dataset_counts = dataset_counts
        if include_objectives:
            return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta, task_target_dict, task_mask_dict
        return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta

    def wrapped_loss(*args: Any, **kwargs: Any):
        result = original_loss(*args, **kwargs)
        training_module._v978_last_grounding_losses = {name: value for name, value in result[2].items() if name in training_module.GROUNDING_OBJECTIVES}
        metrics: dict[str, dict[str, float]] = {}
        if len(args) >= 10:
            auxiliary_dict, masks, task_targets, task_masks, torch = args[2], args[4], args[7], args[8], args[9]
            for objective in training_module.GROUNDING_OBJECTIVES:
                correct = total = 0
                absolute_error = 0.0
                for node_type, predictions in auxiliary_dict.get(objective, {}).items():
                    if node_type not in masks or node_type not in task_masks.get(objective, {}):
                        continue
                    mask = masks[node_type].to(predictions.device) & task_masks[objective][node_type].to(predictions.device)
                    if not bool(mask.any()):
                        continue
                    target = task_targets[objective][node_type].to(predictions.device)[mask]
                    probability = torch.sigmoid(predictions[mask])
                    correct += int(((probability >= 0.5) == (target >= 0.5)).sum().item())
                    total += int(target.numel())
                    absolute_error += float((probability - target).abs().sum().detach().cpu().item())
                if total:
                    metrics[objective] = {"accuracy": correct / total, "mae": absolute_error / total, "examples": float(total)}
        training_module._v978_last_grounding_metrics = metrics
        return result

    def train_hgt_epoch(runtime: Any, **kwargs: Any):
        result = original_train(runtime, **kwargs)
        _rewrite_checkpoint_metadata(training_module, result, kwargs.get("root", runtime.config.root))
        losses = dict(getattr(training_module, "_v978_last_grounding_losses", {}))
        metrics = dict(getattr(training_module, "_v978_last_grounding_metrics", {}))
        counts = dict(getattr(training_module, "_v978_grounding_dataset_counts", {}))
        if losses:
            for objective, value in losses.items():
                runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_loss", float(value))
            runtime.set_telemetry_gauge("hgt_grounding_loss", sum(losses.values()) / len(losses))
        for objective, row in metrics.items():
            runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_accuracy", float(row["accuracy"]))
            runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_mae", float(row["mae"]))
            runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_examples", float(row["examples"]))
        for objective, row in counts.items():
            runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_eligible", int(row[0]))
            runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_positive", int(row[1]))
            runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_negative", int(row[2]))
        calibration = metrics.get("grounding_confidence_calibration", {}).get("mae")
        if calibration is not None:
            runtime.set_telemetry_gauge("hgt_grounding_calibration", max(0.0, 1.0 - float(calibration)))
        runtime.set_telemetry_gauge("hgt_symbol_nodes", int(getattr(training_module, "_v978_last_symbol_nodes", 0)))
        runtime.set_telemetry_gauge("hgt_symbol_edges", int(getattr(training_module, "_v978_last_symbol_edges", 0)))
        runtime.set_telemetry_gauge("hgt_context_nodes", int(getattr(training_module, "_v978_last_context_nodes", 0)))
        runtime.set_telemetry_gauge("hgt_model_schema_version", HGT_V978_MODEL_SCHEMA_VERSION)
        runtime.set_telemetry_gauge("hgt_symbol_schema_version", SYMBOL_SCHEMA_VERSION)
        return result

    training_module.build_hgt_graph = build_hgt_graph
    training_module._loss = wrapped_loss
    training_module.train_hgt_epoch = train_hgt_epoch
    training_module._v978_symbol_graph_installed = True


def _append_edge(torch: Any, existing: Any, source: int, target: int):
    new_edge = torch.tensor([[int(source)], [int(target)]], dtype=torch.long)
    if existing is None:
        return new_edge
    return torch.cat((existing, new_edge), dim=1)
