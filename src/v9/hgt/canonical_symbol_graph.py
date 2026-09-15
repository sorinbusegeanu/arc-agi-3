from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _without_legacy_symbol_tuples(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("semantic_before", "semantic_action", "semantic_options", "semantic_after", "semantic_effects"):
        rows = result.get(key)
        if rows:
            result[key] = [row for row in rows if not (isinstance(row, (list, tuple)) and len(row) == 5 and int(row[0]) == 7)]
    return result


def _proxy(read_view: Any) -> Any:
    return SimpleNamespace(
        nodes=read_view.nodes,
        edges=read_view.edges,
        payloads={uid: _without_legacy_symbol_tuples(dict(payload)) for uid, payload in read_view.payloads.items()},
    )


def install(training_module: Any) -> None:
    if getattr(training_module, "_v978_symbol_graph_installed", False):
        return
    original_graph = training_module.build_hgt_graph
    original_loss = training_module._multitask_loss
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
        symbols: dict[tuple[int, int, int, int], int] = {}
        symbol_occurrences: list[tuple[tuple[int, int, int, int], str, int, int, int]] = []
        for uid in ordered:
            payload = sanitized.payloads.get(uid, {})
            node_type, node_index = memory_index[uid]
            if include_objectives and payload.get("symbol_identity") is not None:
                causal = int(payload.get("causal_watermark", payload.get("symbol_causal_watermark", sanitized.nodes[uid].created_watermark))) <= int(sanitized.nodes[uid].created_watermark)
                control = str(payload.get("cross_modal_control", "aligned"))
                if "shuffled_alignment_discrimination" in task_target_dict:
                    task_target_dict["shuffled_alignment_discrimination"][node_type][node_index] = 0.0 if control == "shuffled" else 1.0
                    task_mask_dict["shuffled_alignment_discrimination"][node_type][node_index] = bool(causal)
                confidence = float(payload.get("grounding_confidence", 1.0 if control == "aligned" else 0.0))
                if "grounding_confidence_calibration" in task_target_dict:
                    task_target_dict["grounding_confidence_calibration"][node_type][node_index] = max(0.0, min(1.0, confidence))
                    task_mask_dict["grounding_confidence_calibration"][node_type][node_index] = bool(causal)
            if payload.get("context_signature") is not None:
                context_key = (int(payload.get("environment_instance_id", 0)), int(payload["context_signature"]))
                context_index = contexts.get(context_key)
                if context_index is None:
                    context_index = len(contexts)
                    contexts[context_key] = context_index
                    values = [0.0] * input_dim
                    values[0] = float(context_key[0] & 0xFFFF) / 65535.0
                    values[1] = float(context_key[1] & 0xFFFF) / 65535.0
                    context_features.append(torch.tensor(values, dtype=torch.float32))
                context_edges.append((node_type, node_index, context_index))
            identity = payload.get("symbol_identity")
            if isinstance(identity, (list, tuple)) and len(identity) == 4:
                symbol_key = tuple(int(value) for value in identity)
                if symbol_key not in symbols:
                    symbols[symbol_key] = len(symbols)
                symbol_occurrences.append((symbol_key, node_type, node_index, int(payload.get("environment_instance_id", 0)), int(payload.get("episode_id", 0))))

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

        if "SYMBOL" in x_dict and symbol_occurrences:
            symbol_index = {key: index for index, key in enumerate(symbols)}
            by_window: dict[tuple[int, int], list[tuple[tuple[int, int, int, int], int]]] = {}
            for symbol_key, node_type, node_index, environment_id, episode_id in symbol_occurrences:
                index = symbol_index.get(symbol_key)
                if index is None or index >= int(x_dict["SYMBOL"].shape[0]):
                    continue
                key = ("SYMBOL", "OBSERVED_IN", node_type)
                edge_index_dict[key] = _append_edge(torch, edge_index_dict.get(key), index, node_index)
                by_window.setdefault((environment_id, episode_id), []).append((symbol_key, index))
            for rows in by_window.values():
                ordered_rows = sorted(rows, key=lambda item: int(item[0][3]))
                for position, (_symbol_key, symbol_idx) in enumerate(ordered_rows):
                    for _other_key, other_idx in ordered_rows[position + 1:]:
                        edge_index_dict[("SYMBOL", "CO_OCCURS", "SYMBOL")] = _append_edge(torch, edge_index_dict.get(("SYMBOL", "CO_OCCURS", "SYMBOL")), symbol_idx, other_idx)
                    if position + 1 < len(ordered_rows):
                        _, next_idx = ordered_rows[position + 1]
                        edge_index_dict[("SYMBOL", "PRECEDES", "SYMBOL")] = _append_edge(torch, edge_index_dict.get(("SYMBOL", "PRECEDES", "SYMBOL")), symbol_idx, next_idx)
                        edge_index_dict[("SYMBOL", "FOLLOWS", "SYMBOL")] = _append_edge(torch, edge_index_dict.get(("SYMBOL", "FOLLOWS", "SYMBOL")), next_idx, symbol_idx)

        training_module._v978_last_symbol_nodes = int(x_dict["SYMBOL"].shape[0]) if "SYMBOL" in x_dict else 0
        training_module._v978_last_symbol_edges = sum(int(value.shape[1]) for key, value in edge_index_dict.items() if "SYMBOL" in (key[0], key[2]))
        training_module._v978_last_context_nodes = int(x_dict["CONTEXT"].shape[0]) if "CONTEXT" in x_dict else 0
        if include_objectives:
            return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta, task_target_dict, task_mask_dict
        return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta

    def multitask_loss(*args: Any, **kwargs: Any):
        result = original_loss(*args, **kwargs)
        training_module._v978_last_grounding_losses = {name: value for name, value in result[2].items() if name in training_module.GROUNDING_OBJECTIVES}
        return result

    def train_hgt_epoch(runtime: Any, **kwargs: Any):
        result = original_train(runtime, **kwargs)
        losses = dict(getattr(training_module, "_v978_last_grounding_losses", {}))
        if losses:
            for objective, value in losses.items():
                runtime.set_telemetry_gauge(f"hgt_grounding_{objective}_loss", float(value))
            runtime.set_telemetry_gauge("hgt_grounding_loss", sum(losses.values()) / len(losses))
            calibration = float(losses.get("grounding_confidence_calibration", 1.0))
            runtime.set_telemetry_gauge("hgt_grounding_calibration", max(0.0, 1.0 - calibration))
        runtime.set_telemetry_gauge("hgt_symbol_nodes", int(getattr(training_module, "_v978_last_symbol_nodes", 0)))
        runtime.set_telemetry_gauge("hgt_symbol_edges", int(getattr(training_module, "_v978_last_symbol_edges", 0)))
        runtime.set_telemetry_gauge("hgt_context_nodes", int(getattr(training_module, "_v978_last_context_nodes", 0)))
        return result

    training_module.build_hgt_graph = build_hgt_graph
    training_module._multitask_loss = multitask_loss
    training_module.train_hgt_epoch = train_hgt_epoch
    training_module._v978_symbol_graph_installed = True


def _append_edge(torch: Any, existing: Any, source: int, target: int):
    new_edge = torch.tensor([[int(source)], [int(target)]], dtype=torch.long)
    if existing is None:
        return new_edge
    return torch.cat((existing, new_edge), dim=1)
