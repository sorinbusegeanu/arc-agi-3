from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v9.memory.model import MemoryLevel
from v9.telemetry import HGTTrainingSample, ModelEvolutionSample, read_gpu_snapshot


@dataclass(frozen=True, slots=True)
class HGTTrainingResult:
    epoch: int
    status: str
    model_version: str
    parent_model_version: str | None
    training_loss: float
    validation_loss: float
    examples: int
    training_steps: int
    checkpoint: str | None


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import HGTConv
    except ImportError as exc:
        raise RuntimeError("HGT training requires torch and torch-geometric") from exc
    return torch, nn, HGTConv


def _node_feature(node: Any, payload: dict[str, Any], dim: int, torch: Any):
    values = [
        float(int(node.level)) / 7.0,
        float(int(node.memory_type)) / 32.0,
        float(node.created_watermark % 100000) / 100000.0,
        float(len(node.structural_key)) / 16.0,
        float(payload.get("recurrence", 0)) / 64.0,
        float(payload.get("compression_benefit", 0.0)) / 64.0,
        float(payload.get("explanatory_reach", 0)) / 64.0,
        float(bool(payload.get("validated", False))),
        float(payload.get("primary_valence", 0)),
        float(payload.get("future_option_delta", 0.0)) / 16.0,
    ]
    seed = hashlib.blake2b(
        json.dumps(
            {
                "key": list(node.structural_key),
                "payload_keys": sorted(payload.keys()),
            },
            sort_keys=True,
        ).encode("utf-8"),
        digest_size=32,
        person=b"v9-hgt-feature",
    ).digest()
    while len(values) < dim:
        byte = seed[(len(values) - 10) % len(seed)]
        values.append((float(byte) / 127.5) - 1.0)
    return torch.tensor(values[:dim], dtype=torch.float32)


def build_hgt_graph(read_view: Any, *, input_dim: int = 64, max_nodes: int = 800):
    torch, _, _ = _require_torch()
    nodes_by_type: dict[str, list[Any]] = {}
    ordered_nodes = sorted(read_view.nodes.items(), key=lambda row: (row[1].created_watermark, row[0]), reverse=True)
    selected = ordered_nodes[: max(1, int(max_nodes))]
    for uid, node in selected:
        node_type = f"M{int(node.level)}"
        nodes_by_type.setdefault(node_type, []).append(uid)
    for rows in nodes_by_type.values():
        rows.sort()

    index_by_uid: dict[Any, tuple[str, int]] = {}
    x_dict: dict[str, Any] = {}
    y_dict: dict[str, Any] = {}
    for node_type, uids in nodes_by_type.items():
        features = []
        labels = []
        for index, uid in enumerate(uids):
            index_by_uid[uid] = (node_type, index)
            node = read_view.nodes[uid]
            payload = dict(read_view.payloads.get(uid, {}))
            features.append(_node_feature(node, payload, input_dim, torch))
            labels.append(int(node.memory_type) // 100)
        x_dict[node_type] = torch.stack(features, dim=0)
        y_dict[node_type] = torch.tensor(labels, dtype=torch.long)

    edges: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for edge in read_view.edges.values():
        source = index_by_uid.get(edge.source)
        target = index_by_uid.get(edge.target)
        if source is None or target is None:
            continue
        edge_type = (source[0], str(edge.relation.value), target[0])
        edges.setdefault(edge_type, []).append((source[1], target[1]))

    edge_index_dict: dict[tuple[str, str, str], Any] = {}
    for key, pairs in edges.items():
        if pairs:
            edge_index_dict[key] = torch.tensor(pairs, dtype=torch.long).t().contiguous()

    return x_dict, edge_index_dict, y_dict


class _HGTWrapper:
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], *, input_dim: int, hidden_dim: int, layers: int, heads: int):
        torch, nn, HGTConv = _require_torch()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = nn.ModuleDict({
                    node_type: nn.Linear(input_dim, hidden_dim)
                    for node_type in metadata[0]
                })
                self.layers = nn.ModuleList([
                    HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
                    for _ in range(layers)
                ])
                self.heads = nn.ModuleDict({
                    node_type: nn.Linear(hidden_dim, 8)
                    for node_type in metadata[0]
                })

            def forward(self, x_dict, edge_index_dict):
                state = {
                    key: self.encoders[key](value).relu()
                    for key, value in x_dict.items()
                }
                for layer in self.layers:
                    updated = layer(state, edge_index_dict)
                    state = {
                        key: (state[key] if updated.get(key) is None else updated[key]).relu()
                        for key in state
                    }
                return {
                    key: self.heads[key](value)
                    for key, value in state.items()
                }

        self.model = Model()


def _split_masks(y_dict: dict[str, Any], torch: Any):
    train_masks = {}
    val_masks = {}
    for node_type, labels in y_dict.items():
        count = int(labels.numel())
        index = torch.arange(count)
        val_count = max(1, int(math.ceil(count * 0.2))) if count >= 5 else 0
        train_count = count - val_count
        train_masks[node_type] = index < train_count
        val_masks[node_type] = index >= train_count if val_count else torch.zeros(count, dtype=torch.bool)
    return train_masks, val_masks


def _loss(logits_dict, y_dict, masks, torch):
    losses = []
    correct = total = 0
    for node_type, logits in logits_dict.items():
        mask = masks[node_type].to(logits.device)
        if not bool(mask.any()):
            continue
        target = y_dict[node_type].to(logits.device)[mask]
        selected = logits[mask]
        loss = torch.nn.functional.cross_entropy(selected, target)
        losses.append(loss)
        correct += int((selected.argmax(dim=-1) == target).sum().item())
        total += int(target.numel())
    if not losses:
        return None, 0.0
    return torch.stack(losses).mean(), correct / max(1, total)


def train_hgt_epoch(
    runtime: Any,
    *,
    epoch: int,
    training_epochs: int,
    learning_rate: float,
    root: str | Path,
) -> HGTTrainingResult:
    try:
        torch, _, _ = _require_torch()
    except RuntimeError:
        return HGTTrainingResult(
            epoch,
            "SKIPPED_DEPENDENCY",
            runtime.unified_telemetry.model_version,
            None,
            0.0,
            0.0,
            len(runtime.read_view.nodes),
            0,
            None,
        )
    config = runtime.config.scientific
    read_view = runtime.read_view
    if len(read_view.nodes) < 8:
        return HGTTrainingResult(epoch, "SKIPPED_INSUFFICIENT_DATA", runtime.unified_telemetry.model_version, None, 0.0, 0.0, len(read_view.nodes), 0, None)

    x_dict, edge_index_dict, y_dict = build_hgt_graph(read_view, max_nodes=int(config.hgt_max_subgraph_nodes))
    metadata = (sorted(x_dict), sorted(edge_index_dict))
    if not metadata[1]:
        return HGTTrainingResult(epoch, "SKIPPED_NO_RELATIONS", runtime.unified_telemetry.model_version, None, 0.0, 0.0, len(read_view.nodes), 0, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = _HGTWrapper(
        metadata,
        input_dim=64,
        hidden_dim=int(config.hgt_hidden_dim),
        layers=int(config.hgt_layers),
        heads=int(config.hgt_heads),
    )
    model = wrapper.model.to(device)

    model_dir = Path(root) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "hgt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    parent_version = manifest.get("current_model_version")
    parent_checkpoint = manifest.get("current_checkpoint")

    if parent_checkpoint:
        checkpoint_path = Path(root) / str(parent_checkpoint)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device)
            try:
                model.load_state_dict(state["model_state"])
            except RuntimeError:
                parent_version = None
                parent_checkpoint = None

    x_device = {key: value.to(device) for key, value in x_dict.items()}
    edges_device = {key: value.to(device) for key, value in edge_index_dict.items()}
    train_masks, val_masks = _split_masks(y_dict, torch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    start = time.perf_counter()
    training_loss = 0.0
    training_steps = 0
    last_grad_norm = 0.0
    model.train()
    for _ in range(max(1, int(training_epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_device, edges_device)
        loss, _ = _loss(logits, y_dict, train_masks, torch)
        if loss is None:
            break
        loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item())
        optimizer.step()
        training_loss = float(loss.detach().cpu().item())
        training_steps += 1

    model.eval()
    with torch.no_grad():
        logits = model(x_device, edges_device)
        val_loss_t, val_accuracy = _loss(logits, y_dict, val_masks, torch)
        train_loss_t, train_accuracy = _loss(logits, y_dict, train_masks, torch)
    validation_loss = float(val_loss_t.cpu().item()) if val_loss_t is not None else training_loss
    training_loss = float(train_loss_t.cpu().item()) if train_loss_t is not None else training_loss
    elapsed = max(1e-9, time.perf_counter() - start)

    version_index = int(manifest.get("version_index", 0)) + 1
    candidate_version = f"hgt-{version_index:06d}"
    checkpoint_rel = f"models/{candidate_version}.pt"
    checkpoint_path = Path(root) / checkpoint_rel

    previous_val = float(manifest.get("validation_loss", float("inf")))
    promote = parent_version is None or validation_loss <= previous_val * 1.02
    status = "PROMOTED" if promote else "REJECTED"
    if promote:
        torch.save(
            {
                "model_state": model.state_dict(),
                "metadata": metadata,
                "input_dim": 64,
                "hidden_dim": int(config.hgt_hidden_dim),
                "layers": int(config.hgt_layers),
                "heads": int(config.hgt_heads),
                "validation_loss": validation_loss,
                "training_loss": training_loss,
            },
            checkpoint_path,
        )
        manifest = {
            "version_index": version_index,
            "current_model_version": candidate_version,
            "current_checkpoint": checkpoint_rel,
            "parent_model_version": parent_version,
            "validation_loss": validation_loss,
            "training_loss": training_loss,
            "graph_generation": int(read_view.generation),
            "examples": sum(int(v.numel()) for v in y_dict.values()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        model_version = candidate_version
    else:
        model_version = str(parent_version)

    gpu = read_gpu_snapshot()
    examples = sum(int(v.numel()) for v in y_dict.values())
    sample = HGTTrainingSample(
        training_loss=training_loss,
        validation_loss=validation_loss,
        training_step_latency_ms=1000.0 * elapsed / max(1, training_steps),
        training_examples_seen=examples,
        effective_batch_size=examples,
        gradient_norm=last_grad_norm,
        learning_rate=float(learning_rate),
        training_steps=training_steps,
        examples_per_second=examples / elapsed,
        gpu_memory_bytes=int(gpu.memory_used_bytes),
        gpu_utilization=float(gpu.utilization_percent),
        historical_retention=train_accuracy,
        current_curriculum_gain=max(0.0, val_accuracy - 0.125),
        cross_family_validation_gain=max(0.0, val_accuracy - train_accuracy),
        loss_by_head={"memory_level": validation_loss},
    )
    runtime.record_hgt_training(sample)
    runtime.record_model_evolution(
        ModelEvolutionSample(
            model_version=model_version,
            parent_model_version=parent_version,
            training_examples_since_parent=examples,
            current_stage_delta=max(0.0, val_accuracy - 0.125),
            historical_retention_delta=train_accuracy - 1.0,
            cross_family_transfer_delta=val_accuracy - train_accuracy,
            reasoning_improvement_delta=0.0,
            inference_latency_delta_ms=0.0,
            promotion_result=status,
        )
    )

    return HGTTrainingResult(
        epoch=epoch,
        status=status,
        model_version=model_version,
        parent_model_version=parent_version,
        training_loss=training_loss,
        validation_loss=validation_loss,
        examples=examples,
        training_steps=training_steps,
        checkpoint=checkpoint_rel if promote else parent_checkpoint,
    )
