from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v9.memory.model import MemoryLevel
from v9.memory.relations import RelationType
from v9.telemetry import HGTTrainingSample, ModelEvolutionSample, read_gpu_snapshot

MODEL_SCHEMA_VERSION = 3
NODE_TYPE = "MEMORY"
SEMANTIC_NODE_TYPES = ("ENTITY", "ACTION", "TEXT", "STATE", "EFFECT")


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
    validation_accuracy: float = 0.0
    inference_latency_ms: float = 0.0
    subgraph_nodes: int = 0
    subgraph_edges: int = 0
    relevance_precision: float = 0.0


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import HGTConv
    except ImportError as exc:
        raise RuntimeError("HGT training requires torch and torch-geometric") from exc
    return torch, nn, HGTConv


def _stable_metadata() -> tuple[list[str], list[tuple[str, str, str]]]:
    node_types = [NODE_TYPE, *SEMANTIC_NODE_TYPES]
    edge_types = [(NODE_TYPE, relation.value, NODE_TYPE) for relation in RelationType]
    for semantic_type in SEMANTIC_NODE_TYPES:
        edge_types.append((NODE_TYPE, "SEMANTIC", semantic_type))
        edge_types.append((semantic_type, "SEMANTIC_OF", NODE_TYPE))
    return node_types, edge_types


def _semantic_rows(payload: dict[str, Any]) -> tuple[tuple[int, int, int, int, float], ...]:
    rows: list[tuple[int, int, int, int, float]] = []
    for key in ("semantic_before", "semantic_action", "semantic_options", "semantic_after", "semantic_effects"):
        for row in payload.get(key, ()) or ():
            if isinstance(row, (list, tuple)) and len(row) == 5:
                rows.append((int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])))
    return tuple(rows)


def _node_feature(node: Any, payload: dict[str, Any], dim: int, torch: Any):
    values = [
        float(int(node.level)) / 7.0,
        float(int(node.memory_type)) / 700.0,
        float(node.created_watermark % 100000) / 100000.0,
        float(len(node.structural_key)) / 16.0,
        float(payload.get("recurrence", 0)) / 64.0,
        float(payload.get("compression_benefit", 0.0)) / 64.0,
        float(payload.get("explanatory_reach", 0)) / 64.0,
        float(bool(payload.get("validated", False))),
        float(payload.get("support", 0)) / 64.0,
        float(len(payload.get("parents", ()))) / 8.0,
        float(bool(payload.get("task_success", False))),
        float(bool(payload.get("task_failure", False))),
        float(bool(payload.get("task_truncated", False))),
        min(1.0, float(payload.get("level_index", 0)) / 32.0),
        min(1.0, float(payload.get("levels_completed", 0)) / 32.0),
        max(-1.0, min(1.0, float(payload.get("primary_valence", 0)))),
    ]
    while len(values) < dim:
        values.append(0.0)
    for kind, subject, relation, obj, value in _semantic_rows(payload):
        if kind == 7:
            base, width = 40, 8
        elif kind == 8:
            base, width = 32, 8
        elif kind == 9:
            base, width = 48, 8
        elif kind in {2, 3, 4, 5}:
            base, width = 16, 16
        else:
            base, width = 10, 6
        digest = hashlib.blake2b(
            f"{kind}:{subject}:{relation}:{obj}".encode("ascii"),
            digest_size=2,
            person=b"v9-sem-slot",
        ).digest()
        slot = base + (int.from_bytes(digest, "little") % width)
        magnitude = float(value) if float(value) != 0.0 else 1.0
        values[slot] = max(-1.0, min(1.0, values[slot] + magnitude))
    return torch.tensor(values[:dim], dtype=torch.float32)


def _semantic_node_type(kind: int) -> str:
    if int(kind) == 8:
        return "ACTION"
    if int(kind) == 7:
        return "TEXT"
    if int(kind) == 9:
        return "EFFECT"
    if int(kind) in {5, 6}:
        return "STATE"
    return "ENTITY"


def _semantic_node_feature(fact: tuple[int, int, int, int, float], dim: int, torch: Any):
    kind, subject, relation, obj, value = fact
    values = [0.0] * int(dim)
    values[0] = float(kind) / 10.0
    values[1] = float(relation) / 32.0
    values[2] = max(-1.0, min(1.0, float(value)))
    values[3] = float(subject & 0xFFFF) / 65535.0
    values[4] = float(obj & 0xFFFF) / 65535.0
    return torch.tensor(values, dtype=torch.float32)


def _recent_behavior_nodes(read_view: Any, limit: int) -> list[tuple[Any, Any]]:
    candidates = []
    for uid, node in read_view.nodes.items():
        if node.level is not MemoryLevel.M0:
            continue
        payload = read_view.payloads.get(uid, {})
        if payload.get("action_id") is None or payload.get("environment_instance_id") is None:
            continue
        candidates.append((uid, node, abs(int(payload.get("primary_valence", 0)))))
    rows = heapq.nlargest(
        max(0, int(limit)),
        candidates,
        key=lambda row: (row[2], int(row[1].created_watermark), row[0]),
    )
    return [(uid, node) for uid, node, _ in rows]


def _select_connected_nodes(read_view: Any, max_nodes: int) -> tuple[Any, ...]:
    maximum = max(1, int(max_nodes))
    selected: dict[Any, Any] = {}
    for uid, node in _recent_behavior_nodes(read_view, maximum // 2):
        selected[uid] = node

    def edge_recency(edge: Any) -> tuple[int, int, int]:
        source = read_view.nodes.get(edge.source)
        target = read_view.nodes.get(edge.target)
        return (
            max(
                int(source.created_watermark) if source is not None else -1,
                int(target.created_watermark) if target is not None else -1,
            ),
            int(edge.source.hi ^ edge.target.hi),
            int(edge.source.lo ^ edge.target.lo),
        )

    for edge in heapq.nlargest(max(maximum, 64), read_view.edges, key=edge_recency):
        missing = [uid for uid in (edge.source, edge.target) if uid not in selected and uid in read_view.nodes]
        if len(selected) + len(missing) > maximum:
            continue
        for uid in missing:
            selected[uid] = read_view.nodes[uid]
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        remaining = ((uid, node) for uid, node in read_view.nodes.items() if uid not in selected)
        for uid, node in heapq.nlargest(
            maximum - len(selected),
            remaining,
            key=lambda row: (int(row[1].created_watermark), row[0]),
        ):
            selected[uid] = node
    return tuple(selected)


def _semantic_priority(fact: tuple[int, int, int, int, float]) -> tuple[int, int, int, int]:
    kind, subject, relation, obj, _ = fact
    priority = {9: 0, 8: 1, 7: 2, 5: 3, 6: 3, 2: 4, 3: 4, 4: 4}.get(int(kind), 5)
    return priority, int(relation), int(subject), int(obj)


def build_hgt_graph(
    read_view: Any,
    *,
    input_dim: int = 64,
    max_nodes: int = 800,
    max_edges: int = 4000,
    max_total_nodes: int = 6000,
    max_total_edges: int = 40000,
    max_semantic_facts_per_memory: int = 16,
    return_discount: float = 0.97,
):
    torch, _, _ = _require_torch()
    selected_uids = _select_connected_nodes(read_view, max_nodes)
    ordered_uids = sorted(selected_uids, key=lambda uid: (int(read_view.nodes[uid].created_watermark), uid))
    index_by_uid = {uid: index for index, uid in enumerate(ordered_uids)}
    features = []
    labels = []
    action_targets = [0.0] * len(ordered_uids)
    action_masks = []
    action_rows: list[tuple[int, int, int, int, int] | None] = []
    episode_rows: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for index, uid in enumerate(ordered_uids):
        node = read_view.nodes[uid]
        payload = dict(read_view.payloads.get(uid, {}))
        features.append(_node_feature(node, payload, input_dim, torch))
        valence = int(payload.get("primary_valence", 0)) if node.level is MemoryLevel.M0 else 0
        labels.append(max(0, min(2, valence + 1)))
        action_id = payload.get("action_id")
        environment_id = payload.get("environment_instance_id")
        context_signature = payload.get("context_signature")
        episode_id = payload.get("episode_id")
        usable_action = (
            node.level is MemoryLevel.M0
            and action_id is not None
            and environment_id is not None
            and context_signature is not None
            and episode_id is not None
        )
        action_masks.append(bool(usable_action))
        if usable_action:
            env, context, action, episode = int(environment_id), int(context_signature), int(action_id), int(episode_id)
            action_rows.append((env, context, action, episode, int(node.created_watermark)))
            episode_rows.setdefault((env, episode), []).append((index, int(node.created_watermark), valence))
        else:
            action_rows.append(None)
    gamma = float(return_discount)
    for rows in episode_rows.values():
        running = 0.0
        for index, _, valence in sorted(rows, key=lambda row: row[1], reverse=True):
            running = max(-1.0, min(1.0, float(valence) + gamma * running))
            action_targets[index] = running
    x_dict = {NODE_TYPE: torch.stack(features, dim=0)}
    y_dict = {NODE_TYPE: torch.tensor(labels, dtype=torch.long)}
    action_target_dict = {NODE_TYPE: torch.tensor(action_targets, dtype=torch.float32)}
    action_mask_dict = {NODE_TYPE: torch.tensor(action_masks, dtype=torch.bool)}
    action_meta = {NODE_TYPE: action_rows}
    eligible = (edge for edge in read_view.edges if edge.source in index_by_uid and edge.target in index_by_uid)
    selected_edges = heapq.nlargest(
        max(1, min(int(max_edges), int(max_total_edges))),
        eligible,
        key=lambda edge: max(
            int(read_view.nodes[edge.source].created_watermark),
            int(read_view.nodes[edge.target].created_watermark),
        ),
    )
    semantic_links = []
    semantic_tables = {}
    semantic_features = {}
    semantic_node_budget = max(0, int(max_total_nodes) - len(ordered_uids))
    semantic_link_budget = max(0, (int(max_total_edges) - len(selected_edges)) // 2)
    semantic_node_count = 0
    for memory_index, uid in enumerate(ordered_uids):
        if len(semantic_links) >= semantic_link_budget:
            break
        payload = dict(read_view.payloads.get(uid, {}))
        facts = sorted(set(_semantic_rows(payload)), key=_semantic_priority)
        for fact in facts[: max(1, int(max_semantic_facts_per_memory))]:
            if len(semantic_links) >= semantic_link_budget:
                break
            node_type = _semantic_node_type(int(fact[0]))
            table = semantic_tables.setdefault(node_type, {})
            semantic_index = table.get(fact)
            if semantic_index is None:
                if semantic_node_count >= semantic_node_budget:
                    continue
                semantic_index = len(table)
                table[fact] = semantic_index
                semantic_features.setdefault(node_type, []).append(_semantic_node_feature(fact, input_dim, torch))
                semantic_node_count += 1
            semantic_links.append((memory_index, node_type, semantic_index))
    for node_type, rows in semantic_features.items():
        x_dict[node_type] = torch.stack(rows, dim=0)
        count = len(rows)
        y_dict[node_type] = torch.zeros(count, dtype=torch.long)
        action_target_dict[node_type] = torch.zeros(count, dtype=torch.float32)
        action_mask_dict[node_type] = torch.zeros(count, dtype=torch.bool)
        action_meta[node_type] = [None] * count
    edges: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for edge in selected_edges:
        edges.setdefault((NODE_TYPE, str(edge.relation.value), NODE_TYPE), []).append(
            (index_by_uid[edge.source], index_by_uid[edge.target])
        )
    for memory_index, node_type, semantic_index in semantic_links:
        edges.setdefault((NODE_TYPE, "SEMANTIC", node_type), []).append((memory_index, semantic_index))
        edges.setdefault((node_type, "SEMANTIC_OF", NODE_TYPE), []).append((semantic_index, memory_index))
    edge_index_dict = {
        key: torch.tensor(pairs, dtype=torch.long).t().contiguous()
        for key, pairs in edges.items()
        if pairs
    }
    return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta


class _HGTWrapper:
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], *, input_dim: int, hidden_dim: int, layers: int, heads: int):
        _, nn, HGTConv = _require_torch()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = nn.ModuleDict({node_type: nn.Linear(input_dim, hidden_dim) for node_type in metadata[0]})
                self.layers = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata, heads=heads) for _ in range(layers)])
                self.valence_heads = nn.ModuleDict({node_type: nn.Linear(hidden_dim, 3) for node_type in metadata[0]})
                self.value_heads = nn.ModuleDict({node_type: nn.Linear(hidden_dim, 1) for node_type in metadata[0]})

            def forward(self, x_dict, edge_index_dict):
                state = {key: self.encoders[key](value).relu() for key, value in x_dict.items()}
                for layer in self.layers:
                    updated = layer(state, edge_index_dict)
                    state = {key: (state[key] if updated.get(key) is None else updated[key]).relu() for key in state}
                return (
                    {key: self.valence_heads[key](value) for key, value in state.items()},
                    {key: self.value_heads[key](value).squeeze(-1) for key, value in state.items()},
                )

        self.model = Model()


def _episode_rank(environment_id: int, episode_id: int) -> int:
    digest = hashlib.blake2b(
        f"{int(environment_id)}:{int(episode_id)}".encode("ascii"),
        digest_size=8,
        person=b"v9-hgt-split",
    ).digest()
    return int.from_bytes(digest, "little")


def _split_masks(
    action_meta: dict[str, list[Any]],
    action_masks: dict[str, Any],
    torch: Any,
    *,
    environment_families: dict[int, str] | None = None,
):
    """Prefer held-out environment families; fall back to held-out episodes."""
    train_masks: dict[str, Any] = {}
    val_masks: dict[str, Any] = {}
    families = environment_families or {}
    for node_type, action_mask in action_masks.items():
        rows = action_meta[node_type]
        count = int(action_mask.numel())
        present_families = sorted({
            families.get(int(row[0]), "")
            for row in rows
            if row is not None and families.get(int(row[0]), "")
        })
        validation_families: set[str] = set()
        if len(present_families) >= 2:
            family_count = min(len(present_families) - 1, max(1, int(math.ceil(len(present_families) * 0.2))))
            validation_families = set(
                sorted(
                    present_families,
                    key=lambda family: hashlib.blake2b(
                        family.encode("utf-8"),
                        digest_size=8,
                        person=b"v9-hgt-family",
                    ).digest(),
                )[:family_count]
            )

        train = torch.zeros(count, dtype=torch.bool)
        validation = torch.zeros(count, dtype=torch.bool)
        if validation_families:
            for index, row in enumerate(rows):
                if row is None:
                    continue
                family = families.get(int(row[0]), "")
                if family in validation_families:
                    validation[index] = True
                else:
                    train[index] = True
        else:
            episodes = sorted(
                {(int(row[0]), int(row[3])) for row in rows if row is not None},
                key=lambda key: (_episode_rank(*key), key),
            )
            if len(episodes) < 2:
                train_masks[node_type] = action_mask.clone()
                val_masks[node_type] = torch.zeros(count, dtype=torch.bool)
                continue
            val_count = min(len(episodes) - 1, max(1, int(math.ceil(len(episodes) * 0.2))))
            validation_episodes = set(episodes[:val_count])
            for index, row in enumerate(rows):
                if row is None:
                    continue
                key = (int(row[0]), int(row[3]))
                if key in validation_episodes:
                    validation[index] = True
                else:
                    train[index] = True
        train_masks[node_type] = train
        val_masks[node_type] = validation
    return train_masks, val_masks


def _masked_count(masks: dict[str, Any], action_masks: dict[str, Any]) -> int:
    return sum(int((masks[key] & action_masks[key]).sum().item()) for key in masks)


def _loss(logits_dict, value_dict, y_dict, masks, action_targets, action_masks, torch):
    classification_losses, value_losses = [], []
    correct = total = 0
    for node_type, logits in logits_dict.items():
        mask = masks[node_type].to(logits.device) & action_masks[node_type].to(logits.device)
        if not bool(mask.any()):
            continue
        target = y_dict[node_type].to(logits.device)[mask]
        selected = logits[mask]
        classification_losses.append(torch.nn.functional.cross_entropy(selected, target))
        correct += int((selected.argmax(dim=-1) == target).sum().item())
        total += int(target.numel())
    for node_type, values in value_dict.items():
        mask = masks[node_type].to(values.device) & action_masks[node_type].to(values.device)
        if bool(mask.any()):
            target = action_targets[node_type].to(values.device)[mask]
            value_losses.append(torch.nn.functional.smooth_l1_loss(values[mask], target))
    if not classification_losses and not value_losses:
        return None, 0.0
    anchor = next(iter(value_dict.values()))
    classification = torch.stack(classification_losses).mean() if classification_losses else anchor.new_tensor(0.0)
    value_loss = torch.stack(value_losses).mean() if value_losses else anchor.new_tensor(0.0)
    return 0.25 * classification + value_loss, correct / max(1, total)


def _should_promote(
    parent_version: str | None,
    parent_validation_loss: float,
    candidate_validation_loss: float,
    parent_validation_accuracy: float,
    candidate_validation_accuracy: float,
) -> bool:
    if not math.isfinite(candidate_validation_loss):
        return False
    if parent_version is None or not math.isfinite(parent_validation_loss):
        return True
    loss_ok = float(candidate_validation_loss) <= float(parent_validation_loss) * 1.01
    accuracy_ok = not math.isfinite(parent_validation_accuracy) or float(candidate_validation_accuracy) >= float(parent_validation_accuracy) - 0.02
    return bool(loss_ok and accuracy_ok)


def train_hgt_epoch(runtime: Any, *, epoch: int, training_epochs: int, learning_rate: float, root: str | Path, _budget_scale: float = 1.0, _oom_retry: int = 0) -> HGTTrainingResult:
    try:
        torch, _, _ = _require_torch()
    except RuntimeError:
        examples = int(getattr(getattr(runtime, "graph", None), "memory_count", lambda: 0)())
        return HGTTrainingResult(epoch, "SKIPPED_DEPENDENCY", runtime.unified_telemetry.model_version, None, 0.0, 0.0, examples, 0, None)
    config = runtime.config.scientific
    memory_node_budget = max(64, int(config.hgt_max_subgraph_nodes * float(_budget_scale)))
    canonical_edge_budget = max(256, int(config.hgt_max_subgraph_edges * float(_budget_scale)))
    total_node_budget = max(memory_node_budget, int(config.hgt_max_total_nodes * float(_budget_scale)))
    total_edge_budget = max(canonical_edge_budget, int(config.hgt_max_total_edges * float(_budget_scale)))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        runtime.set_telemetry_gauge("hgt_vram_free_before_training_bytes", int(free_bytes))
        runtime.set_telemetry_gauge("hgt_vram_total_bytes", int(total_bytes))
        if int(free_bytes) < int(config.hgt_min_free_vram_bytes):
            pressure = max(0.25, float(free_bytes) / max(1.0, float(config.hgt_min_free_vram_bytes)))
            memory_node_budget = max(64, int(memory_node_budget * pressure))
            canonical_edge_budget = max(256, int(canonical_edge_budget * pressure))
            total_node_budget = max(memory_node_budget, int(total_node_budget * pressure))
            total_edge_budget = max(canonical_edge_budget, int(total_edge_budget * pressure))
            runtime.set_telemetry_gauge("hgt_vram_preflight_shedding_factor", float(pressure))
    training_view_builder = getattr(runtime.graph, "training_view", None)
    read_view = (
        training_view_builder(
            max_nodes=memory_node_budget,
            max_edges=canonical_edge_budget,
        )
        if callable(training_view_builder)
        else runtime.read_view
    )
    if len(read_view.nodes) < 8:
        return HGTTrainingResult(epoch, "SKIPPED_INSUFFICIENT_DATA", runtime.unified_telemetry.model_version, None, 0.0, 0.0, len(read_view.nodes), 0, None)
    x_dict, edge_index_dict, y_dict, action_targets, action_masks, action_meta = build_hgt_graph(
        read_view,
        max_nodes=memory_node_budget,
        max_edges=canonical_edge_budget,
        max_total_nodes=total_node_budget,
        max_total_edges=total_edge_budget,
        max_semantic_facts_per_memory=int(config.hgt_max_semantic_facts_per_memory),
    )
    realized_nodes = sum(int(value.shape[0]) for value in x_dict.values())
    realized_edges = sum(int(value.shape[1]) for value in edge_index_dict.values())
    runtime.set_telemetry_gauge("hgt_requested_memory_nodes", int(memory_node_budget))
    runtime.set_telemetry_gauge("hgt_requested_canonical_edges", int(canonical_edge_budget))
    runtime.set_telemetry_gauge("hgt_total_node_budget", int(total_node_budget))
    runtime.set_telemetry_gauge("hgt_total_edge_budget", int(total_edge_budget))
    runtime.set_telemetry_gauge("hgt_realized_total_nodes", int(realized_nodes))
    runtime.set_telemetry_gauge("hgt_realized_total_edges", int(realized_edges))
    runtime.set_telemetry_gauge("hgt_semantic_nodes", int(realized_nodes - len(x_dict[NODE_TYPE])))
    runtime.set_telemetry_gauge("hgt_oom_retry_count", int(_oom_retry))
    selected_examples = sum(int(v.numel()) for v in y_dict.values())
    action_examples = sum(int(mask.sum().item()) for mask in action_masks.values())
    metadata = _stable_metadata()
    if not edge_index_dict:
        return HGTTrainingResult(epoch, "SKIPPED_NO_RELATIONS", runtime.unified_telemetry.model_version, None, 0.0, 0.0, action_examples, 0, None)
    if action_examples <= 0:
        return HGTTrainingResult(epoch, "SKIPPED_NO_ACTION_EVIDENCE", runtime.unified_telemetry.model_version, None, 0.0, 0.0, 0, 0, None)
    environment_families: dict[int, str] = {}
    for rows in action_meta.values():
        for row in rows:
            if row is None:
                continue
            environment_id = int(row[0])
            if environment_id in environment_families:
                continue
            try:
                environment_families[environment_id] = str(runtime.environments.resolve(environment_id).family)
            except KeyError:
                pass
    train_masks, val_masks = _split_masks(
        action_meta,
        action_masks,
        torch,
        environment_families=environment_families,
    )
    training_examples = _masked_count(train_masks, action_masks)
    validation_examples = _masked_count(val_masks, action_masks)
    if training_examples <= 0:
        return HGTTrainingResult(epoch, "SKIPPED_NO_TRAINING_EVIDENCE", runtime.unified_telemetry.model_version, None, 0.0, 0.0, action_examples, 0, None)
    if validation_examples <= 0:
        return HGTTrainingResult(epoch, "SKIPPED_NO_VALIDATION_EVIDENCE", runtime.unified_telemetry.model_version, None, 0.0, 0.0, action_examples, 0, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _HGTWrapper(
        metadata,
        input_dim=64,
        hidden_dim=int(config.hgt_hidden_dim),
        layers=int(config.hgt_layers),
        heads=int(config.hgt_heads),
    ).model.to(device)
    model_dir = Path(root) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "hgt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    parent_version, parent_checkpoint = manifest.get("current_model_version"), manifest.get("current_checkpoint")
    checkpoint_state = None
    if int(manifest.get("model_schema_version", 0)) != MODEL_SCHEMA_VERSION:
        parent_version = parent_checkpoint = None
    if parent_checkpoint:
        checkpoint_path = Path(root) / str(parent_checkpoint)
        if checkpoint_path.exists():
            checkpoint_state = torch.load(checkpoint_path, map_location=device)
            try:
                if int(checkpoint_state.get("model_schema_version", 0)) != MODEL_SCHEMA_VERSION:
                    raise RuntimeError("HGT checkpoint schema changed")
                model.load_state_dict(checkpoint_state["model_state"])
            except (RuntimeError, KeyError):
                checkpoint_state = None
                parent_version = parent_checkpoint = None
    x_device = {key: value.to(device) for key, value in x_dict.items()}
    edges_device = {key: value.to(device) for key, value in edge_index_dict.items()}

    parent_validation_loss = float("inf")
    parent_validation_accuracy = float("nan")
    if parent_version is not None:
        model.eval()
        with torch.no_grad():
            parent_logits, parent_values = model(x_device, edges_device)
            parent_val_loss_t, parent_validation_accuracy = _loss(
                parent_logits,
                parent_values,
                y_dict,
                val_masks,
                action_targets,
                action_masks,
                torch,
            )
        if parent_val_loss_t is not None:
            parent_validation_loss = float(parent_val_loss_t.cpu().item())

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    if checkpoint_state is not None and checkpoint_state.get("optimizer_state"):
        try:
            optimizer.load_state_dict(checkpoint_state["optimizer_state"])
        except (ValueError, RuntimeError):
            pass
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    start = time.perf_counter()
    training_loss = 0.0
    training_steps = 0
    last_grad_norm = 0.0
    model.train()
    for _ in range(max(1, int(training_epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(x_device, edges_device)
        loss, _ = _loss(logits, values, y_dict, train_masks, action_targets, action_masks, torch)
        if loss is None:
            break
        loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item())
        optimizer.step()
        training_loss = float(loss.detach().cpu().item())
        training_steps += 1
    if training_steps <= 0:
        return HGTTrainingResult(epoch, "SKIPPED_NO_TRAINING_EVIDENCE", runtime.unified_telemetry.model_version, parent_version, 0.0, parent_validation_loss if math.isfinite(parent_validation_loss) else 0.0, action_examples, 0, parent_checkpoint)

    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        logits, values = model(x_device, edges_device)
        val_loss_t, val_accuracy = _loss(logits, values, y_dict, val_masks, action_targets, action_masks, torch)
        train_loss_t, train_accuracy = _loss(logits, values, y_dict, train_masks, action_targets, action_masks, torch)
    inference_latency_ms = 1000.0 * (time.perf_counter() - inference_started)
    if val_loss_t is None:
        return HGTTrainingResult(epoch, "SKIPPED_NO_VALIDATION_EVIDENCE", runtime.unified_telemetry.model_version, parent_version, training_loss, 0.0, action_examples, training_steps, parent_checkpoint)
    validation_loss = float(val_loss_t.cpu().item())
    training_loss = float(train_loss_t.cpu().item()) if train_loss_t is not None else training_loss
    elapsed = max(1e-9, time.perf_counter() - start)
    version_index = int(manifest.get("version_index", 0)) + 1
    candidate_version = f"hgt-{version_index:06d}"
    checkpoint_rel = f"models/{candidate_version}.pt"
    checkpoint_path = Path(root) / checkpoint_rel
    action_score_sums: dict[int, dict[int, list[float]]] = {}
    context_score_sums: dict[int, dict[int, dict[int, list[float]]]] = {}
    for node_type, rows in action_meta.items():
        predicted = values[node_type].detach().cpu()
        for index, row in enumerate(rows):
            if row is None:
                continue
            environment_id, context_signature, action_id, _, _ = row
            score = float(predicted[index].item())
            action_score_sums.setdefault(environment_id, {}).setdefault(action_id, []).append(score)
            context_score_sums.setdefault(environment_id, {}).setdefault(context_signature, {}).setdefault(action_id, []).append(score)
    action_scores = {
        environment_id: {action_id: sum(scores) / len(scores) for action_id, scores in actions.items() if scores}
        for environment_id, actions in action_score_sums.items()
    }
    context_action_scores = {
        environment_id: {
            context: {action_id: sum(scores) / len(scores) for action_id, scores in actions.items() if scores}
            for context, actions in contexts.items()
        }
        for environment_id, contexts in context_score_sums.items()
    }
    promote = _should_promote(
        parent_version,
        parent_validation_loss,
        validation_loss,
        parent_validation_accuracy,
        val_accuracy,
    )
    status = "PROMOTED" if promote else "REJECTED"
    if promote:
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metadata": metadata,
                "input_dim": 64,
                "hidden_dim": int(config.hgt_hidden_dim),
                "layers": int(config.hgt_layers),
                "heads": int(config.hgt_heads),
                "validation_loss": validation_loss,
                "validation_accuracy": val_accuracy,
                "training_loss": training_loss,
                "action_scores": action_scores,
                "context_action_scores": context_action_scores,
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
        manifest = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "version_index": version_index,
            "current_model_version": candidate_version,
            "current_checkpoint": checkpoint_rel,
            "parent_model_version": parent_version,
            "validation_loss": validation_loss,
            "validation_accuracy": val_accuracy,
            "promotion_baseline_validation_loss": None if not math.isfinite(parent_validation_loss) else parent_validation_loss,
            "promotion_baseline_validation_accuracy": None if not math.isfinite(parent_validation_accuracy) else parent_validation_accuracy,
            "training_loss": training_loss,
            "graph_generation": int(read_view.generation),
            "selected_nodes": selected_examples,
            "examples": action_examples,
            "training_examples": training_examples,
            "validation_examples": validation_examples,
            "action_scores": action_scores,
            "context_action_scores": context_action_scores,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        model_version = candidate_version
        try:
            runtime.set_hgt_action_scores(action_scores, context_action_scores=context_action_scores)
        except TypeError:
            runtime.set_hgt_action_scores(action_scores)
    else:
        model_version = str(parent_version)
    gpu = read_gpu_snapshot()
    sample = HGTTrainingSample(
        training_loss=training_loss,
        validation_loss=validation_loss,
        training_step_latency_ms=1000.0 * elapsed / max(1, training_steps),
        training_examples_seen=training_examples,
        effective_batch_size=training_examples,
        gradient_norm=last_grad_norm,
        learning_rate=float(learning_rate),
        training_steps=training_steps,
        examples_per_second=(training_examples * max(1, training_steps)) / elapsed,
        gpu_memory_bytes=int(gpu.memory_used_bytes),
        gpu_utilization=float(gpu.utilization_percent),
        historical_retention=val_accuracy,
        current_curriculum_gain=max(0.0, val_accuracy - (1.0 / 3.0)),
        cross_family_validation_gain=val_accuracy,
        loss_by_head={"primary_valence": validation_loss, "discounted_action_value": validation_loss},
    )
    runtime.record_hgt_training(sample)
    retention_delta = 0.0 if not math.isfinite(parent_validation_accuracy) else val_accuracy - parent_validation_accuracy
    runtime.record_model_evolution(
        ModelEvolutionSample(
            model_version=model_version,
            parent_model_version=parent_version,
            training_examples_since_parent=training_examples,
            current_stage_delta=max(0.0, val_accuracy - (1.0 / 3.0)),
            historical_retention_delta=retention_delta,
            cross_family_transfer_delta=val_accuracy - train_accuracy,
            reasoning_improvement_delta=float(val_accuracy if not math.isfinite(parent_validation_accuracy) else val_accuracy - parent_validation_accuracy),
            inference_latency_delta_ms=float(inference_latency_ms - float(runtime.unified_telemetry.diagnostic_metrics().get("inference_latency_ms", 0.0))),
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
        examples=action_examples,
        training_steps=training_steps,
        checkpoint=checkpoint_rel if promote else parent_checkpoint,
        validation_accuracy=float(val_accuracy),
        inference_latency_ms=float(inference_latency_ms),
        subgraph_nodes=sum(int(value.shape[0]) for value in x_dict.values()),
        subgraph_edges=sum(int(value.shape[1]) for value in edge_index_dict.values()),
        relevance_precision=action_examples / max(1, selected_examples),
    )
