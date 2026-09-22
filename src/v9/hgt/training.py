from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from .epoch_dataset import (
    action_ranking_pairs,
    transition_training_rows,
    transition_training_rows_from_records,
)
from .grounding_objectives import GROUNDING_OBJECTIVES
from typing import Any

from v9.memory.model import MemoryLevel
from v9.memory.relations import RelationType
from v9.telemetry import HGTTrainingSample, ModelEvolutionSample, read_gpu_snapshot
from v9.runtime.scientific_modes import ScientificVisibilityMode
from .training_cut import TrainingCut, TrainingDeterminismMode

# Authoritative on-disk tensor/objective schema. Checkpoint compatibility must
# not depend on which runtime installer happened to be imported first.
MODEL_SCHEMA_VERSION = 8
MEMORY_NODE_TYPES = (
    "M0_EPISODE",
    "M1_GROUNDED_CONTINGENCY",
    "M1_NORMALIZED_RELATION",
    "M2_FAMILY",
    "M3_ROLE",
    "M4_CONCEPT",
    "M5_CONSEQUENCE",
    "M6_OUTCOME",
    "M7_STRATEGY",
)
NODE_TYPE = "M0_EPISODE"
BASE_OBJECTIVES = ("transition", "consequence", "relevance", "correspondence", "similarity", "strategy", "grounding", "deliberation_improvement", "invariance")
OBJECTIVE_NAMES = BASE_OBJECTIVES + GROUNDING_OBJECTIVES
AUX_OBJECTIVES = ("transition", "relevance", "correspondence", "similarity", "grounding", "deliberation_improvement", "invariance") + GROUNDING_OBJECTIVES
SEMANTIC_NODE_TYPES = ("ENTITY", "ACTION", "SYMBOL", "STATE", "EFFECT")


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
    training_accuracy: float = 0.0
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


def _memory_node_type(node: Any) -> str:
    return {
        1: "M0_EPISODE",
        100: "M1_GROUNDED_CONTINGENCY",
        150: "M1_NORMALIZED_RELATION",
        200: "M2_FAMILY",
        300: "M3_ROLE",
        400: "M4_CONCEPT",
        500: "M5_CONSEQUENCE",
        600: "M6_OUTCOME",
        700: "M7_STRATEGY",
    }[int(node.memory_type)]


def _metadata_for_graph(
    x_dict: dict[str, Any],
    edge_index_dict: dict[tuple[str, str, str], Any],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    # HGT only needs node/edge types realized in this training cut. Keeping the
    # full Cartesian memory schema creates hundreds of empty relation modules.
    node_types = sorted(x_dict)
    edge_types = sorted(edge_index_dict)
    return node_types, edge_types


def _semantic_rows(payload: dict[str, Any]) -> tuple[tuple[int, int, int, int, float], ...]:
    rows: list[tuple[int, int, int, int, float]] = []
    for key in ("semantic_before", "semantic_action", "semantic_options", "semantic_after", "semantic_effects"):
        for row in payload.get(key, ()) or ():
            if isinstance(row, (list, tuple)) and len(row) == 5:
                rows.append((int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])))
    return tuple(rows)


def _predictive_semantic_rows(payload: dict[str, Any]) -> tuple[tuple[int, int, int, int, float], ...]:
    """Only information available before observing the selected action outcome."""
    rows: list[tuple[int, int, int, int, float]] = []
    for key in ("semantic_before", "semantic_action", "semantic_options"):
        for row in payload.get(key, ()) or ():
            if isinstance(row, (list, tuple)) and len(row) == 5:
                rows.append((int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])))
    return tuple(rows)


def _node_feature(node: Any, payload: dict[str, Any], dim: int, torch: Any):
    # Outcome fields (valence, success/failure/truncation, levels completed,
    # semantic_after/effects) are labels. They must not be policy inputs.
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
    ]
    while len(values) < dim:
        values.append(0.0)
    for kind, subject, relation, obj, value in _predictive_semantic_rows(payload):
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
        return "SYMBOL"
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
    maximum = max(0, int(limit))
    if maximum <= 0:
        return []
    per_environment: dict[int, list[tuple[tuple[int, int, int, int], Any, Any]]] = {}
    global_heap: list[tuple[tuple[int, int, int, int], Any, Any]] = []
    per_environment_cap = 4
    for uid, node in read_view.nodes.items():
        if node.level is not MemoryLevel.M0:
            continue
        payload = read_view.payloads.get(uid, {})
        if payload.get("action_id") is None or payload.get("environment_instance_id") is None:
            continue
        environment_id = int(payload["environment_instance_id"])
        rank = (abs(int(payload.get("primary_valence", 0))), int(node.created_watermark), int(uid.hi), int(uid.lo))
        row = (rank, uid, node)
        bucket = per_environment.setdefault(environment_id, [])
        heapq.heappush(bucket, row)
        if len(bucket) > per_environment_cap:
            heapq.heappop(bucket)
        heapq.heappush(global_heap, row)
        if len(global_heap) > maximum:
            heapq.heappop(global_heap)
    selected: dict[Any, Any] = {}
    ordered_buckets = {environment_id: sorted(rows, key=lambda row: row[0], reverse=True) for environment_id, rows in per_environment.items()}
    depth = 0
    environment_ids = sorted(ordered_buckets)
    while len(selected) < maximum:
        added = False
        for environment_id in environment_ids:
            rows = ordered_buckets[environment_id]
            if depth >= len(rows):
                continue
            _, uid, node = rows[depth]
            if uid not in selected:
                selected[uid] = node
                added = True
                if len(selected) >= maximum:
                    break
        if not added:
            break
        depth += 1
    if len(selected) < maximum:
        for _, uid, node in sorted(global_heap, key=lambda row: row[0], reverse=True):
            if uid in selected:
                continue
            selected[uid] = node
            if len(selected) >= maximum:
                break
    return list(selected.items())


def _select_connected_nodes(read_view: Any, max_nodes: int) -> tuple[Any, ...]:
    maximum = max(1, int(max_nodes))
    selected: dict[Any, Any] = {}

    # Preserve action supervision while guaranteeing that every available Hydra
    # memory level participates in the HGT graph. A recency-only graph otherwise
    # becomes dominated by M0/M1 and higher abstractions contribute no messages.
    behavior_budget = max(1, maximum // 2)
    for uid, node in _recent_behavior_nodes(read_view, behavior_budget):
        selected[uid] = node

    available_levels = {
        MemoryLevel(level): []
        for level in range(8)
    }
    for uid, node in read_view.nodes.items():
        available_levels[node.level].append((uid, node))
    abstraction_budget = max(0, maximum - len(selected))
    nonempty_levels = [level for level, rows in available_levels.items() if rows]
    per_level = max(1, abstraction_budget // max(1, len(nonempty_levels)))
    for level in nonempty_levels:
        rows = heapq.nlargest(
            per_level,
            available_levels[level],
            key=lambda row: (int(row[1].created_watermark), row[0]),
        )
        for uid, node in rows:
            if len(selected) >= maximum:
                break
            selected.setdefault(uid, node)

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
    include_objectives: bool = False,
):
    torch, _, _ = _require_torch()
    selected_uids = _select_connected_nodes(read_view, max_nodes)
    ordered_uids = sorted(selected_uids, key=lambda uid: (int(read_view.nodes[uid].created_watermark), uid))

    uids_by_type: dict[str, list[Any]] = {node_type: [] for node_type in MEMORY_NODE_TYPES}
    for uid in ordered_uids:
        uids_by_type[_memory_node_type(read_view.nodes[uid])].append(uid)
    index_by_uid = {
        uid: (node_type, index)
        for node_type, uids in uids_by_type.items()
        for index, uid in enumerate(uids)
    }

    relation_nodes: dict[str, set[Any]] = {}
    for edge in read_view.edges:
        if edge.source not in index_by_uid or edge.target not in index_by_uid:
            continue
        relation_nodes.setdefault(str(edge.relation.value), set()).update((edge.source, edge.target))

    x_dict: dict[str, Any] = {}
    y_dict: dict[str, Any] = {}
    action_target_dict: dict[str, Any] = {}
    action_mask_dict: dict[str, Any] = {}
    action_meta: dict[str, list[Any]] = {}
    task_target_dict: dict[str, dict[str, Any]] = {name: {} for name in AUX_OBJECTIVES}
    task_mask_dict: dict[str, dict[str, Any]] = {name: {} for name in AUX_OBJECTIVES}
    episode_rows: dict[tuple[int, int], list[tuple[str, int, int, int, bool, bool, bool, int]]] = {}

    for node_type, uids in uids_by_type.items():
        if not uids:
            continue
        features = []
        labels = []
        action_targets = [0.0] * len(uids)
        action_masks: list[bool] = []
        rows_meta: list[Any] = []
        targets: dict[str, list[float]] = {name: [] for name in AUX_OBJECTIVES}
        masks: dict[str, list[bool]] = {name: [] for name in AUX_OBJECTIVES}
        for index, uid in enumerate(uids):
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
            semantic_rows = _semantic_rows(payload)
            transition_positive = bool(payload.get("semantic_effects")) or (
                payload.get("context_signature") is not None
                and payload.get("next_context_signature") is not None
                and int(payload.get("context_signature")) != int(payload.get("next_context_signature"))
            )
            targets["transition"].append(float(transition_positive))
            masks["transition"].append(bool(usable_action))
            relevance_positive = bool(
                abs(int(payload.get("primary_valence", 0))) > 0
                or int(payload.get("support", 0)) > 1
                or int(payload.get("recurrence", 0)) > 1
                or bool(payload.get("validated", False))
                or int(payload.get("explanatory_reach", 0)) > 0
            )
            targets["relevance"].append(float(relevance_positive))
            masks["relevance"].append(True)
            targets["correspondence"].append(float(uid in relation_nodes.get(RelationType.TRANSFER_CORRESPONDENCE.value, set())))
            masks["correspondence"].append(node.level in {MemoryLevel.M3, MemoryLevel.M4})
            targets["similarity"].append(float(uid in relation_nodes.get(RelationType.SIMILAR_TO.value, set())))
            masks["similarity"].append(node.level in {MemoryLevel.M2, MemoryLevel.M3, MemoryLevel.M4})
            grounding_positive = uid in relation_nodes.get(RelationType.GROUNDS.value, set()) or any(int(row[0]) == 7 for row in semantic_rows)
            targets["grounding"].append(float(grounding_positive))
            masks["grounding"].append(bool(semantic_rows) or node.level in {MemoryLevel.M0, MemoryLevel.M1})
            has_symbol = payload.get("symbol_identity") is not None or any(int(row[0]) == 7 for row in semantic_rows)
            cross_modal = uid in relation_nodes.get(RelationType.GROUNDS.value, set()) or uid in relation_nodes.get(RelationType.TRANSFER_CORRESPONDENCE.value, set())
            prospective = bool(payload.get("symbol_prediction_gain", 0.0) > 0.0 or payload.get("prospective_prediction", False))
            heldout = bool(payload.get("heldout_transfer", False))
            composition = bool(payload.get("novel_composition", False))
            calibrated = float(payload.get("grounding_confidence", 1.0 if grounding_positive else 0.0))
            grounding_targets = {
                "symbol_conditioned_interaction_prediction": float(has_symbol and transition_positive),
                "symbol_conditioned_relevant_memory_retrieval": float(has_symbol and relevance_positive),
                "world_to_symbol_generalization": float(cross_modal and has_symbol),
                "heldout_symbol_composition": float(composition and heldout),
                "symbol_conditioned_action_ranking": float(has_symbol and usable_action and int(payload.get("primary_valence", 0)) > 0),
                "shuffled_alignment_discrimination": float(cross_modal and has_symbol),
                "grounding_confidence_calibration": max(0.0, min(1.0, calibrated)),
            }
            causal_watermark = int(payload.get("grounding_causal_watermark", node.created_watermark))
            target_watermark = int(node.created_watermark)
            causal_ok = causal_watermark <= target_watermark
            for grounding_objective, grounding_target in grounding_targets.items():
                targets[grounding_objective].append(grounding_target)
                masks[grounding_objective].append(bool(causal_ok and (has_symbol or cross_modal)))
            deliberation_target = 1.0 if bool(payload.get("task_success", False)) or int(payload.get("levels_completed", 0)) > 0 else (-1.0 if bool(payload.get("task_failure", False)) else 0.0)
            targets["deliberation_improvement"].append(deliberation_target)
            masks["deliberation_improvement"].append(bool(usable_action))
            invariance_positive = bool(payload.get("validated", False)) or uid in relation_nodes.get(RelationType.OUTCOME_EQUIVALENT.value, set())
            targets["invariance"].append(float(invariance_positive))
            masks["invariance"].append(node.level in {MemoryLevel.M4, MemoryLevel.M5, MemoryLevel.M6})
            action_masks.append(bool(usable_action))
            if usable_action:
                env, context, action, episode = int(environment_id), int(context_signature), int(action_id), int(episode_id)
                rows_meta.append((
                    env,
                    context,
                    action,
                    episode,
                    int(node.created_watermark),
                    str(payload.get("game_scenario", "")),
                ))
                episode_rows.setdefault((env, episode), []).append((
                    node_type, index, int(node.created_watermark), valence,
                    bool(payload.get("task_success", False)),
                    bool(payload.get("task_failure", False)),
                    bool(payload.get("task_truncated", False)),
                    int(payload.get("levels_completed", 0)),
                ))
            else:
                rows_meta.append(None)
        x_dict[node_type] = torch.stack(features, dim=0)
        y_dict[node_type] = torch.tensor(labels, dtype=torch.long)
        action_target_dict[node_type] = torch.tensor(action_targets, dtype=torch.float32)
        action_mask_dict[node_type] = torch.tensor(action_masks, dtype=torch.bool)
        action_meta[node_type] = rows_meta
        for name in AUX_OBJECTIVES:
            task_target_dict[name][node_type] = torch.tensor(targets[name], dtype=torch.float32)
            task_mask_dict[name][node_type] = torch.tensor(masks[name], dtype=torch.bool)

    gamma = float(return_discount)
    for rows in episode_rows.values():
        ordered = sorted(rows, key=lambda row: row[2])
        previous_levels = 0
        immediate = []
        for node_type, index, watermark, valence, task_success, task_failure, task_truncated, levels_completed in ordered:
            level_gain = max(0, int(levels_completed) - int(previous_levels))
            previous_levels = max(int(previous_levels), int(levels_completed))
            signal = float(valence) + float(task_success) - float(task_failure) - (0.10 if task_truncated else 0.0)
            if level_gain:
                signal += min(1.0, 0.5 * float(level_gain))
            immediate.append((node_type, index, max(-1.0, min(1.0, signal))))
        running = 0.0
        for node_type, index, signal in reversed(immediate):
            running = max(-1.0, min(1.0, float(signal) + gamma * running))
            action_target_dict[node_type][index] = running

    eligible = (edge for edge in read_view.edges if edge.source in index_by_uid and edge.target in index_by_uid)
    selected_edges = heapq.nlargest(
        max(1, min(int(max_edges), int(max_total_edges))),
        eligible,
        key=lambda edge: max(
            int(read_view.nodes[edge.source].created_watermark),
            int(read_view.nodes[edge.target].created_watermark),
        ),
    )
    edges: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for edge in selected_edges:
        source_type, source_index = index_by_uid[edge.source]
        target_type, target_index = index_by_uid[edge.target]
        edges.setdefault((source_type, str(edge.relation.value), target_type), []).append((source_index, target_index))

    semantic_links = []
    semantic_tables: dict[str, dict[Any, int]] = {}
    semantic_features: dict[str, list[Any]] = {}
    semantic_node_budget = max(0, int(max_total_nodes) - len(ordered_uids))
    semantic_link_budget = max(0, (int(max_total_edges) - len(selected_edges)) // 2)
    semantic_node_count = 0
    for uid in ordered_uids:
        if len(semantic_links) >= semantic_link_budget:
            break
        memory_type, memory_index = index_by_uid[uid]
        payload = dict(read_view.payloads.get(uid, {}))
        facts = sorted(set(_semantic_rows(payload)) | ({(7, int(payload["symbol_identity"][2]), int(payload["symbol_identity"][1]), int(payload["symbol_identity"][0]), float(payload["symbol_identity"][3]))} if payload.get("symbol_identity") is not None else set()), key=_semantic_priority)
        for fact in facts[: max(1, int(max_semantic_facts_per_memory))]:
            if len(semantic_links) >= semantic_link_budget:
                break
            semantic_type = _semantic_node_type(int(fact[0]))
            table = semantic_tables.setdefault(semantic_type, {})
            semantic_index = table.get(fact)
            if semantic_index is None:
                if semantic_node_count >= semantic_node_budget:
                    continue
                semantic_index = len(table)
                table[fact] = semantic_index
                semantic_features.setdefault(semantic_type, []).append(_semantic_node_feature(fact, input_dim, torch))
                semantic_node_count += 1
            semantic_links.append((memory_type, memory_index, semantic_type, semantic_index))

    for semantic_type, rows in semantic_features.items():
        x_dict[semantic_type] = torch.stack(rows, dim=0)
        count = len(rows)
        y_dict[semantic_type] = torch.zeros(count, dtype=torch.long)
        action_target_dict[semantic_type] = torch.zeros(count, dtype=torch.float32)
        action_mask_dict[semantic_type] = torch.zeros(count, dtype=torch.bool)
        action_meta[semantic_type] = [None] * count
        for name in AUX_OBJECTIVES:
            task_target_dict[name][semantic_type] = torch.zeros(count, dtype=torch.float32)
            task_mask_dict[name][semantic_type] = torch.zeros(count, dtype=torch.bool)

    for memory_type, memory_index, semantic_type, semantic_index in semantic_links:
        edges.setdefault((memory_type, "SYMBOL_OCCURRENCE" if semantic_type == "SYMBOL" else "SEMANTIC", semantic_type), []).append((memory_index, semantic_index))
        edges.setdefault((semantic_type, "OCCURS_IN_MEMORY" if semantic_type == "SYMBOL" else "SEMANTIC_OF", memory_type), []).append((semantic_index, memory_index))

    edge_index_dict = {
        key: torch.tensor(pairs, dtype=torch.long).t().contiguous()
        for key, pairs in edges.items()
        if pairs
    }
    if include_objectives:
        return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta, task_target_dict, task_mask_dict
    return x_dict, edge_index_dict, y_dict, action_target_dict, action_mask_dict, action_meta


class _HGTWrapper:
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], *, input_dim: int, hidden_dim: int, layers: int, heads: int):
        torch, nn, HGTConv = _require_torch()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = nn.ModuleDict({node_type: nn.Linear(input_dim, hidden_dim) for node_type in metadata[0]})
                self.layers = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata, heads=heads) for _ in range(layers)])
                self.valence_heads = nn.ModuleDict({node_type: nn.Linear(hidden_dim, 3) for node_type in metadata[0]})
                self.value_heads = nn.ModuleDict({node_type: nn.Linear(hidden_dim, 1) for node_type in metadata[0]})
                self.objective_heads = nn.ModuleDict({
                    objective: nn.ModuleDict({node_type: nn.Linear(hidden_dim, 1) for node_type in metadata[0]})
                    for objective in AUX_OBJECTIVES
                })
                self.objective_log_vars = nn.ParameterDict({
                    objective: nn.Parameter(torch.zeros(()))
                    for objective in OBJECTIVE_NAMES
                })

            def forward(self, x_dict, edge_index_dict):
                state = {key: self.encoders[key](value).relu() for key, value in x_dict.items()}
                for layer in self.layers:
                    updated = layer(state, edge_index_dict)
                    state = {key: (state[key] if updated.get(key) is None else updated[key]).relu() for key in state}
                auxiliary = {
                    objective: {
                        key: self.objective_heads[objective][key](value).squeeze(-1)
                        for key, value in state.items()
                    }
                    for objective in AUX_OBJECTIVES
                }
                return (
                    {key: self.valence_heads[key](value) for key, value in state.items()},
                    {key: self.value_heads[key](value).squeeze(-1) for key, value in state.items()},
                    auxiliary,
                )

        self.model = Model()


def _contextualize_transition_rows(rows: list[dict[str, Any]], read_view: Any) -> list[dict[str, Any]]:
    """Attach bounded M0-M7 relational context summaries to transition rows."""
    by_context: dict[int, list[tuple[int, int, int]]] = {}
    for uid, payload in read_view.payloads.items():
        context = payload.get("context_signature")
        if context is None:
            continue
        node = read_view.nodes.get(uid)
        if node is None:
            continue
        by_context.setdefault(int(context), []).append((int(node.level), int(node.created_watermark), int(payload.get("primary_valence", 0))))
    contextualized = []
    for row in rows:
        related = sorted(by_context.get(int(row["context_signature"]), ()), key=lambda item: item[1], reverse=True)[:32]
        copy = dict(row)
        copy["hydra_context_levels"] = tuple(level for level, _, _ in related)
        copy["hydra_context_valence_sum"] = sum(valence for _, _, valence in related)
        copy["hydra_context_count"] = len(related)
        contextualized.append(copy)
    return contextualized


def _context_is_validation(game: str, context: int, validation_fraction: float) -> bool:
    digest = hashlib.blake2b(
        f"{game}:{int(context)}".encode("utf-8"),
        digest_size=8,
        person=b"v9-hgt-val",
    ).digest()
    threshold = int(max(0.0, min(1.0, float(validation_fraction))) * (1 << 64))
    return int.from_bytes(digest, "big") < threshold


def _split_policy_masks(action_meta, action_masks, torch, *, validation_fraction: float):
    train_masks = {}
    validation_masks = {}
    for node_type, mask in action_masks.items():
        train = mask.clone()
        validation = torch.zeros_like(mask, dtype=torch.bool)
        rows = action_meta.get(node_type, ())
        for index, row in enumerate(rows):
            if row is None or not bool(mask[index]):
                continue
            _env, context, _action, _episode, _watermark, game = row
            if _context_is_validation(str(game), int(context), validation_fraction):
                train[index] = False
                validation[index] = True
        train_masks[node_type] = train
        validation_masks[node_type] = validation
    return train_masks, validation_masks


def _split_ranking_pairs(pairs, *, validation_fraction: float):
    train = []
    validation = []
    for best, worst in pairs:
        game = str(best.get("game_scenario", ""))
        context = int(best.get("context_signature", 0))
        target = validation if _context_is_validation(game, context, validation_fraction) else train
        target.append((best, worst))
    return train, validation


def _explicit_action_ranking_loss(
    value_dict,
    action_meta,
    policy_masks,
    pairs,
    torch,
):
    if not pairs:
        return None, 0, 0
    lookup: dict[tuple[str, int, int], list[Any]] = {}
    for node_type, rows in action_meta.items():
        values = value_dict.get(node_type)
        masks = policy_masks.get(node_type)
        if values is None or masks is None:
            continue
        for index, row in enumerate(rows):
            if row is None or not bool(masks[index]):
                continue
            _env, context, action, _episode, _watermark, game = row
            lookup.setdefault((str(game), int(context), int(action)), []).append(values[index])

    terms = []
    correct = total = 0
    for best, worst in pairs:
        game = str(best.get("game_scenario", ""))
        context = int(best.get("context_signature", 0))
        best_values = lookup.get((game, context, int(best.get("action_id", 0))), ())
        worst_values = lookup.get((game, context, int(worst.get("action_id", 0))), ())
        if not best_values or not worst_values:
            continue
        best_score = torch.stack(tuple(best_values)).mean()
        worst_score = torch.stack(tuple(worst_values)).mean()
        terms.append(torch.nn.functional.softplus(-(best_score - worst_score)))
        correct += int(float(best_score.detach().cpu()) > float(worst_score.detach().cpu()))
        total += 1
    if not terms:
        return None, 0, 0
    return torch.stack(terms).mean(), correct, total


def _policy_validation_loss(
    logits_dict,
    value_dict,
    y_dict,
    action_targets,
    validation_masks,
    torch,
):
    terms = []
    for node_type, mask in validation_masks.items():
        logits = logits_dict.get(node_type)
        values = value_dict.get(node_type)
        if logits is None or values is None or not bool(mask.any()):
            continue
        selected = mask.to(logits.device)
        target_class = y_dict[node_type].to(logits.device)[selected]
        target_value = action_targets[node_type].to(values.device)[selected]
        terms.append(torch.nn.functional.cross_entropy(logits[selected], target_class))
        terms.append(torch.nn.functional.smooth_l1_loss(values[selected], target_value))
    if not terms:
        return None
    return torch.stack(terms).mean()


def _bounded_advantage_scores(scores: dict[int, float], scale: float) -> dict[int, float]:
    if not scores or scale <= 0.0:
        return {int(action): 0.0 for action in scores}
    mean = sum(float(value) for value in scores.values()) / len(scores)
    centered = {int(action): float(value) - mean for action, value in scores.items()}
    maximum = max((abs(value) for value in centered.values()), default=0.0)
    if maximum <= 1e-12:
        return {action: 0.0 for action in centered}
    return {action: float(scale) * value / maximum for action, value in centered.items()}


def _policy_score_scale(config: Any, *, validation_accuracy: float, validation_pairs: int) -> float:
    if validation_pairs <= 0:
        return 0.0
    threshold = float(config.hgt_min_validation_ranking_accuracy)
    accuracy = float(validation_accuracy)
    if accuracy <= threshold:
        return 0.0
    quality = (accuracy - threshold) / max(1e-9, 1.0 - threshold)
    return float(config.hgt_max_policy_score) * max(0.0, min(1.0, quality))


def _masked_count(masks: dict[str, Any], action_masks: dict[str, Any]) -> int:
    return sum(int((masks[key] & action_masks[key]).sum().item()) for key in masks)


def _loss(
    logits_dict,
    value_dict,
    auxiliary_dict,
    y_dict,
    masks,
    action_targets,
    action_masks,
    task_targets,
    task_masks,
    torch,
    *,
    objective_weights,
    log_vars,
    dynamic_weighting: bool,
):
    raw_losses: dict[str, Any] = {}
    correct = total = 0

    consequence_losses = []
    for node_type, logits in logits_dict.items():
        mask = masks[node_type].to(logits.device) & action_masks[node_type].to(logits.device)
        if not bool(mask.any()):
            continue
        target = y_dict[node_type].to(logits.device)[mask]
        selected = logits[mask]
        consequence_losses.append(torch.nn.functional.cross_entropy(selected, target))
        correct += int((selected.argmax(dim=-1) == target).sum().item())
        total += int(target.numel())
    if consequence_losses:
        raw_losses["consequence"] = torch.stack(consequence_losses).mean()

    strategy_losses = []
    for node_type, values in value_dict.items():
        mask = masks[node_type].to(values.device) & action_masks[node_type].to(values.device)
        if bool(mask.any()):
            target = action_targets[node_type].to(values.device)[mask]
            strategy_losses.append(torch.nn.functional.smooth_l1_loss(values[mask], target))
    if strategy_losses:
        raw_losses["strategy"] = torch.stack(strategy_losses).mean()

    for objective in AUX_OBJECTIVES:
        losses = []
        for node_type, predictions in auxiliary_dict[objective].items():
            policy_mask = masks[node_type].to(predictions.device)
            action_mask = action_masks[node_type].to(predictions.device)
            mask = task_masks[objective][node_type].to(predictions.device) & (
                (~action_mask) | policy_mask
            )
            if not bool(mask.any()):
                continue
            target = task_targets[objective][node_type].to(predictions.device)[mask]
            selected = predictions[mask]
            if objective == "deliberation_improvement":
                losses.append(torch.nn.functional.smooth_l1_loss(selected, target))
            else:
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(selected, target))
        if losses:
            raw_losses[objective] = torch.stack(losses).mean()

    if not raw_losses:
        return None, 0.0, {}

    weights = list(float(weight) for weight in objective_weights)
    if len(weights) < len(OBJECTIVE_NAMES):
        weights.extend([0.5] * (len(OBJECTIVE_NAMES) - len(weights)))
    weight_map = {name: float(weight) for name, weight in zip(OBJECTIVE_NAMES, weights)}
    total_loss = None
    active_weight = 0.0
    for objective, loss in raw_losses.items():
        weight = max(1e-9, weight_map.get(objective, 1.0))
        if dynamic_weighting:
            log_var = log_vars[objective]
            term = weight * (torch.exp(-log_var) * loss + log_var)
        else:
            term = weight * loss
        total_loss = term if total_loss is None else total_loss + term
        active_weight += weight
    total_loss = total_loss / max(1e-9, active_weight)
    by_head = {name: float(loss.detach().cpu().item()) for name, loss in raw_losses.items()}
    return total_loss, correct / max(1, total), by_head

def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


def _retry_after_oom(runtime: Any, *, epoch: int, training_epochs: int, learning_rate: float, root: str | Path, allow_promotion: bool, budget_scale: float, oom_retry: int, exc: RuntimeError):
    config = runtime.config.scientific
    if not _is_cuda_oom(exc) or int(oom_retry) >= int(config.hgt_oom_retry_limit):
        raise exc
    # Do not keep the failed autograd frame (and its CUDA tensors) alive while
    # the bounded retry constructs a smaller graph.
    exc.__traceback__ = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    next_scale = max(0.125, float(budget_scale) * 0.5)
    runtime.set_telemetry_gauge("hgt_oom_retry_count", int(oom_retry) + 1)
    runtime.set_telemetry_gauge("hgt_oom_shedding_factor", float(next_scale))
    return train_hgt_epoch(
        runtime,
        epoch=epoch,
        training_epochs=training_epochs,
        learning_rate=learning_rate,
        root=root,
        allow_promotion=allow_promotion,
        _budget_scale=next_scale,
        _oom_retry=int(oom_retry) + 1,
    )


def resolve_hgt_behavior_test(runtime: Any, *, root: str | Path, accepted: bool) -> str | None:
    """Resolve the candidate that was actually exercised during the just-finished sampling epoch."""
    manifest_path = Path(root) / "models" / "hgt_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = manifest.get("candidate_model_version")
    if not candidate or manifest.get("candidate_status") != "TESTING_PENDING_BEHAVIOR":
        return None
    if accepted:
        accepted_checkpoint = f"models/{candidate}.pt"
        manifest.update({
            "candidate_status": "PROMOTED",
            "last_accepted_model_version": str(candidate),
            "accepted_model_version": str(candidate),
            "accepted_checkpoint": accepted_checkpoint,
            "current_model_version": str(candidate),
            "current_checkpoint": accepted_checkpoint,
            "parent_model_version": None,
            "candidate_model_version": None,
        })
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        runtime.set_telemetry_gauge("hgt_behavior_test_result", "PROMOTED")
        return str(candidate)
    rolled_back = rollback_hgt_model(runtime, root=root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["candidate_status"] = "REJECTED_BEHAVIOR_GATE"
    manifest["candidate_model_version"] = None
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    runtime.set_telemetry_gauge("hgt_behavior_test_result", "REJECTED_BEHAVIOR_GATE")
    return rolled_back

def load_hgt_policy_version(
    runtime: Any, *, root: str | Path, model_version: str
) -> str | None:
    """Load one HGT checkpoint into runtime policy without mutating the model manifest."""
    selected = str(model_version)
    checkpoint_path = Path(root) / "models" / f"{selected}.pt"
    if not checkpoint_path.exists():
        return None
    torch, _, _ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    action_scores = {
        int(environment): {
            int(action): float(score) for action, score in actions.items()
        }
        for environment, actions in dict(checkpoint.get("action_scores", {})).items()
    }
    context_action_scores = {
        int(environment): {
            int(context): {
                int(action): float(score) for action, score in actions.items()
            }
            for context, actions in contexts.items()
        }
        for environment, contexts in dict(
            checkpoint.get("context_action_scores", {})
        ).items()
    }
    try:
        runtime.set_hgt_action_scores(
            action_scores, context_action_scores=context_action_scores
        )
    except TypeError:
        runtime.set_hgt_action_scores(action_scores)
    runtime.unified_telemetry.model_version = selected
    return selected


def rollback_hgt_model(runtime: Any, *, root: str | Path) -> str | None:
    """Restore the parent checkpoint after a measured behavioral regression."""
    model_dir = Path(root) / "models"
    manifest_path = model_dir / "hgt_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_version = manifest.get("parent_model_version") or manifest.get("last_accepted_model_version")
    current_version = manifest.get("current_model_version")
    if not current_version:
        return None
    if not parent_version:
        manifest.update(
            {
                "current_model_version": None,
                "current_checkpoint": None,
                "parent_model_version": None,
                "rollback_from_model_version": str(current_version),
                "rollback_to_model_version": "untrained",
                "accepted_model_version": None,
                "accepted_checkpoint": None,
            }
        )
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        try:
            runtime.set_hgt_action_scores({}, context_action_scores={})
        except TypeError:
            runtime.set_hgt_action_scores({})
        runtime.unified_telemetry.model_version = "untrained"
        runtime.set_telemetry_gauge("hgt_behavior_rollback", 1)
        runtime.set_telemetry_gauge("hgt_rollback_from_model", str(current_version))
        runtime.set_telemetry_gauge("hgt_rollback_to_model", "untrained")
        return "untrained"
    if parent_version == current_version:
        return None
    checkpoint_rel = f"models/{parent_version}.pt"
    checkpoint_path = Path(root) / checkpoint_rel
    if not checkpoint_path.exists():
        return None
    torch, _, _ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    action_scores = {
        int(environment): {int(action): float(score) for action, score in actions.items()}
        for environment, actions in dict(checkpoint.get("action_scores", {})).items()
    }
    context_action_scores = {
        int(environment): {
            int(context): {int(action): float(score) for action, score in actions.items()}
            for context, actions in contexts.items()
        }
        for environment, contexts in dict(checkpoint.get("context_action_scores", {})).items()
    }
    manifest.update(
        {
            "current_model_version": str(parent_version),
            "current_checkpoint": checkpoint_rel,
            "parent_model_version": None,
            "accepted_model_version": str(parent_version),
            "accepted_checkpoint": checkpoint_rel,
            "rollback_from_model_version": str(current_version),
        }
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    try:
        runtime.set_hgt_action_scores(action_scores, context_action_scores=context_action_scores)
    except TypeError:
        runtime.set_hgt_action_scores(action_scores)
    runtime.unified_telemetry.model_version = str(parent_version)
    runtime.set_telemetry_gauge("hgt_behavior_rollback", 1)
    runtime.set_telemetry_gauge("hgt_rollback_from_model", str(current_version))
    runtime.set_telemetry_gauge("hgt_rollback_to_model", str(parent_version))
    return str(parent_version)



def _checkpoint_architecture(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, (list, tuple)) or len(metadata) != 2:
        raise RuntimeError("HGT checkpoint is missing self-describing metadata")
    node_types = [str(value) for value in metadata[0]]
    edge_types = [tuple(str(part) for part in edge) for edge in metadata[1]]
    if any(len(edge) != 3 for edge in edge_types):
        raise RuntimeError("HGT checkpoint contains invalid edge metadata")
    return {
        "model_schema_version": int(checkpoint.get("model_schema_version", 0)),
        "metadata": (node_types, edge_types),
        "input_dim": int(checkpoint["input_dim"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "layers": int(checkpoint["layers"]),
        "heads": int(checkpoint["heads"]),
    }


def _architecture_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".metadata.json")


def _write_architecture_sidecar(checkpoint_path: Path, architecture: dict[str, Any]) -> None:
    payload = dict(architecture)
    payload["metadata"] = [
        list(architecture["metadata"][0]),
        [list(edge) for edge in architecture["metadata"][1]],
    ]
    sidecar = _architecture_sidecar_path(checkpoint_path)
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)


def _load_self_describing_model(checkpoint: dict[str, Any], device: Any):
    architecture = _checkpoint_architecture(checkpoint)
    if architecture["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise RuntimeError("HGT checkpoint schema changed")
    model = _HGTWrapper(
        architecture["metadata"],
        input_dim=architecture["input_dim"],
        hidden_dim=architecture["hidden_dim"],
        layers=architecture["layers"],
        heads=architecture["heads"],
    ).model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, architecture


def _load_compatible_training_parent(
    checkpoint: object,
    *,
    device: Any,
    model: Any,
    current_architecture: dict[str, Any],
    runtime: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resume compatible tensors while retaining older parents for behavior.

    Action-score policies remain readable across HGT tensor schema changes.
    Model and optimizer tensors do not, so an accepted older checkpoint stays
    available for matched evaluation and rollback while training starts fresh.
    """
    if not isinstance(checkpoint, dict):
        raise RuntimeError("HGT checkpoint payload is not a mapping")
    try:
        checkpoint_schema = int(checkpoint.get("model_schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("HGT checkpoint has an invalid schema version") from exc
    if checkpoint_schema != MODEL_SCHEMA_VERSION:
        runtime.set_telemetry_gauge("hgt_training_parent_behavior_only", 1)
        runtime.set_telemetry_gauge("hgt_training_parent_schema_version", checkpoint_schema)
        runtime.set_telemetry_gauge("hgt_training_schema_version", MODEL_SCHEMA_VERSION)
        return None, None

    parent_model, parent_architecture = _load_self_describing_model(checkpoint, device)
    same_architecture = all(
        parent_architecture[key] == current_architecture[key]
        for key in ("metadata", "input_dim", "hidden_dim", "layers", "heads")
    )
    if same_architecture:
        model.load_state_dict(parent_model.state_dict())
        resumable = checkpoint
    else:
        _migrate_model_state(
            parent_model,
            parent_architecture["metadata"],
            model,
            current_architecture["metadata"],
        )
        runtime.set_telemetry_gauge("hgt_architecture_evolved", 1)
        runtime.set_telemetry_gauge("hgt_parent_metadata_edge_types", len(parent_architecture["metadata"][1]))
        runtime.set_telemetry_gauge("hgt_current_metadata_edge_types", len(current_architecture["metadata"][1]))
        resumable = dict(checkpoint)
        resumable["optimizer_state"] = None
    del parent_model
    return resumable, parent_architecture


def _migrate_model_state(source_model: Any, source_metadata: tuple[list[str], list[tuple[str, str, str]]], target_model: Any, target_metadata: tuple[list[str], list[tuple[str, str, str]]]) -> None:
    source = source_model.state_dict()
    target = target_model.state_dict()
    source_edges = {edge: index for index, edge in enumerate(source_metadata[1])}
    target_edges = {edge: index for index, edge in enumerate(target_metadata[1])}
    relation_tensors = {
        f"layers.{layer}.{name}.weight"
        for layer in range(len(getattr(target_model, "layers", ())))
        for name in ("k_rel", "v_rel")
    }
    for key, value in source.items():
        if key in relation_tensors:
            continue
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            target[key].copy_(value)
    for key in relation_tensors:
        if key not in source or key not in target:
            continue
        for edge, source_index in source_edges.items():
            target_index = target_edges.get(edge)
            if target_index is not None:
                target[key][target_index].copy_(source[key][source_index])
    target_model.load_state_dict(target)

def train_hgt_epoch(runtime: Any, *, epoch: int, training_epochs: int, learning_rate: float, root: str | Path, allow_promotion: bool = True, _budget_scale: float = 1.0, _oom_retry: int = 0) -> HGTTrainingResult:
    try:
        torch, _, _ = _require_torch()
    except RuntimeError:
        examples = int(getattr(getattr(runtime, "graph", None), "memory_count", lambda: 0)())
        return HGTTrainingResult(epoch, "SKIPPED_DEPENDENCY", runtime.unified_telemetry.model_version, None, 0.0, 0.0, examples, 0, None)
    config = runtime.config.scientific
    matched_reasoning = (
        config.scientific_visibility_mode is ScientificVisibilityMode.MATCHED_REASONING
    )
    deterministic_seed = int(config.random_seeds[0]) + int(epoch) * 1_000_003
    if matched_reasoning:
        torch.manual_seed(deterministic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(deterministic_seed)
        torch.use_deterministic_algorithms(True)
    memory_node_budget = max(64, int(config.hgt_max_subgraph_nodes * float(_budget_scale)))
    canonical_edge_budget = max(256, int(config.hgt_max_subgraph_edges * float(_budget_scale)))
    total_node_budget = max(memory_node_budget, int(config.hgt_max_total_nodes * float(_budget_scale)))
    total_edge_budget = max(canonical_edge_budget, int(config.hgt_max_total_edges * float(_budget_scale)))
    if torch.cuda.is_available() and not matched_reasoning:
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        # Keep the configured graph budgets authoritative. Free VRAM is only a
        # pressure signal for shedding; it must never enlarge a graph because
        # forward/backward activation memory is not represented by mem_get_info().
        capacity_scale = 1.0
        runtime.set_telemetry_gauge("hgt_vram_capacity_scale", capacity_scale)
        runtime.set_telemetry_gauge("hgt_vram_free_before_training_bytes", int(free_bytes))
        runtime.set_telemetry_gauge("hgt_vram_total_bytes", int(total_bytes))
        if int(free_bytes) < int(config.hgt_min_free_vram_bytes):
            pressure = max(0.25, float(free_bytes) / max(1.0, float(config.hgt_min_free_vram_bytes)))
            memory_node_budget = max(64, int(memory_node_budget * pressure))
            canonical_edge_budget = max(256, int(canonical_edge_budget * pressure))
            total_node_budget = max(memory_node_budget, int(total_node_budget * pressure))
            total_edge_budget = max(canonical_edge_budget, int(total_edge_budget * pressure))
            runtime.set_telemetry_gauge("hgt_vram_preflight_shedding_factor", float(pressure))
    evidence_materializer = getattr(runtime, "training_evidence", None)
    evidence_branch = str(getattr(runtime, "_hgt_training_evidence_branch", ""))
    selected_evidence_records = ()
    if evidence_materializer is not None and evidence_branch:
        selected_evidence_records = tuple(
            sorted(
                (
                    record
                    for record in evidence_materializer.records()
                    if record.kind.value == "interaction"
                    and str(record.label_payload.get("sampling_branch", "")) == evidence_branch
                ),
                key=lambda record: (
                    str(record.label_payload.get("game_scenario", "")),
                    int(record.label_payload.get("actor_id", 0)),
                    int(record.label_payload.get("episode_id", 0)),
                    int(record.label_payload.get("global_step", 0)),
                    int(record.label_payload.get("producer_sequence", 0)),
                    record.evidence_id.value,
                ),
            )
        )
        epoch_transition_rows = transition_training_rows_from_records(
            (dict(record.label_payload) for record in selected_evidence_records),
            max_rows=int(config.hgt_transition_chunk_rows),
            active_episode_limit=int(config.hgt_active_episode_limit),
        )
        runtime.set_telemetry_gauge("hgt_training_evidence_authority", "canonical_wal")
    else:
        # Explicit legacy migration path for v9.7.8/v9.7.9 roots that predate
        # WAL-embedded TrainingEvidenceRecords.
        epoch_dataset_path = getattr(runtime, "_hgt_training_dataset_path", None)
        epoch_transition_rows = transition_training_rows(epoch_dataset_path) if epoch_dataset_path and Path(epoch_dataset_path).exists() else []
        runtime.set_telemetry_gauge("hgt_training_evidence_authority", "legacy_epoch_jsonl")
    epoch_ranking_pairs = action_ranking_pairs(epoch_transition_rows) if epoch_transition_rows else []
    transition_train_rows = list(epoch_transition_rows)
    runtime.set_telemetry_gauge("hgt_training_dataset_transitions", len(epoch_transition_rows))
    runtime.set_telemetry_gauge("hgt_action_ranking_pairs", len(epoch_ranking_pairs))
    training_view_builder = getattr(runtime.graph, "training_view", None)
    read_view = (
        training_view_builder(
            max_nodes=memory_node_budget,
            max_edges=canonical_edge_budget,
        )
        if callable(training_view_builder)
        else runtime.read_view
    )
    transition_train_rows = _contextualize_transition_rows(transition_train_rows, read_view)
    runtime.set_telemetry_gauge("hgt_contextual_transition_rows", len(transition_train_rows))
    if len(read_view.nodes) < 8:
        return HGTTrainingResult(epoch, "SKIPPED_INSUFFICIENT_DATA", runtime.unified_telemetry.model_version, None, 0.0, 0.0, len(read_view.nodes), 0, None)
    x_dict, edge_index_dict, y_dict, action_targets, action_masks, action_meta, task_targets, task_masks = build_hgt_graph(
        read_view,
        max_nodes=memory_node_budget,
        max_edges=canonical_edge_budget,
        max_total_nodes=total_node_budget,
        max_total_edges=total_edge_budget,
        max_semantic_facts_per_memory=int(config.hgt_max_semantic_facts_per_memory),
        include_objectives=True,
    )
    realized_nodes = sum(int(value.shape[0]) for value in x_dict.values())
    realized_edges = sum(int(value.shape[1]) for value in edge_index_dict.values())
    runtime.set_telemetry_gauge("hgt_requested_memory_nodes", int(memory_node_budget))
    runtime.set_telemetry_gauge("hgt_requested_canonical_edges", int(canonical_edge_budget))
    runtime.set_telemetry_gauge("hgt_total_node_budget", int(total_node_budget))
    runtime.set_telemetry_gauge("hgt_total_edge_budget", int(total_edge_budget))
    runtime.set_telemetry_gauge("hgt_realized_total_nodes", int(realized_nodes))
    runtime.set_telemetry_gauge("hgt_realized_total_edges", int(realized_edges))
    memory_nodes = sum(int(x_dict[node_type].shape[0]) for node_type in MEMORY_NODE_TYPES if node_type in x_dict)
    runtime.set_telemetry_gauge("hgt_semantic_nodes", int(realized_nodes - memory_nodes))
    for node_type in MEMORY_NODE_TYPES:
        runtime.set_telemetry_gauge(f"hgt_{node_type.lower()}_nodes", int(x_dict[node_type].shape[0]) if node_type in x_dict else 0)
    for level in range(8):
        runtime.set_telemetry_gauge(
            f"hgt_m{level}_nodes",
            sum(
                int(x_dict[node_type].shape[0])
                for node_type in MEMORY_NODE_TYPES
                if node_type.startswith(f"M{level}_") and node_type in x_dict
            ),
        )
    runtime.set_telemetry_gauge("hgt_oom_retry_count", int(_oom_retry))
    selected_examples = sum(int(v.numel()) for v in y_dict.values())
    action_examples = sum(int(mask.sum().item()) for mask in action_masks.values())
    action_opportunities = sum(int(mask.numel()) for mask in action_masks.values())
    metadata = _metadata_for_graph(x_dict, edge_index_dict)
    runtime.set_telemetry_gauge("hgt_metadata_node_types", len(metadata[0]))
    runtime.set_telemetry_gauge("hgt_metadata_edge_types", len(metadata[1]))
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
                environment_families[environment_id] = "unregistered"
                runtime.set_telemetry_gauge(
                    "hgt_unregistered_environment_count",
                    sum(1 for family in environment_families.values() if family == "unregistered"),
                )
    train_masks, validation_masks = _split_policy_masks(
        action_meta,
        action_masks,
        torch,
        validation_fraction=float(config.hgt_validation_fraction),
    )
    training_examples = _masked_count(train_masks, action_masks)
    validation_examples = _masked_count(validation_masks, action_masks)
    if training_examples <= 0:
        # Tiny cuts still train, but cannot claim held-out validation.
        train_masks = {key: mask.clone() for key, mask in action_masks.items()}
        validation_masks = {
            key: torch.zeros_like(mask, dtype=torch.bool)
            for key, mask in action_masks.items()
        }
        training_examples = _masked_count(train_masks, action_masks)
        validation_examples = 0
    if training_examples <= 0:
        return HGTTrainingResult(epoch, "SKIPPED_NO_TRAINING_EVIDENCE", runtime.unified_telemetry.model_version, None, 0.0, 0.0, action_examples, 0, None)

    training_ranking_pairs, validation_ranking_pairs = _split_ranking_pairs(
        epoch_ranking_pairs,
        validation_fraction=float(config.hgt_validation_fraction),
    )
    stream_batch_size = max(1, min(2048, int(config.hgt_epoch_batch_size)))
    ranking_batches = [
        training_ranking_pairs[index:index + stream_batch_size]
        for index in range(0, len(training_ranking_pairs), stream_batch_size)
    ]
    dynamic_training_steps = int(training_epochs) if matched_reasoning else max(
        int(training_epochs),
        min(64, max(1, int(math.ceil(training_examples / 2048.0)))),
        len(ranking_batches),
    )
    runtime.set_telemetry_gauge("hgt_training_batches", max(1, len(ranking_batches)))
    runtime.set_telemetry_gauge(
        "hgt_training_coverage_planned",
        float(training_examples) / max(1, action_examples),
    )
    runtime.set_telemetry_gauge("hgt_validation_examples", int(validation_examples))
    runtime.set_telemetry_gauge("hgt_training_ranking_pairs", len(training_ranking_pairs))
    runtime.set_telemetry_gauge("hgt_validation_ranking_pairs", len(validation_ranking_pairs))
    runtime.set_telemetry_gauge("hgt_dynamic_training_steps", int(dynamic_training_steps))
    runtime.set_telemetry_gauge("hgt_training_examples_current", int(training_examples))
    runtime.set_telemetry_gauge("hgt_examples_per_training_step_target", 2048)

    device = torch.device("cpu" if matched_reasoning else ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available() and not matched_reasoning:
        torch.cuda.reset_peak_memory_stats()
    model_dir = Path(root) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "hgt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    accepted_version = manifest.get("last_accepted_model_version") or manifest.get("accepted_model_version")
    # The active model is authoritative for continued training. During a
    # behavior-gate candidate epoch it may be newer than the last accepted
    # model, and its checkpoint fully describes the architecture to restore.
    parent_version = manifest.get("current_model_version") or accepted_version
    parent_checkpoint = manifest.get("current_checkpoint") if parent_version else None
    if parent_version and not parent_checkpoint:
        parent_checkpoint = f"models/{parent_version}.pt"
    if matched_reasoning:
        from v9.research.experiment_manifest import ExperimentManifest

        experiment = ExperimentManifest.load(runtime.config.experiment_manifest)
        evidence_checksum = (
            evidence_materializer.manifest.checksum
            if evidence_materializer is not None
            else "0" * 64
        )
        training_cut = TrainingCut(
            evidence_checksum,
            tuple(record.evidence_id.value for record in selected_evidence_records),
            (("model", deterministic_seed), ("replay", deterministic_seed + 1)),
            (("learning_rate", format(float(learning_rate), ".17g")),),
            int(dynamic_training_steps),
            int(stream_batch_size),
            1,
            1,
            int(MODEL_SCHEMA_VERSION),
            TrainingDeterminismMode.DETERMINISTIC_CPU,
            str(parent_version or "untrained"),
            experiment.manifest_id.value,
        )
        cut_directory = Path(root) / "hgt" / "training_cuts"
        cut_directory.mkdir(parents=True, exist_ok=True)
        cut_path = cut_directory / f"cut-{training_cut.checksum}.json"
        cut_payload = {
            key: (value.value if hasattr(value, "value") else value)
            for key, value in asdict(training_cut).items()
        }
        temporary_cut = cut_path.with_suffix(".tmp")
        with temporary_cut.open("w", encoding="utf-8") as handle:
            json.dump(cut_payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_cut, cut_path)
        runtime.set_telemetry_gauge("training_cut_checksum", training_cut.checksum)
        runtime.__dict__["_active_training_cut"] = training_cut
    checkpoint_state = None
    parent_architecture = None

    model = _HGTWrapper(
        metadata,
        input_dim=64,
        hidden_dim=int(config.hgt_hidden_dim),
        layers=int(config.hgt_layers),
        heads=int(config.hgt_heads),
    ).model.to(device)
    if parent_checkpoint:
        checkpoint_path = Path(root) / str(parent_checkpoint)
        if checkpoint_path.exists():
            # Inspect compatibility on CPU so a behavior-only legacy parent
            # cannot consume accelerator memory before being rejected for
            # tensor resumption.
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            try:
                current_architecture = {
                    "metadata": metadata,
                    "input_dim": 64,
                    "hidden_dim": int(config.hgt_hidden_dim),
                    "layers": int(config.hgt_layers),
                    "heads": int(config.hgt_heads),
                }
                checkpoint_state, parent_architecture = _load_compatible_training_parent(
                    loaded_checkpoint,
                    device=device,
                    model=model,
                    current_architecture=current_architecture,
                    runtime=runtime,
                )
            except (RuntimeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"accepted HGT checkpoint {checkpoint_path} is incompatible or corrupt: {exc}") from exc
    try:
        x_device = {key: value.to(device) for key, value in x_dict.items()}
        edges_device = {key: value.to(device) for key, value in edge_index_dict.items()}
    except RuntimeError as exc:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _retry_after_oom(
            runtime,
            epoch=epoch,
            training_epochs=training_epochs,
            learning_rate=learning_rate,
            root=root,
            allow_promotion=allow_promotion,
            budget_scale=_budget_scale,
            oom_retry=_oom_retry,
            exc=exc,
        )


    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    if checkpoint_state is not None and checkpoint_state.get("optimizer_state"):
        try:
            optimizer.load_state_dict(checkpoint_state["optimizer_state"])
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"accepted HGT optimizer state is incompatible or corrupt: {exc}") from exc
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    start = time.perf_counter()
    training_loss = 0.0
    training_steps = 0
    last_grad_norm = 0.0
    model.train()
    try:
        ranking_correct = ranking_total = ranking_pairs_trained = 0
        loop_count = max(1, int(dynamic_training_steps))
        for step_index in range(loop_count):
            optimizer.zero_grad(set_to_none=True)
            logits, values, auxiliary = model(x_device, edges_device)
            loss, consequence_accuracy, _ = _loss(
                logits,
                values,
                auxiliary,
                y_dict,
                train_masks,
                action_targets,
                action_masks,
                task_targets,
                task_masks,
                torch,
                objective_weights=config.hgt_loss_weights,
                log_vars=model.objective_log_vars,
                dynamic_weighting=bool(config.hgt_dynamic_loss_weighting),
            )
            ranking_batch = (
                ranking_batches[step_index % len(ranking_batches)]
                if ranking_batches
                else ()
            )
            ranking_loss, step_correct, step_total = _explicit_action_ranking_loss(
                values,
                action_meta,
                train_masks,
                ranking_batch,
                torch,
            )
            if ranking_loss is not None:
                weighted_ranking = float(config.hgt_ranking_loss_weight) * ranking_loss
                loss = weighted_ranking if loss is None else loss + weighted_ranking
                ranking_correct += int(step_correct)
                ranking_total += int(step_total)
                ranking_pairs_trained += int(step_total)
            if loss is None:
                break
            loss.backward()
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item())
            optimizer.step()
            training_loss = float(loss.detach().cpu().item())
            training_steps += 1
    except RuntimeError as exc:
        loss = logits = values = auxiliary = ranking_loss = None
        ranking_batch = ()
        del model, optimizer, x_device, edges_device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _retry_after_oom(
            runtime,
            epoch=epoch,
            training_epochs=training_epochs,
            learning_rate=learning_rate,
            root=root,
            allow_promotion=allow_promotion,
            budget_scale=_budget_scale,
            oom_retry=_oom_retry,
            exc=exc,
        )
    coverage = float(training_examples) / max(1, action_examples)
    train_ranking_accuracy = float(ranking_correct) / max(1, ranking_total)
    runtime.set_telemetry_gauge("hgt_training_coverage", coverage)
    runtime.set_telemetry_gauge("hgt_action_ranking_accuracy", train_ranking_accuracy)
    runtime.set_telemetry_gauge("hgt_action_ranking_training_pairs", int(ranking_pairs_trained))
    if training_steps <= 0:
        return HGTTrainingResult(epoch, "SKIPPED_NO_TRAINING_EVIDENCE", runtime.unified_telemetry.model_version, parent_version, 0.0, 0.0, action_examples, 0, parent_checkpoint)
    if matched_reasoning and training_steps != int(training_cut.optimizer_step_count):
        runtime.set_telemetry_gauge("training_cut_failure", "INCOMPLETE_OPTIMIZER_WORK")
        return HGTTrainingResult(
            epoch,
            "FAILED_TRAINING_CUT_INCOMPLETE",
            runtime.unified_telemetry.model_version,
            parent_version,
            training_loss,
            0.0,
            action_examples,
            training_steps,
            parent_checkpoint,
        )

    runtime.set_telemetry_gauge("hgt_transition_training_examples", len(epoch_transition_rows))
    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        logits, values, auxiliary = model(x_device, edges_device)
        base_train_loss_t, consequence_accuracy, training_loss_by_head = _loss(
            logits,
            values,
            auxiliary,
            y_dict,
            train_masks,
            action_targets,
            action_masks,
            task_targets,
            task_masks,
            torch,
            objective_weights=config.hgt_loss_weights,
            log_vars=model.objective_log_vars,
            dynamic_weighting=bool(config.hgt_dynamic_loss_weighting),
        )
        train_ranking_loss_t, train_rank_correct, train_rank_total = _explicit_action_ranking_loss(
            values,
            action_meta,
            train_masks,
            training_ranking_pairs,
            torch,
        )
        validation_policy_loss_t = _policy_validation_loss(
            logits,
            values,
            y_dict,
            action_targets,
            validation_masks,
            torch,
        )
        validation_ranking_loss_t, validation_rank_correct, validation_rank_total = _explicit_action_ranking_loss(
            values,
            action_meta,
            validation_masks,
            validation_ranking_pairs,
            torch,
        )

    total_train_loss_t = base_train_loss_t
    if train_ranking_loss_t is not None:
        weighted = float(config.hgt_ranking_loss_weight) * train_ranking_loss_t
        total_train_loss_t = weighted if total_train_loss_t is None else total_train_loss_t + weighted
    validation_terms = [
        value for value in (validation_policy_loss_t, validation_ranking_loss_t)
        if value is not None
    ]
    validation_loss = (
        float(torch.stack(validation_terms).mean().cpu().item())
        if validation_terms
        else 0.0
    )
    validation_accuracy = (
        float(validation_rank_correct) / max(1, int(validation_rank_total))
        if validation_rank_total
        else 0.0
    )
    train_accuracy = (
        float(train_rank_correct) / max(1, int(train_rank_total))
        if train_rank_total
        else float(consequence_accuracy)
    )
    training_loss_by_head = dict(training_loss_by_head)
    if train_ranking_loss_t is not None:
        training_loss_by_head["action_ranking"] = float(
            train_ranking_loss_t.detach().cpu().item()
        )
    runtime.set_telemetry_gauge("hgt_validation_loss", validation_loss)
    runtime.set_telemetry_gauge("hgt_validation_ranking_accuracy", validation_accuracy)
    runtime.set_telemetry_gauge("hgt_validation_ranking_pairs_used", int(validation_rank_total))
    runtime.set_telemetry_gauge("hgt_consequence_accuracy", float(consequence_accuracy))
    inference_latency_ms = 1000.0 * (time.perf_counter() - inference_started)
    training_loss = float(total_train_loss_t.cpu().item()) if total_train_loss_t is not None else training_loss
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
            environment_id, context_signature, action_id, _, _, _game = row
            score = float(predicted[index].item())
            action_score_sums.setdefault(environment_id, {}).setdefault(action_id, []).append(score)
            context_score_sums.setdefault(environment_id, {}).setdefault(context_signature, {}).setdefault(action_id, []).append(score)
    raw_action_scores = {
        environment_id: {action_id: sum(scores) / len(scores) for action_id, scores in actions.items() if scores}
        for environment_id, actions in action_score_sums.items()
    }
    raw_context_action_scores = {
        environment_id: {
            context: {action_id: sum(scores) / len(scores) for action_id, scores in actions.items() if scores}
            for context, actions in contexts.items()
        }
        for environment_id, contexts in context_score_sums.items()
    }
    policy_score_scale = _policy_score_scale(
        config,
        validation_accuracy=validation_accuracy,
        validation_pairs=int(validation_rank_total),
    )
    action_scores = {
        environment_id: _bounded_advantage_scores(scores, policy_score_scale)
        for environment_id, scores in raw_action_scores.items()
    }
    context_action_scores = {
        environment_id: {
            context: _bounded_advantage_scores(scores, policy_score_scale)
            for context, scores in contexts.items()
        }
        for environment_id, contexts in raw_context_action_scores.items()
    }
    runtime.set_telemetry_gauge("hgt_policy_score_scale", float(policy_score_scale))
    runtime.set_telemetry_gauge(
        "hgt_policy_gate_open",
        int(policy_score_scale > 0.0),
    )
    # Every newly trained version becomes the candidate used by the next epoch.
    # Its scientific evaluation is the next epoch's matched HGT-ON/OFF sampling.
    promote = True
    status = "TESTING_PENDING_BEHAVIOR"
    storage_governor = getattr(runtime, "storage_governor", None)
    if storage_governor is not None:
        storage_governor.reconcile_filesystem()
        if not storage_governor.status().checkpoint_creation_allowed:
            runtime.set_telemetry_gauge("hgt_promotion_result", "STORAGE_PRESSURE_DEFERRED")
            return HGTTrainingResult(
                epoch,
                "STORAGE_PRESSURE_DEFERRED",
                runtime.unified_telemetry.model_version,
                parent_version,
                training_loss,
                0.0,
                action_examples,
                training_steps,
                parent_checkpoint,
            )
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
                "training_loss": training_loss,
                "training_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "policy_score_scale": policy_score_scale,
                "loss_by_head": dict(training_loss_by_head),
                "objective_weights": list(config.hgt_loss_weights),
                "objective_names": list(OBJECTIVE_NAMES),
                "symbol_node_type": "SYMBOL",
                "symbol_schema_version": 2,
                "dynamic_loss_weighting": bool(config.hgt_dynamic_loss_weighting),
                "action_scores": action_scores,
                "context_action_scores": context_action_scores,
                "training_cut_checksum": (
                    training_cut.checksum if matched_reasoning else None
                ),
                "scientific_config_id": config.config_id.value,
                "visibility_mode": config.scientific_visibility_mode.value,
                "learned_developmental_feedback": config.learned_developmental_feedback.value,
                "experiment_manifest_id": (
                    experiment.manifest_id.value if matched_reasoning else None
                ),
            },
            temporary_checkpoint,
        )
        with temporary_checkpoint.open("rb") as checkpoint_handle:
            os.fsync(checkpoint_handle.fileno())
        os.replace(temporary_checkpoint, checkpoint_path)
        _write_architecture_sidecar(
            checkpoint_path,
            {
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "metadata": metadata,
                "input_dim": 64,
                "hidden_dim": int(config.hgt_hidden_dim),
                "layers": int(config.hgt_layers),
                "heads": int(config.hgt_heads),
            },
        )
        manifest = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "version_index": version_index,
            "current_model_version": candidate_version,
            "current_checkpoint": checkpoint_rel,
            "accepted_model_version": accepted_version,
            "accepted_checkpoint": parent_checkpoint,
            "candidate_model_version": candidate_version,
            "candidate_status": "TESTING_PENDING_BEHAVIOR",
            "parent_model_version": parent_version,
            "training_loss": training_loss,
            "training_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "policy_score_scale": policy_score_scale,
            "loss_by_head": dict(training_loss_by_head),
            "objective_weights": list(config.hgt_loss_weights),
            "dynamic_loss_weighting": bool(config.hgt_dynamic_loss_weighting),
            "graph_generation": int(read_view.generation),
            "selected_nodes": selected_examples,
            "examples": action_examples,
            "epoch_dataset_transitions": len(epoch_transition_rows),
            "action_ranking_pairs": len(epoch_ranking_pairs),
            "training_coverage": float(coverage),
            "training_examples": training_examples,
            "action_scores": action_scores,
            "context_action_scores": context_action_scores,
            "training_cut_checksum": (
                training_cut.checksum if matched_reasoning else None
            ),
            "scientific_config_id": config.config_id.value,
            "visibility_mode": config.scientific_visibility_mode.value,
            "learned_developmental_feedback": config.learned_developmental_feedback.value,
            "experiment_manifest_id": (
                experiment.manifest_id.value if matched_reasoning else None
            ),
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        if storage_governor is not None:
            storage_governor.reconcile_filesystem()
        model_version = candidate_version
        runtime.set_hgt_action_scores(action_scores, context_action_scores=context_action_scores)
    else:
        model_version = str(parent_version)
    gpu = read_gpu_snapshot()
    sample = HGTTrainingSample(
        training_loss=training_loss,
        validation_loss=float(validation_loss),
        training_step_latency_ms=1000.0 * elapsed / max(1, training_steps),
        training_examples_seen=int(training_examples),
        effective_batch_size=int(stream_batch_size),
        gradient_norm=last_grad_norm,
        learning_rate=float(learning_rate),
        training_steps=training_steps,
        examples_per_second=float(training_examples) / elapsed,
        gpu_memory_bytes=int(gpu.memory_used_bytes),
        gpu_utilization=float(gpu.utilization_percent),
        historical_retention=0.0,
        current_curriculum_gain=max(0.0, train_accuracy - (1.0 / 3.0)),
        cross_family_validation_gain=0.0,
        loss_by_head=dict(training_loss_by_head),
    )
    runtime.record_hgt_training(sample)
    retention_delta = 0.0
    runtime.record_model_evolution(
        ModelEvolutionSample(
            model_version=model_version,
            parent_model_version=parent_version,
            training_examples_since_parent=training_examples,
            current_stage_delta=max(0.0, train_accuracy - (1.0 / 3.0)),
            historical_retention_delta=0.0,
            cross_family_transfer_delta=0.0,
            reasoning_improvement_delta=float(train_accuracy),
            inference_latency_delta_ms=float(inference_latency_ms - float(runtime.unified_telemetry.diagnostic_metrics().get("inference_latency_ms", 0.0))),
            promotion_result=status,
        )
    )
    result = HGTTrainingResult(
        epoch=epoch,
        status=status,
        model_version=model_version,
        parent_model_version=parent_version,
        training_loss=training_loss,
        validation_loss=0.0,
        examples=action_examples,
        training_steps=training_steps,
        checkpoint=checkpoint_rel if promote else parent_checkpoint,
        validation_accuracy=float(validation_accuracy),
        training_accuracy=float(train_accuracy),
        inference_latency_ms=float(inference_latency_ms),
        subgraph_nodes=sum(int(value.shape[0]) for value in x_dict.values()),
        subgraph_edges=sum(int(value.shape[1]) for value in edge_index_dict.values()),
        relevance_precision=max(0.0, min(1.0, action_examples / max(1, action_opportunities))),
    )
    if torch.cuda.is_available():
        runtime.set_telemetry_gauge("hgt_peak_allocated_bytes", int(torch.cuda.max_memory_allocated()))
        runtime.set_telemetry_gauge("hgt_peak_reserved_bytes", int(torch.cuda.max_memory_reserved()))
        del logits, values, auxiliary, x_device, edges_device, optimizer, model
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info()
        runtime.set_telemetry_gauge("hgt_vram_free_after_training_bytes", int(free_after))
    return result
