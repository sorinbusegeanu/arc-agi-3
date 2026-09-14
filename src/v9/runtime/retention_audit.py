from __future__ import annotations

import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from v9.hgt.training import _node_feature
from v9.memory.identity import stable_u64
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
from v9.runtime.multiprocess import EncodedTransition


def _decode_symbols(value: Any) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.encode("utf-8"))
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return tuple(text.encode("utf-8"))
    return ()


def _identity_tuple(row: dict[str, Any]) -> tuple[str, str, str, str]:
    identity = dict(row.get("identity") or {})
    return (
        str(identity.get("family", row.get("adapter", "unknown"))),
        str(identity.get("environment_type", row.get("game", "unknown"))),
        str(identity.get("config", "trace")),
        str(identity.get("instance", f"trace:{row.get('game','unknown')}")),
    )


def _schema_id(row: dict[str, Any], key: str, fallback: int) -> int:
    value = row.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _transition_from_trace(row: dict[str, Any], actor_id: int, sequence: int) -> EncodedTransition:
    before_actions = tuple(int(item["token"]) for item in row.get("available_actions_before", ()) if isinstance(item, dict) and "token" in item)
    after_actions = tuple(int(item["token"]) for item in row.get("available_actions_after", ()) if isinstance(item, dict) and "token" in item)
    chosen = dict(row.get("chosen_action") or {})
    boundary = dict(row.get("boundary") or {})
    progress = dict(row.get("task_progress") or {})
    action_schema_id = _schema_id(row, "action_schema_id", 0)
    observation_schema_id = _schema_id(row, "observation_schema_id", 0)
    option_signature = stable_u64(action_schema_id, *tuple(sorted(set(after_actions))), person=b"v9-action-set")
    scope = str(boundary.get("scope", "NONE"))
    if "." in scope:
        scope = scope.rsplit(".", 1)[-1]
    return EncodedTransition(
        actor_id=int(actor_id),
        producer_sequence=int(sequence),
        global_step=int(row.get("step", sequence - 1)),
        environment_identity=_identity_tuple(row),
        episode_id=int(row.get("episode", 1)),
        observation_schema_id=observation_schema_id,
        before_signature=int(row.get("before_signature", 0)),
        action_id=int(chosen.get("token", 0)),
        after_signature=int(row.get("after_signature", 0)),
        available_actions_after=len(after_actions),
        primary_valence=int(boundary.get("primary_valence", 0) or 0),
        symbols=_decode_symbols(row.get("symbols_after")),
        curriculum_step=None,
        game_scenario=str(row.get("game", "unknown")),
        symbols_only=False,
        action_schema_id=action_schema_id,
        available_action_set_signature=int(option_signature),
        boundary_scope=scope,
        task_success=bool(progress.get("success", False)),
        task_failure=bool(progress.get("failure", False)),
        task_truncated=bool(progress.get("truncated", False)),
        level_index=int(progress.get("level_index", 0) or 0),
        levels_completed=int(progress.get("levels_completed", 0) or 0),
    )


def _m0_payload(prepared: Any) -> dict[str, Any]:
    m0 = prepared.m0
    if m0 is None:
        return {}
    return {
        "modality_id": int(m0.modality_id),
        "environment_instance_id": int(m0.provenance.environment_instance_id),
        "episode_id": int(m0.provenance.episode_id.value),
        "context_signature": int(m0.context_signature),
        "payload_digest": int(m0.payload_digest),
        "action_id": m0.action_id,
        "outcome_signature": m0.outcome_signature,
        "next_context_signature": m0.next_context_signature,
        "symbol_identity": m0.symbol_identity,
        "primary_valence": int(m0.primary_valence),
        "future_option_delta": float(m0.future_option_delta),
        "realized_cost": int(m0.realized_cost),
    }


def _hgt_feature(prepared: Any) -> list[float]:
    m0 = prepared.m0
    if m0 is None:
        return []
    payload = _m0_payload(prepared)
    node = CanonicalNode(
        m0.uid,
        MemoryLevel.M0,
        MemoryType.EPISODE,
        (int(m0.uid.hi), int(m0.uid.lo)),
        int(prepared.event.identity.causal_watermark if prepared.event is not None else 0),
    )
    try:
        import torch
        tensor = _node_feature(node, payload, 64, torch)
        return [float(v) for v in tensor.tolist()]
    except ImportError:
        class _Tensor(list):
            def tolist(self):
                return list(self)
        class _TorchShim:
            float32 = float
            @staticmethod
            def tensor(values, dtype=None):
                return _Tensor(float(value) for value in values)
        tensor = _node_feature(node, payload, 64, _TorchShim)
        return [float(v) for v in tensor.tolist()]


def _classification(row: dict[str, Any], prepared: Any) -> dict[str, dict[str, str]]:
    has_text = bool(row.get("symbols_after"))
    before_observation = row.get("before_observation")
    has_structured_observation = isinstance(before_observation, (dict, list))
    semantic_actions = any(
        isinstance(item, dict) and str(item.get("semantic", "")) != str(item.get("token", ""))
        for item in row.get("available_actions_before", ())
    )
    return {
        "raw_observation": {
            "encoded_transition": "HASHED_ONLY",
            "m0_m1": "HASHED_ONLY",
            "hgt": "HASH_DERIVED",
        },
        "spatial_structure": {
            "encoded_transition": "HASHED_ONLY" if has_structured_observation else "NOT_APPLICABLE",
            "m0_m1": "HASHED_ONLY" if has_structured_observation else "NOT_APPLICABLE",
            "hgt": "HASH_DERIVED" if has_structured_observation else "NOT_APPLICABLE",
        },
        "text_content": {
            "encoded_transition": "PRESERVED_AS_SYMBOL_STREAM" if has_text else "ABSENT",
            "m0_m1": "PRESERVED_AS_SYMBOL_EVENTS" if has_text and prepared.symbols else ("ABSENT" if not has_text else "LOST"),
            "hgt": "INDIRECT_GRAPH_ONLY" if has_text and prepared.symbols else "ABSENT",
        },
        "action_identity": {
            "encoded_transition": "PRESERVED_TOKEN",
            "m0_m1": "PRESERVED_TOKEN",
            "hgt": "PRESERVED_NUMERIC_TOKEN",
        },
        "action_semantics": {
            "encoded_transition": "LOST" if semantic_actions else "NOT_APPLICABLE",
            "m0_m1": "LOST" if semantic_actions else "NOT_APPLICABLE",
            "hgt": "LOST" if semantic_actions else "NOT_APPLICABLE",
        },
        "available_action_set": {
            "encoded_transition": "COUNT_AND_HASH_ONLY",
            "m0_m1": "COUNT_AND_HASH_ONLY",
            "hgt": "INDIRECT_ONLY",
        },
        "reward_boundary": {
            "encoded_transition": "PRESERVED",
            "m0_m1": "PRESERVED",
            "hgt": "PRESERVED_NUMERIC",
        },
        "level_game_progress": {
            "encoded_transition": "PRESERVED",
            "m0_m1": "PRESERVED_IN_NORMALIZED_RELATION",
            "hgt": "INDIRECT_GRAPH_ONLY",
        },
    }


def _retention_score(classification: dict[str, dict[str, str]]) -> float:
    weights = {
        "PRESERVED": 1.0,
        "PRESERVED_TOKEN": 1.0,
        "PRESERVED_NUMERIC_TOKEN": 0.9,
        "PRESERVED_NUMERIC": 0.9,
        "PRESERVED_AS_SYMBOL_STREAM": 0.9,
        "PRESERVED_AS_SYMBOL_EVENTS": 0.8,
        "PRESERVED_IN_NORMALIZED_RELATION": 0.8,
        "COUNT_AND_HASH_ONLY": 0.35,
        "INDIRECT_GRAPH_ONLY": 0.35,
        "INDIRECT_ONLY": 0.25,
        "HASHED_ONLY": 0.2,
        "HASH_DERIVED": 0.1,
        "LOST": 0.0,
        "ABSENT": math.nan,
        "NOT_APPLICABLE": math.nan,
    }
    values: list[float] = []
    for stages in classification.values():
        value = weights.get(stages["hgt"], 0.0)
        if not math.isnan(value):
            values.append(value)
    return sum(values) / max(1, len(values))


def run_retention_audit(trace_bundle: str | Path, *, root: str | Path) -> Path:
    source = Path(trace_bundle)
    if not source.exists():
        raise FileNotFoundError(source)
    target_root = Path(root) / "retention"
    target_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    with zipfile.ZipFile(source, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for game_index, game in enumerate(manifest.get("games", ())):
            trace_name = str(game["file"])
            lines = archive.read(trace_name).decode("utf-8").splitlines()
            output_rows: list[dict[str, Any]] = []
            scores: list[float] = []
            field_counts: dict[str, dict[str, int]] = {}
            for sequence, line in enumerate(lines, start=1):
                row = json.loads(line)
                if row.get("event") == "trace_error":
                    continue
                transition = _transition_from_trace(row, game_index + 1, sequence)
                prepared = prepare_ingestion(
                    IngestionTask(sequence, sequence, transition)
                )
                classification = _classification(row, prepared)
                score = _retention_score(classification)
                scores.append(score)
                for field, stages in classification.items():
                    stage = stages["hgt"]
                    field_counts.setdefault(field, {})[stage] = field_counts.setdefault(field, {}).get(stage, 0) + 1
                output_rows.append({
                    "game": row.get("game"),
                    "episode": row.get("episode"),
                    "step": row.get("step"),
                    "raw": {
                        "before_observation": row.get("before_observation"),
                        "symbols": row.get("symbols_after"),
                        "available_actions": row.get("available_actions_before"),
                        "chosen_action": row.get("chosen_action"),
                        "boundary": row.get("boundary"),
                        "task_progress": row.get("task_progress"),
                    },
                    "encoded_transition": asdict(transition),
                    "m0": None if prepared.m0 is None else asdict(prepared.m0),
                    "m1_grounded": None if prepared.m1g is None else asdict(prepared.m1g),
                    "m1_normalized": None if prepared.m1n is None else asdict(prepared.m1n),
                    "symbol_events": [
                        {
                            "m0": asdict(symbol.m0),
                            "m1_grounded": asdict(symbol.m1g),
                            "m1_normalized": asdict(symbol.m1n),
                            "aligned_m1_normalized": None if symbol.aligned_m1n is None else asdict(symbol.aligned_m1n),
                        }
                        for symbol in prepared.symbols
                    ],
                    "hgt_feature_64": _hgt_feature(prepared),
                    "retention": classification,
                    "hgt_retention_score": score,
                })

            name = Path(trace_name).with_suffix(".retention.jsonl").name
            path = target_root / name
            with path.open("w", encoding="utf-8") as handle:
                for output in output_rows:
                    handle.write(json.dumps(output, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            summaries.append({
                "game": game.get("game"),
                "source_file": trace_name,
                "report_file": name,
                "steps": len(output_rows),
                "mean_hgt_retention_score": sum(scores) / max(1, len(scores)),
                "field_hgt_classifications": field_counts,
            })

    summary = {
        "schema_version": 1,
        "source_trace_bundle": source.name,
        "games": summaries,
        "overall_mean_hgt_retention_score": (
            sum(float(row["mean_hgt_retention_score"]) for row in summaries) / max(1, len(summaries))
        ),
    }
    summary_path = target_root / "retention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    bundle = target_root / "retention_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(summary_path, summary_path.name)
        for row in summaries:
            path = target_root / str(row["report_file"])
            archive.write(path, path.name)
    return bundle
