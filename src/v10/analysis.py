"""Prefix-bounded multimodal analysis and recorded-continuation scoring."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .frames import DELTA_KIND, FrameStore, IMAGE_KIND
from .runlog import append_jsonl, read_json
from .tools.game import (
    ActionError, PREDICTION_OPERATORS, PREDICTION_PATHS,
    evaluate_prediction, validate_decision,
)
from .tools.history import History
from .tools.memory import (
    DEFAULT_CONTEXT_ENTRIES, ENTRY_TYPES, MODEL_PATCH_FIELDS, MODEL_PATCH_OPS, GameMemory,
    MemoryValidationError, default_state_dir, game_memory_dir,
    validate_model_patches,
)


DECISION_SCHEMA = "v10.decision/1"
ROOT_FIELDS = {
    "observations", "patterns", "hypotheses", "prediction_audit", "goals",
    "priority_question", "decision", "memory_patch",
}
class DecisionValidationError(ValueError):
    """Model output failed the v10 decision contract."""


def _require_object(value: Any, name: str, required: set[str], allowed: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DecisionValidationError(f"{name} must be an object")
    missing = sorted(required - set(value))
    if missing:
        raise DecisionValidationError(f"{name} missing required field: {missing[0]}")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise DecisionValidationError(f"{name} has unknown field: {unknown[0]}")
    return value


def _require_string(value: Any, name: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise DecisionValidationError(f"{name} must be a non-empty string")
    return value


def _require_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DecisionValidationError(f"{name} must be a list of strings")
    return value


def _check_evidence_ids(value: Any, name: str, known: set[str]) -> None:
    for evidence_id in _require_string_list(value, name):
        if evidence_id not in known:
            if evidence_id.startswith("F"):
                raise DecisionValidationError(
                    f"{name} cites frame handle {evidence_id}; cite an evidence ID instead. "
                    f"Available evidence IDs: {', '.join(sorted(known))}"
                )
            raise DecisionValidationError(
                f"{name} references unavailable evidence: {evidence_id}. "
                f"Available evidence IDs: {', '.join(sorted(known))}"
            )


def validate_model_decision(
    payload: Any,
    legal_action_specs: Sequence[Mapping[str, Any]],
    visible_evidence_ids: set[str],
    visible_prediction_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Strictly validate the complete decision and reject future evidence IDs."""
    decision = _require_object(payload, "decision", ROOT_FIELDS, ROOT_FIELDS)
    list_fields = ("observations", "patterns", "hypotheses", "prediction_audit", "goals", "memory_patch")
    for field in list_fields:
        if not isinstance(decision[field], list):
            raise DecisionValidationError(f"decision.{field} must be a list")

    for index, item in enumerate(decision["observations"]):
        name = f"observations[{index}]"
        _require_object(item, name, {"id", "claim", "evidence_ids"}, {"id", "claim", "evidence_ids"})
        _require_string(item["id"], f"{name}.id")
        _require_string(item["claim"], f"{name}.claim")
        _check_evidence_ids(item["evidence_ids"], f"{name}.evidence_ids", visible_evidence_ids)

    for index, item in enumerate(decision["patterns"]):
        name = f"patterns[{index}]"
        _require_object(item, name, {"id", "relation", "conditions", "evidence_ids", "exceptions", "status"},
                        {"id", "relation", "conditions", "evidence_ids", "exceptions", "status"})
        for field in ("id", "relation", "conditions", "status"):
            _require_string(item[field], f"{name}.{field}")
        if item["status"] not in {"candidate", "supported", "contradicted"}:
            raise DecisionValidationError(f"{name}.status is invalid")
        _require_string_list(item["exceptions"], f"{name}.exceptions")
        _check_evidence_ids(item["evidence_ids"], f"{name}.evidence_ids", visible_evidence_ids)

    for index, item in enumerate(decision["hypotheses"]):
        name = f"hypotheses[{index}]"
        _require_object(item, name, {"id", "claim", "alternatives", "prediction", "falsification", "status"},
                        {"id", "claim", "alternatives", "prediction", "falsification", "status", "evidence_ids"})
        for field in ("id", "claim", "prediction", "falsification", "status"):
            _require_string(item[field], f"{name}.{field}")
        if item["status"] not in {"candidate", "supported", "rejected"}:
            raise DecisionValidationError(f"{name}.status is invalid")
        _require_string_list(item["alternatives"], f"{name}.alternatives")
        if "evidence_ids" in item:
            _check_evidence_ids(item["evidence_ids"], f"{name}.evidence_ids", visible_evidence_ids)

    prior_predictions = visible_prediction_ids or set()
    for index, item in enumerate(decision["prediction_audit"]):
        name = f"prediction_audit[{index}]"
        _require_object(item, name, {"prediction_id", "verdict", "evidence_ids"},
                        {"prediction_id", "verdict", "evidence_ids"})
        prediction_id = _require_string(item["prediction_id"], f"{name}.prediction_id")
        if prediction_id not in prior_predictions:
            raise DecisionValidationError(f"{name} references unavailable prediction: {prediction_id}")
        if item["verdict"] not in {"matched", "failed", "inconclusive"}:
            raise DecisionValidationError(f"{name}.verdict is invalid")
        _check_evidence_ids(item["evidence_ids"], f"{name}.evidence_ids", visible_evidence_ids)

    for index, item in enumerate(decision["goals"]):
        name = f"goals[{index}]"
        _require_object(item, name, {"id", "claim", "status", "evidence_ids"},
                        {"id", "claim", "status", "evidence_ids"})
        for field in ("id", "claim", "status"):
            _require_string(item[field], f"{name}.{field}")
        if item["status"] not in {"unverified", "supported", "rejected"}:
            raise DecisionValidationError(f"{name}.status is invalid")
        _check_evidence_ids(item["evidence_ids"], f"{name}.evidence_ids", visible_evidence_ids)

    _require_string(decision["priority_question"], "decision.priority_question")
    plan = _require_object(decision["decision"], "decision.decision",
                           {"mode", "continue_or_reset", "reset_justification", "actions"},
                           {"mode", "continue_or_reset", "reset_justification", "actions"})
    if not isinstance(plan["mode"], str):
        raise DecisionValidationError("decision.decision.mode must be a string")
    if not isinstance(plan["continue_or_reset"], str):
        raise DecisionValidationError("decision.decision.continue_or_reset must be a string")
    if plan["reset_justification"] is not None and not isinstance(plan["reset_justification"], str):
        raise DecisionValidationError("decision.decision.reset_justification must be a string or null")
    actions = plan["actions"]
    if not isinstance(actions, list) or len(actions) > 4:
        raise DecisionValidationError("decision.decision.actions must contain 0 to 4 actions")
    if plan["continue_or_reset"] == "reset" and actions:
        raise DecisionValidationError("a reset decision must not also propose actions")
    try:
        decision = validate_decision(decision, legal_action_specs, max_actions=4)
    except ActionError as exc:
        raise DecisionValidationError(str(exc)) from exc
    prediction_ids: set[str] = set()
    for index, item in enumerate(actions):
        allowed = {
            "action", "args", "prediction_id", "expected_observation", "stop_if",
            "measurable_prediction", "unmeasurable_reason",
        }
        _require_object(item, f"actions[{index}]",
                        {"action", "args", "prediction_id", "expected_observation", "stop_if", "measurable_prediction"},
                        allowed)
        prediction_id = _require_string(item["prediction_id"], f"actions[{index}].prediction_id")
        if prediction_id in prediction_ids:
            raise DecisionValidationError(f"actions[{index}].prediction_id is duplicated")
        prediction_ids.add(prediction_id)
        _require_string(item["expected_observation"], f"actions[{index}].expected_observation")
        _require_string_list(item["stop_if"], f"actions[{index}].stop_if")
        if item["measurable_prediction"] is None:
            _require_string(item.get("unmeasurable_reason"), f"actions[{index}].unmeasurable_reason")

    for index, patch in enumerate(decision["memory_patch"]):
        if not isinstance(patch, dict):
            raise DecisionValidationError(f"memory_patch[{index}] must be an object")
        if "evidence_ids" in patch:
            _check_evidence_ids(patch["evidence_ids"], f"memory_patch[{index}].evidence_ids", visible_evidence_ids)
    try:
        validate_model_patches(decision["memory_patch"], visible_evidence_ids)
    except MemoryValidationError as exc:
        raise DecisionValidationError(str(exc)) from exc
    return decision


def _action_specs(transitions: Sequence[dict[str, Any]], current: dict[str, Any]) -> list[dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for transition in transitions:
        action_id = transition.get("action")
        args = transition.get("args", {})
        if not isinstance(action_id, str) or not isinstance(args, dict) or not args:
            continue
        properties: dict[str, Any] = {}
        for key, value in args.items():
            if key in {"x", "y"} and isinstance(value, int):
                height, width = transition.get("frame_shape", [1, 1])
                bound = width - 1 if key == "x" else height - 1
                properties[key] = {"type": "integer", "minimum": 0, "maximum": max(0, int(bound))}
            elif isinstance(value, bool):
                properties[key] = {"type": "boolean"}
            elif isinstance(value, int):
                properties[key] = {"type": "integer"}
            elif isinstance(value, float):
                properties[key] = {"type": "number"}
            elif isinstance(value, str):
                properties[key] = {"type": "string"}
        observed[action_id] = {
            "id": action_id,
            "parameters": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
        }
    specs = []
    for action_id in current.get("legal_actions", []):
        specs.append(observed.get(action_id, {
            "id": action_id,
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }))
    return specs


def build_analysis_input(
    run_dir: str | Path,
    through_action: int = 4,
    skill_path: str | Path | None = None,
    *,
    conditioning: str = "memory_delta",
    horizon: int = 4,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
    include_prior_decisions: bool = False,
) -> dict[str, Any]:
    """Build model context using only the requested replay prefix."""
    if int(through_action) < 1:
        raise ValueError("through_action must be positive")
    if not 1 <= int(horizon) <= 4:
        raise ValueError("horizon must be between 1 and 4")
    if conditioning not in {"pixels_actions", "pixels_actions_memory", "memory_delta"}:
        raise ValueError("unknown conditioning profile")
    if int(memory_context_entries) < 0:
        raise ValueError("memory_context_entries must be non-negative")
    run_path = Path(run_dir)
    manifest = read_json(run_path / "replay_manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError(f"missing replay manifest in {run_path}")
    visible_records = History.read_prefix(run_path / "evidence.jsonl", int(through_action))
    prefix = [row for row in visible_records if row.get("type") == "transition"]
    last = prefix[-1]
    visible_step = int(last["step"])
    first_state = next((row for row in visible_records if row.get("type") == "state"), None)
    initial_evidence_id = first_state.get("evidence_id") if first_state else None
    visible_ids = {str(initial_evidence_id)} if initial_evidence_id else set()
    visible_ids.update(str(row["evidence_id"]) for row in prefix if row.get("evidence_id"))
    visible_predictions = {
        str(row.get("prediction", {}).get("prediction_id")) for row in visible_records
        if row.get("type") == "audit" and row.get("kind") == "prediction"
        and isinstance(row.get("prediction"), Mapping) and row.get("prediction", {}).get("prediction_id")
    }
    visible_prediction_rows = [
        row for row in visible_records
        if row.get("type") == "audit" and row.get("kind") == "prediction"
        and isinstance(row.get("prediction"), Mapping)
        and row.get("prediction", {}).get("prediction_id") in visible_predictions
    ]
    prediction_results = [
        row for row in visible_records
        if row.get("type") == "audit" and row.get("kind") == "prediction_result"
        and row.get("prediction_id") in visible_predictions
    ]
    for row in [*visible_prediction_rows, *prediction_results]:
        if row.get("evidence_id"):
            visible_ids.add(str(row["evidence_id"]))
        for field in ("prediction_evidence_id", "transition_evidence_id"):
            if row.get(field):
                visible_ids.add(str(row[field]))
    initial_frame_id = prefix[0].get("pre_frame_image_id")
    current_frame_id = last.get("post_frame_image_id")
    selected_images: list[tuple[str, str]] = []
    if initial_frame_id:
        selected_images.append(("Initial state image (frame handle; cite the initial state evidence ID)", initial_frame_id))
    recent_prefix = prefix[-int(horizon):]
    if conditioning == "memory_delta":
        for transition in recent_prefix:
            if transition.get("delta_image_id"):
                selected_images.append((
                    f"Delta visualization (not a game frame) after action {transition['action_index']}, evidence {transition['evidence_id']}",
                    transition["delta_image_id"],
                ))
    if current_frame_id:
        selected_images.append((f"Current state after {len(prefix)} analyzed actions (actual game frame)", current_frame_id))
    frames = FrameStore(run_path)
    image_paths = []
    image_labels = []
    for label, image_id in selected_images:
        entry = next((item for item in frames.entries() if item.get("frame_id") == image_id), None)
        if entry is None or entry.get("kind") not in {IMAGE_KIND, DELTA_KIND}:
            raise ValueError(f"missing rendered frame image: {image_id}")
        image_paths.append(frames.path_of(image_id))
        image_labels.append(label)

    game_id = manifest.get("game_id") or Path(str(manifest.get("source_trace", "unknown"))).stem
    run_id = manifest.get("run_id") or manifest.get("source_sha256", "replay")[:16]
    memory_store = GameMemory(
        run_path, store_dir=game_memory_dir(state_dir or default_state_dir(), str(game_id)),
        run_id=str(run_id),
    )
    memory = memory_store.compact(max_entries=memory_context_entries, through_step=visible_step)
    if conditioning == "pixels_actions":
        memory = {**memory, "entries": [], "count": 0, "compacted": False}
    for entry in memory["entries"]:
        visible_ids.update(str(value) for value in entry.get("evidence_ids", []))
    context = {
        "analysis_schema": DECISION_SCHEMA,
        "trace": manifest["source_trace"],
        "initial_state_evidence_id": initial_evidence_id,
        "current_evidence_id": last["evidence_id"],
        "citable_evidence_ids": sorted(visible_ids),
        "visible_prediction_ids": sorted(visible_predictions),
        "frame_ids_are_image_handles_not_evidence": True,
        "through_action": len(prefix),
        "current_state": {
            "level": last.get("levels_completed"),
            "win_levels": last.get("win_levels"),
            "game_state": last.get("game_state"),
            "legal_actions": last.get("legal_actions", []),
            "frame_id": last.get("post_frame_id"),
            "frame_shape": last.get("frame_shape"),
        },
        "transitions": [
            {
                "evidence_id": row["evidence_id"],
                "action_index": row["action_index"],
                "action": row["action"],
                "args": row.get("args", {}),
                "pre_frame_id": row.get("pre_frame_id"),
                "post_frame_id": row.get("post_frame_id"),
                "prediction_evidence_id": row.get("prediction_evidence_id"),
                "measurements": row.get("measurements"),
                "prediction": row.get("prediction"),
            }
            for row in prefix
        ],
        "prediction_audits": [
            {
                "evidence_id": row.get("evidence_id"),
                "prediction_id": row.get("prediction_id"),
                "transition_evidence_id": row.get("transition_evidence_id"),
                "verdict": row.get("verdict"), "details": row.get("details"),
            }
            for row in prediction_results
        ],
        "memory": memory if conditioning != "pixels_actions" else None,
        "conditioning": conditioning,
        "horizon": int(horizon),
        "budgets": {"remaining_action_budget": None, "reset_cost": "unknown"},
        "available_tools": ["game.observe", "game.act", "vision.diff", "history.query", "memory.read", "memory.propose_update"],
    }
    if include_prior_decisions:
        prior = []
        decisions_path = run_path / "decisions.jsonl"
        if decisions_path.exists():
            for row in read_jsonl(decisions_path):
                if int(row.get("through_action", 0)) >= len(prefix):
                    continue
                decision = row.get("decision", {}).get("decision", {})
                prior.append({
                    "through_action": row.get("through_action"),
                    "mode": decision.get("mode"),
                    "priority_question": row.get("decision", {}).get("priority_question"),
                    "planned_actions": [item.get("action") for item in decision.get("actions", [])],
                })
        context["prior_decision_summaries"] = prior[-4:]
    default_skill = run_path / "SKILL.md"
    if not default_skill.exists():
        default_skill = Path(__file__).with_name("SKILL.md")
    skill_file = Path(skill_path) if skill_path is not None else default_skill
    system_prompt = skill_file.read_text(encoding="utf-8")
    skill_digest = hashlib.sha256(skill_file.read_bytes()).hexdigest()
    prompt = (
        "Analyze the visible replay prefix and return one JSON object matching the supplied decision contract. "
        "Do not infer outcomes for actions that have not occurred. Cite only citable_evidence_ids in the context. "
        "F-prefixed frame IDs are image handles, not evidence citations. Delta images are changed-pixel overlays, "
        "not actual game states; use the current-state image for positions.\n\n"
        "Decision contract:\n"
        + json.dumps(_decision_contract_description(), ensure_ascii=False, indent=2)
        + "\n\nVisible replay context:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "context": context,
        "image_paths": image_paths,
        "image_labels": image_labels,
        "visible_evidence_ids": visible_ids,
        "visible_prediction_ids": visible_predictions,
        "legal_action_specs": _action_specs(prefix, last),
        "prefix_transitions": prefix,
        "visible_step": visible_step,
        "manifest": manifest,
        "skill_sha256": skill_digest,
        "run_id": str(run_id),
        "game_id": str(game_id),
        "memory_store_dir": memory_store.store_dir,
    }


def _decision_contract_description() -> dict[str, Any]:
    return {
        "required_fields": sorted(ROOT_FIELDS),
        "observations_item": {"id": "string", "claim": "string", "evidence_ids": "visible IDs"},
        "patterns_item": {"id": "string", "relation": "string", "conditions": "string", "evidence_ids": "visible IDs", "exceptions": "string[]", "status": "candidate|supported|contradicted"},
        "hypotheses_item": {"id": "string", "claim": "string", "alternatives": "string[]", "prediction": "string", "falsification": "string", "status": "candidate|supported|rejected"},
        "prediction_audit_item": {"prediction_id": "visible prior prediction", "verdict": "matched|failed|inconclusive", "evidence_ids": "visible IDs"},
        "goals_item": {"id": "string", "claim": "string", "status": "unverified|supported|rejected", "evidence_ids": "visible IDs"},
        "decision": {
            "mode": "explore|exploit|recover", "continue_or_reset": "continue|reset",
            "reset_justification": "string or null", "actions": "0 to 4 legal actions",
        },
        "action_item": {
            "action": "legal action ID", "args": "object validated against available action arguments",
            "prediction_id": "unique string", "expected_observation": "string",
            "measurable_prediction": {
                "required": True,
                "when_measurable": {
                    "all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}],
                    "each_predicate": "exactly path, op, value",
                    "operators": list(PREDICTION_OPERATORS),
                    "between_bounds": "inclusive [minimum, maximum]",
                    "allowed_paths": sorted(PREDICTION_PATHS),
                },
                "when_no_measurement_fits": "null and a non-empty unmeasurable_reason",
            },
            "unmeasurable_reason": "required only when measurable_prediction is null",
            "stop_if": "string[]",
        },
        "evidence_ids": {
            "allowed_values": "citable_evidence_ids in visible context",
            "frame_ids": "F-prefixed image handles are not evidence IDs and must never be cited",
        },
        "memory_patch": {
            "upsert": {
                "required": ["op", "id", "type", "claim", "evidence_ids"],
                "type": list(ENTRY_TYPES),
                "evidence_ids": "non-empty list of citable_evidence_ids",
                "optional_fields": sorted(MODEL_PATCH_FIELDS - {"op", "id", "type", "claim", "evidence_ids"}),
            },
            "retire": {"required": ["op", "id"], "allowed_fields": ["op", "id"]},
            "allowed_ops": list(MODEL_PATCH_OPS),
            "verdict": "harness-owned; do not emit",
            "observation_facts": "use decision.observations; memory entry type state is used for durable state facts",
        },
    }


def _decision_repair_prompt(
    user_prompt: str,
    validation_error: str,
    visible_evidence_ids: set[str],
) -> str:
    """Append actionable repair guidance based on the canonical decision schema."""
    return (
        user_prompt
        + "\n\nYour previous JSON failed validation: " + validation_error
        + "\nReturn the complete corrected decision JSON. Keep valid fields unchanged."
        + "\nCite only these evidence IDs: " + ", ".join(sorted(visible_evidence_ids))
        + ". F-prefixed frame handles identify images and are not citable evidence."
        + "\nEvery action must contain measurable_prediction. Use null plus unmeasurable_reason only if no listed measurement path fits."
        + "\nFor measurable_prediction, use {\"all\":[{\"path\":\"measurements.changed_pixels\",\"op\":\"gt\",\"value\":0}]}. "
        + "Allowed paths: " + ", ".join(sorted(PREDICTION_PATHS)) + "."
        + " Operators: " + ", ".join(PREDICTION_OPERATORS)
        + "; between uses inclusive [minimum, maximum]. Do not invent field names or comparison operators."
        + "\nMemory patches use op upsert or retire. Upsert types: " + ", ".join(ENTRY_TYPES)
        + "; each upsert requires id, type, claim, and non-empty evidence_ids. Verdicts are harness-owned."
        + "\nDo not add unsupported facts or evidence."
    )


def score_recorded_plan(
    decision: Mapping[str, Any],
    future_transitions: Sequence[dict[str, Any]],
    initial_state: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score only the contiguous plan prefix actually present in the recorded trace."""
    actions = decision["decision"]["actions"]
    audits = []
    diverged = False
    previous_levels = (initial_state or {}).get("levels_completed")
    for index, planned in enumerate(actions):
        audit: dict[str, Any] = {"prediction_id": planned["prediction_id"], "plan_index": index}
        if diverged:
            audit.update({"verdict": "unscorable", "reason": "prior action diverged from recorded trace"})
        elif index >= len(future_transitions):
            diverged = True
            audit.update({"verdict": "unscorable", "reason": "recorded continuation ended"})
        else:
            actual = future_transitions[index]
            if planned["action"] != actual.get("action") or planned.get("args", {}) != actual.get("args", {}):
                diverged = True
                audit.update({
                    "verdict": "unscorable", "reason": "planned action differs from recorded action",
                    "recorded_evidence_id": actual.get("evidence_id"),
                    "recorded_action": actual.get("action"), "recorded_args": actual.get("args", {}),
                })
            else:
                outcome = {
                    "measurements": actual.get("measurements", {}),
                    "game_state": actual.get("game_state"),
                    "level": actual.get("level"),
                    "levels_completed": actual.get("levels_completed"),
                    "win_levels": actual.get("win_levels"),
                    "terminal": actual.get("terminal", False),
                }
                verdict = evaluate_prediction(
                    planned.get("measurable_prediction"), outcome,
                    unmeasurable_reason=planned.get("unmeasurable_reason"),
                )
                audit.update({
                    **verdict,
                    "recorded_evidence_id": actual.get("evidence_id"),
                    "recorded_action": actual.get("action"),
                })
                current_levels = actual.get("levels_completed")
                if previous_levels is not None and current_levels is not None and current_levels != previous_levels:
                    diverged = True
                    audit["interrupt_reason"] = "level_change"
                previous_levels = current_levels
                state = str(actual.get("game_state", "")).casefold()
                if state in {"win", "game_over", "finished", "done", "success", "failed", "terminal"}:
                    diverged = True
                    audit["interrupt_reason"] = "terminal_state"
                if verdict.get("verdict") == "failed":
                    diverged = True
                    audit["interrupt_reason"] = "prediction_mismatch"
        audits.append(audit)
    return audits


def _parse_model_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise DecisionValidationError(f"response is not valid JSON: {exc}") from exc


def run_analysis(
    run_dir: str | Path,
    client: Any,
    through_action: int = 4,
    skill_path: str | Path | None = None,
    *,
    conditioning: str = "memory_delta",
    horizon: int = 4,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
    include_prior_decisions: bool = False,
    persist_memory: bool = True,
    write_decision: bool = True,
) -> dict[str, Any]:
    """Call the model on a bounded prefix, validate, then score its recorded continuation."""
    inputs = build_analysis_input(
        run_dir, through_action=through_action, skill_path=skill_path,
        conditioning=conditioning, horizon=horizon, state_dir=state_dir,
        memory_context_entries=memory_context_entries,
        include_prior_decisions=include_prior_decisions,
    )
    first = client.complete(
        inputs["system_prompt"], inputs["user_prompt"], inputs["image_paths"], inputs["image_labels"],
    )
    responses = [first]
    repair_error = None
    try:
        decision = validate_model_decision(
            _parse_model_content(first["content"]),
            inputs["legal_action_specs"],
            inputs["visible_evidence_ids"],
            inputs["visible_prediction_ids"],
        )
    except DecisionValidationError as exc:
        repair_error = str(exc)
        repair_prompt = _decision_repair_prompt(
            inputs["user_prompt"], repair_error, inputs["visible_evidence_ids"],
        )
        repaired = client.complete(
            inputs["system_prompt"], repair_prompt, inputs["image_paths"], inputs["image_labels"],
        )
        responses.append(repaired)
        decision = validate_model_decision(
            _parse_model_content(repaired["content"]),
            inputs["legal_action_specs"],
            inputs["visible_evidence_ids"],
            inputs["visible_prediction_ids"],
        )

    # The future continuation is read and scored only after the model response is valid.
    prefix_last = inputs["prefix_transitions"][-1]
    all_records = History(run_dir).records()
    future = [
        row for row in all_records
        if row.get("type") == "transition"
        and int(row.get("action_index", 0)) > int(prefix_last["action_index"])
    ]
    future = [row for row in future if int(row.get("reset_generation", 0)) == int(prefix_last.get("reset_generation", 0))]
    audits = score_recorded_plan(decision, future, initial_state=prefix_last)
    memory_result = {"accepted": [], "rejected": [], "revision": None}
    if persist_memory and conditioning != "pixels_actions":
        memory_store = GameMemory(
            run_dir, store_dir=inputs["memory_store_dir"], run_id=inputs["run_id"],
        )
        memory_result = memory_store.propose_update(
            decision["memory_patch"],
            known_evidence_ids=inputs["visible_evidence_ids"],
            current_step=inputs["visible_step"],
        )
    record = {
        "schema": "v10.analysis_run/1",
        "decision_schema": DECISION_SCHEMA,
        "prompt_version": "v10.analysis_prompt/1",
        "source_trace_sha256": inputs["manifest"].get("source_sha256"),
        "skill_sha256": inputs["skill_sha256"],
        "model": responses[-1].get("model"),
        "through_action": int(through_action),
        "conditioning": conditioning,
        "horizon": int(horizon),
        "visible_evidence_ids": sorted(inputs["visible_evidence_ids"]),
        "visible_transition_ids": [row["evidence_id"] for row in inputs["prefix_transitions"]],
        "input_frame_image_ids": [Path(path).stem for path in inputs["image_paths"]],
        "input_image_labels": inputs["image_labels"],
        "decision": decision,
        "prediction_audits": audits,
        "memory_updates": memory_result,
        "repair_attempted": len(responses) > 1,
        "initial_validation_error": repair_error,
        "usage": _sum_usage(responses),
        "latency_seconds": round(sum(float(response.get("latency_seconds", 0.0)) for response in responses), 6),
    }
    if write_decision:
        append_jsonl(Path(run_dir) / "decisions.jsonl", record)
    return record


def _sum_usage(responses: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [response.get("usage", {}).get(key) for response in responses]
        result[key] = sum(int(value) for value in values if value is not None) if any(value is not None for value in values) else None
    return result


__all__ = [
    "DECISION_SCHEMA", "DecisionValidationError", "build_analysis_input",
    "run_analysis", "score_recorded_plan", "validate_model_decision",
]
