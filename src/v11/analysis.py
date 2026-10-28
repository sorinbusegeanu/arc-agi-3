"""Prefix-bounded multimodal analysis and recorded-continuation scoring."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .frames import DELTA_KIND, FrameStore, IMAGE_KIND
from .runlog import append_jsonl, read_json
from .tools.game import ActionError, validate_decision
from .tools.history import History
from .tools.memory import (
    DEFAULT_CONTEXT_ENTRIES, GameMemory, default_state_dir, game_memory_dir, shared_skill_path,
)


DECISION_SCHEMA = "v10.action_set/2"
REASONED_ACTION_SCHEMA = "v10.reasoned_action/1"

_DIRECTION_BY_ACTION = {
    "ACTION1": "up",
    "ACTION2": "down",
    "ACTION3": "left",
    "ACTION4": "right",
}
_MODEL_ALIAS_BY_ACTION = {**{action: name for action, name in _DIRECTION_BY_ACTION.items()}, "ACTION6": "click"}


def model_action_aliases(legal_action_specs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map legal model-facing aliases to canonical ARC action IDs."""
    legal_ids = {str(spec.get("id")) for spec in legal_action_specs}
    return {
        name: action_id
        for action_id, name in _MODEL_ALIAS_BY_ACTION.items()
        if action_id in legal_ids
    }


def model_action_name(action_id: Any) -> Any:
    """Return a direction label for the four canonical ARC movement actions."""
    return _DIRECTION_BY_ACTION.get(str(action_id), action_id)


def model_facing_action_specs(
    legal_action_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose directional names in prompts while retaining each action schema."""
    result = []
    for spec in legal_action_specs:
        item = dict(spec)
        action_id = str(item.get("id", ""))
        if action_id in _MODEL_ALIAS_BY_ACTION:
            item["id"] = _MODEL_ALIAS_BY_ACTION[action_id]
        result.append(item)
    return result


def _canonicalize_model_actions(payload: Any, legal_action_specs: Sequence[Mapping[str, Any]]) -> Any:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("actions"), list):
        return payload
    aliases = model_action_aliases(legal_action_specs)
    if not aliases:
        return payload
    canonical_ids = set(aliases.values())
    normalized = dict(payload)
    normalized_actions = []
    for index, action in enumerate(payload["actions"]):
        if isinstance(action, Mapping):
            action_name = action.get("action")
            if isinstance(action_name, str) and action_name in aliases:
                action = {**action, "action": aliases[action_name]}
            elif isinstance(action_name, str) and action_name in canonical_ids:
                alias = _DIRECTION_BY_ACTION.get(action_name, _MODEL_ALIAS_BY_ACTION[action_name])
                kind = "directional name" if action_name in _DIRECTION_BY_ACTION else "model-facing name"
                raise ActionError(
                    f"actions[{index}] must use {kind} {alias!r} instead of {action_name}"
                )
        normalized_actions.append(action)
    normalized["actions"] = normalized_actions
    return normalized


class DecisionValidationError(ValueError):
    """Model output failed the v10 decision contract."""


def validate_model_decision(
    payload: Any,
    legal_action_specs: Sequence[Mapping[str, Any]],
    max_actions: int = 4,
) -> dict[str, Any]:
    """Validate the model's action-only plan against current legal action schemas."""
    try:
        return validate_decision(
            _canonicalize_model_actions(payload, legal_action_specs),
            legal_action_specs, max_actions=max_actions,
        )
    except ActionError as exc:
        raise DecisionValidationError(str(exc)) from exc


_REASONED_TOP_LEVEL = {
    "state_assessment", "hypothesis", "trajectory_rationale", "prediction", "actions",
}
_CERTAINTY_LABELS = {"observed", "inferred", "unknown"}
_PREDICTED_VISUAL_CHANGE = {"yes", "no", "unknown"}
_PREDICTED_OUTCOMES = {"level_change", "GAME_OVER", "WIN", "no_terminal", "unknown"}


def _bounded_text(value: Any, field: str, limit: int, *, empty_ok: bool = False) -> str:
    if not isinstance(value, str) or (not empty_ok and not value.strip()):
        raise DecisionValidationError(f"{field} must be a non-empty string")
    if len(value) > limit:
        raise DecisionValidationError(f"{field} exceeds the {limit}-character limit")
    return value.strip()


def _require_keys(value: Any, expected: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DecisionValidationError(f"{field} must contain exactly: {', '.join(sorted(expected))}")
    return value


def validate_reasoned_action(
    payload: Any,
    legal_action_specs: Sequence[Mapping[str, Any]],
    max_actions: int = 8,
) -> dict[str, Any]:
    """Validate structured reasoning, then apply the unchanged legal-action validator."""
    response = _require_keys(payload, _REASONED_TOP_LEVEL, "reasoned action")

    assessment = _require_keys(
        response["state_assessment"],
        {"player", "target_candidates", "obstacles", "other_relevant_changes"},
        "state_assessment",
    )
    player = _require_keys(assessment["player"], {"location", "certainty", "evidence"}, "player")
    _bounded_text(player["location"], "player.location", 120)
    if not isinstance(player["certainty"], str) or player["certainty"] not in _CERTAINTY_LABELS:
        raise DecisionValidationError("player.certainty must be observed, inferred, or unknown")
    _bounded_text(player["evidence"], "player.evidence", 180)

    candidates = assessment["target_candidates"]
    if not isinstance(candidates, list) or len(candidates) > 5:
        raise DecisionValidationError("state_assessment.target_candidates must be a list of at most five candidates")
    for index, raw in enumerate(candidates):
        candidate = _require_keys(raw, {"description", "location", "relevance", "certainty"}, f"target_candidates[{index}]")
        for key, limit in (("description", 100), ("location", 100), ("relevance", 140)):
            _bounded_text(candidate[key], f"target_candidates[{index}].{key}", limit)
        if not isinstance(candidate["certainty"], str) or candidate["certainty"] not in _CERTAINTY_LABELS:
            raise DecisionValidationError(f"target_candidates[{index}].certainty must be observed, inferred, or unknown")

    obstacles = assessment["obstacles"]
    if not isinstance(obstacles, list) or len(obstacles) > 8:
        raise DecisionValidationError("state_assessment.obstacles must be a list of at most eight obstacles")
    for index, raw in enumerate(obstacles):
        obstacle = _require_keys(raw, {"description", "location", "status", "evidence"}, f"obstacles[{index}]")
        for key, limit in (("description", 100), ("location", 100), ("evidence", 160)):
            _bounded_text(obstacle[key], f"obstacles[{index}].{key}", limit)
        if not isinstance(obstacle["status"], str) or obstacle["status"] not in {"observed_blocked", "suspected_blocked", "unknown"}:
            raise DecisionValidationError(f"obstacles[{index}].status is not a supported categorical label")

    changes = assessment["other_relevant_changes"]
    if not isinstance(changes, list) or len(changes) > 5:
        raise DecisionValidationError("state_assessment.other_relevant_changes must have at most five entries")
    for index, change in enumerate(changes):
        _bounded_text(change, f"other_relevant_changes[{index}]", 180)

    hypothesis = response["hypothesis"]
    if hypothesis is not None:
        if not isinstance(hypothesis, Mapping) or set(hypothesis) != {"claim", "evidence"}:
            raise DecisionValidationError("hypothesis must contain claim and evidence")
        _bounded_text(hypothesis["claim"], "hypothesis.claim", 240)
        _bounded_text(hypothesis["evidence"], "hypothesis.evidence", 180)
        hypothesis = {"claim": hypothesis["claim"], "evidence": hypothesis["evidence"]}

    _bounded_text(response["trajectory_rationale"], "trajectory_rationale", 420)
    prediction = _require_keys(
        response["prediction"], {"visible_change", "environment_outcome", "description"}, "prediction",
    )
    if not isinstance(prediction["visible_change"], str) or prediction["visible_change"] not in _PREDICTED_VISUAL_CHANGE:
        raise DecisionValidationError("prediction.visible_change must be yes, no, or unknown")
    if not isinstance(prediction["environment_outcome"], str) or prediction["environment_outcome"] not in _PREDICTED_OUTCOMES:
        raise DecisionValidationError("prediction.environment_outcome is not a supported outcome")
    _bounded_text(prediction["description"], "prediction.description", 240)

    try:
        validated = validate_decision(
            _canonicalize_model_actions({"actions": response["actions"]}, legal_action_specs),
            legal_action_specs, max_actions=max_actions,
        )
    except ActionError as exc:
        raise DecisionValidationError(str(exc)) from exc
    return {
        "state_assessment": dict(assessment), "hypothesis": dict(hypothesis) if hypothesis else None,
        "trajectory_rationale": response["trajectory_rationale"],
        "prediction": dict(prediction), "actions": validated["actions"],
    }


def reasoned_action_contract_description(
    legal_action_specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Single canonical model response contract; ``actions`` is the only action list."""
    faced_specs: list[dict[str, Any]] = []
    if legal_action_specs is None:
        example_action, example_args = "up", {}
    else:
        faced_specs = model_facing_action_specs(legal_action_specs)
        example_spec = faced_specs[0] if faced_specs else {"id": "legal action"}
        example_action = str(example_spec.get("id", "legal action"))
        schema = example_spec.get("parameters", example_spec.get("args_schema", {}))
        properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
        example_args = {}
        if isinstance(properties, Mapping):
            for name in schema.get("required", []) if isinstance(schema, Mapping) else []:
                property_schema = properties.get(name, {})
                if not isinstance(property_schema, Mapping):
                    continue
                if "const" in property_schema:
                    example_args[name] = property_schema["const"]
                elif property_schema.get("type") == "integer":
                    example_args[name] = property_schema.get("minimum", 0)
                elif property_schema.get("type") == "number":
                    example_args[name] = property_schema.get("minimum", 0)
    click_only = bool(faced_specs) and {str(spec.get("id")) for spec in faced_specs} == {"click"}
    assessment_example = {
        "player": (
            {"location": "unknown", "certainty": "unknown", "evidence": "unknown or not present"}
            if click_only else
            {"location": "visual region or unknown", "certainty": "observed | inferred | unknown", "evidence": "brief visual basis"}
        ),
        "target_candidates": [] if click_only else [{"description": "visual candidate", "location": "visual region", "relevance": "why it may matter", "certainty": "observed | inferred | unknown"}],
        "obstacles": [] if click_only else [{"description": "visual element", "location": "visual region", "status": "observed_blocked | suspected_blocked | unknown", "evidence": "brief basis"}],
        "other_relevant_changes": ["brief observation"],
    }
    return {
        "state_assessment": assessment_example,
        "hypothesis": None,
        "trajectory_rationale": "one to three concise sentences; no chain-of-thought",
        "prediction": {"visible_change": "yes | no | unknown", "environment_outcome": "level_change | GAME_OVER | WIN | no_terminal | unknown", "description": "expected result of the first action only; do not describe later actions or the whole plan"},
        "actions": [{"action": example_action, "args": example_args}],
    }


def reasoned_action_limits_description() -> str:
    """Compact field caps shown to the model so valid output does not need repair."""
    return (
        "String limits: player location 120/evidence 180; target description 100, location 100, relevance 140; "
        "obstacle description 100, location 100, evidence 160; each observed change 180; "
        "hypothesis claim 240/evidence 180; trajectory rationale 420; prediction description 240 characters. "
        "Prediction fields apply only to the first action in actions and are compared only with that action's observed transition. "
        "Maximum lists: five targets, eight obstacles, five observed changes, and eight actions."
    )


def reasoned_action_repair_prompt(user_prompt: str, validation_error: str) -> str:
    return (
        user_prompt + "\n\nYour previous structured JSON failed validation: " + validation_error
        + "\nReturn one corrected JSON object using exactly the required structured fields. "
        + "Keep actions as the only action sequence, obey the categorical labels and field limits, and do not add prose."
    )


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
    conditioning: str = "pixels_actions_memory",
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
    if conditioning == "memory_delta":
        conditioning = "pixels_actions_memory"
    if conditioning not in {"pixels_actions", "pixels_actions_memory"}:
        raise ValueError("unknown conditioning profile")
    if not 0 <= int(memory_context_entries) <= 5:
        raise ValueError("memory_context_entries must be between 0 and 5")
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
    current_frame_id = last.get("post_frame_image_id")
    selected_images: list[tuple[str, str]] = []
    previous_frame_id = last.get("pre_frame_image_id")
    if previous_frame_id and previous_frame_id != current_frame_id:
        selected_images.append(("Previous actual game frame", previous_frame_id))
    if current_frame_id:
        selected_images.append(("Current actual game frame", current_frame_id))
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
    # Replay decisions consume the same model-authored Markdown as live runs.
    memory = memory_store.view_path.read_text(encoding="utf-8") if memory_store.view_path.is_file() else ""
    if conditioning == "pixels_actions" or int(memory_context_entries) == 0:
        memory = ""
    elif memory:
        # Offline experiments select whole Markdown blocks, never partial sentences.
        blocks = memory.strip().split("\n\n")
        selected: list[str] = []
        entries = 0
        for block in blocks:
            if entries >= int(memory_context_entries):
                break
            selected.append(block)
            if not all(line.lstrip().startswith("#") for line in block.splitlines() if line.strip()):
                entries += 1
        memory = "\n\n".join(selected) + "\n"
    context = {
        "current_state": {
            "level": last.get("levels_completed"),
            "win_levels": last.get("win_levels"),
            "game_state": last.get("game_state"),
            "legal_actions": model_facing_action_specs(_action_specs(prefix, last)),
        },
        "recent_transitions": [
            {
                "action": model_action_name(row["action"]),
                "args": row.get("args", {}),
                "measurements": row.get("measurements"),
                "game_state": row.get("game_state"),
                "levels_completed": row.get("levels_completed"),
            }
            for row in prefix[-int(horizon):]
        ],
        "memory": memory if conditioning != "pixels_actions" else None,
    }
    if include_prior_decisions:
        prior = []
        decisions_path = run_path / "decisions.jsonl"
        if decisions_path.exists():
            for row in read_jsonl(decisions_path):
                if int(row.get("through_action", 0)) >= len(prefix):
                    continue
                decision = row.get("decision", {})
                prior.append({
                    "through_action": row.get("through_action"),
                    "planned_actions": [item.get("action") for item in decision.get("actions", [])],
                })
        context["prior_decision_summaries"] = prior[-4:]
    default_skill = run_path / "SKILL.md"
    if not default_skill.exists():
        default_skill = shared_skill_path()
    skill_file = Path(skill_path) if skill_path is not None else default_skill
    system_prompt = skill_file.read_text(encoding="utf-8")
    skill_digest = hashlib.sha256(skill_file.read_bytes()).hexdigest()
    prompt = (
        f"Choose up to {horizon} legal actions from the visible replay and return the structured reasoned-action JSON. "
        "Do not provide chain-of-thought. Keep actions as the only action list. If RESET is listed, use it alone and include args.justification.\n\n"
        "Required response shape (`v10.reasoned_action/1`):\n"
        + json.dumps(reasoned_action_contract_description(context["current_state"]["legal_actions"]), ensure_ascii=False, indent=2)
        + "\n\n" + reasoned_action_limits_description()
        + "\n\nVisible replay context:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "context": context,
        "initial_state_evidence_id": initial_evidence_id,
        "image_paths": image_paths,
        "image_labels": image_labels,
        "visible_evidence_ids": visible_ids,
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
    """Compatibility name for the current structured action response shape."""
    return reasoned_action_contract_description()


def _decision_repair_prompt(
    user_prompt: str,
    validation_error: str,
) -> str:
    """Compatibility wrapper for repairing the single reasoned-action schema."""
    return reasoned_action_repair_prompt(user_prompt, validation_error)


def score_recorded_plan(
    decision: Mapping[str, Any],
    future_transitions: Sequence[dict[str, Any]],
    initial_state: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare a proposed action sequence with the recorded continuation."""
    actions = decision.get("actions", [])
    matches = []
    diverged = False
    previous_levels = (initial_state or {}).get("levels_completed")
    for index, planned in enumerate(actions):
        match: dict[str, Any] = {"plan_index": index, "planned_action": planned["action"]}
        if diverged:
            match.update({"status": "unscorable", "reason": "prior action diverged or interrupted the trace"})
        elif index >= len(future_transitions):
            diverged = True
            match.update({"status": "unscorable", "reason": "recorded continuation ended"})
        else:
            actual = future_transitions[index]
            if planned["action"] != actual.get("action") or planned.get("args", {}) != actual.get("args", {}):
                diverged = True
                match.update({
                    "status": "mismatch", "reason": "planned action differs from recorded action",
                    "recorded_evidence_id": actual.get("evidence_id"),
                    "recorded_action": actual.get("action"), "recorded_args": actual.get("args", {}),
                })
            else:
                match.update({
                    "status": "matched",
                    "recorded_evidence_id": actual.get("evidence_id"),
                    "recorded_action": actual.get("action"),
                })
                current_levels = actual.get("levels_completed")
                if previous_levels is not None and current_levels is not None and current_levels != previous_levels:
                    diverged = True
                    match["interrupt_reason"] = "level_change"
                previous_levels = current_levels
                state = str(actual.get("game_state", "")).casefold()
                if state in {"win", "game_over", "finished", "done", "success", "failed", "terminal"}:
                    diverged = True
                    match["interrupt_reason"] = "terminal_state"
        matches.append(match)
    return matches


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
    conditioning: str = "pixels_actions_memory",
    horizon: int = 4,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
    include_prior_decisions: bool = False,
    write_decision: bool = True,
) -> dict[str, Any]:
    """Call the model on a bounded prefix, validate, then score its recorded continuation."""
    if conditioning == "memory_delta":
        conditioning = "pixels_actions_memory"
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
        decision = validate_reasoned_action(
            _parse_model_content(first["content"]),
            inputs["legal_action_specs"],
            max_actions=horizon,
        )
    except DecisionValidationError as exc:
        repair_error = str(exc)
        repair_prompt = _decision_repair_prompt(
            inputs["user_prompt"], repair_error,
        )
        repaired = client.complete(
            inputs["system_prompt"], repair_prompt, inputs["image_paths"], inputs["image_labels"],
        )
        responses.append(repaired)
        decision = validate_reasoned_action(
            _parse_model_content(repaired["content"]),
            inputs["legal_action_specs"],
            max_actions=horizon,
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
    action_matches = score_recorded_plan(decision, future, initial_state=prefix_last)
    record = {
        "schema": "v10.analysis_run/2",
        "decision_schema": REASONED_ACTION_SCHEMA,
        "prompt_version": "v10.analysis_prompt/2",
        "source_trace_sha256": inputs["manifest"].get("source_sha256"),
        "skill_sha256": inputs["skill_sha256"],
        "backend": responses[-1].get("backend", getattr(client, "backend", "strata")),
        "model": responses[-1].get("model"),
        "through_action": int(through_action),
        "conditioning": conditioning,
        "horizon": int(horizon),
        "visible_evidence_ids": sorted(inputs["visible_evidence_ids"]),
        "visible_transition_ids": [row["evidence_id"] for row in inputs["prefix_transitions"]],
        "input_frame_image_ids": [Path(path).stem for path in inputs["image_paths"]],
        "input_image_labels": inputs["image_labels"],
        "decision": decision,
        "action_matches": action_matches,
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
    "DECISION_SCHEMA", "REASONED_ACTION_SCHEMA", "DecisionValidationError", "build_analysis_input",
    "reasoned_action_contract_description", "reasoned_action_limits_description", "reasoned_action_repair_prompt",
    "run_analysis", "score_recorded_plan", "validate_model_decision", "validate_reasoned_action",
]
