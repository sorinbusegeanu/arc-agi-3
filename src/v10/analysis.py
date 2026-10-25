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
    DEFAULT_CONTEXT_ENTRIES, GameMemory, default_state_dir, game_memory_dir,
)


DECISION_SCHEMA = "v10.action_set/2"
class DecisionValidationError(ValueError):
    """Model output failed the v10 decision contract."""


def validate_model_decision(
    payload: Any,
    legal_action_specs: Sequence[Mapping[str, Any]],
    max_actions: int = 4,
) -> dict[str, Any]:
    """Validate the model's action-only plan against current legal action schemas."""
    try:
        return validate_decision(payload, legal_action_specs, max_actions=max_actions)
    except ActionError as exc:
        raise DecisionValidationError(str(exc)) from exc


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
    memory = memory_store.compact(max_entries=memory_context_entries, through_step=visible_step)
    if conditioning == "pixels_actions":
        memory = {**memory, "entries": [], "count": 0, "compacted": False}
    for entry in memory["entries"]:
        visible_ids.update(str(value) for value in entry.get("evidence_ids", []))
    context = {
        "current_state": {
            "level": last.get("levels_completed"),
            "win_levels": last.get("win_levels"),
            "game_state": last.get("game_state"),
            "legal_actions": last.get("legal_actions", []),
        },
        "recent_transitions": [
            {
                "action": row["action"],
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
        default_skill = Path(__file__).with_name("SKILL.md")
    skill_file = Path(skill_path) if skill_path is not None else default_skill
    system_prompt = skill_file.read_text(encoding="utf-8")
    skill_digest = hashlib.sha256(skill_file.read_bytes()).hexdigest()
    prompt = (
        f"Choose up to {horizon} legal actions from the visible replay and return only the action list. "
        "Each action must contain only action and args. Do not include analysis, memory updates, predictions, "
        "goals, or narrative. If RESET is listed, use it alone and include args.justification.\n\n"
        "Required response shape:\n"
        + json.dumps(_decision_contract_description(), ensure_ascii=False, indent=2)
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
    return {
        "actions": [{
            "action": "LEGAL_ACTION_ID",
            "args": {},
        }],
    }


def _decision_repair_prompt(
    user_prompt: str,
    validation_error: str,
) -> str:
    """Append concise repair guidance for the action-list contract."""
    return (
        user_prompt
        + "\n\nYour previous JSON failed validation: " + validation_error
        + "\nReturn only a corrected JSON object shaped as {\"actions\":[...]}."
        + " Each action must contain only action and args. Do not include analysis, memory updates, predictions, goals, prose, or markdown."
    )


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
        decision = validate_model_decision(
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
        decision = validate_model_decision(
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
        "decision_schema": DECISION_SCHEMA,
        "prompt_version": "v10.analysis_prompt/2",
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
        "decision": {"actions": decision["actions"]},
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
    "DECISION_SCHEMA", "DecisionValidationError", "build_analysis_input",
    "run_analysis", "score_recorded_plan", "validate_model_decision",
]
