"""Live single-agent loop over GameTools and a Strata-compatible model client."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analysis import (
    DECISION_SCHEMA,
    DecisionValidationError,
    _decision_repair_prompt,
    _decision_contract_description,
    model_action_name,
    model_facing_action_specs,
    validate_model_decision,
)
from .frames import DELTA_KIND, IMAGE_KIND, FrameStore
from .runlog import append_jsonl, write_json
from .strata_client import StrataRequestError
from .tools.game import ActionError, BudgetExceeded, GameTools, evaluate_prediction
from .tools.history import History
from .tools.memory import (
    DEFAULT_CONTEXT_ENTRIES,
    GameMemory,
    default_state_dir,
    game_memory_dir,
)


def _parse_content(content: Any) -> Any:
    if not isinstance(content, str):
        raise DecisionValidationError("response content must be text")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DecisionValidationError(f"response is not valid JSON: {exc}") from exc
    return payload


def _prepare_live_run_directory(destination: Path) -> None:
    """Create a clean run directory without following or removing the directory itself."""
    if destination.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {destination}")
    resolved = destination.resolve()
    protected = {
        Path("/").resolve(), Path.home().resolve(), Path("/tmp").resolve(),
        Path("/var/tmp").resolve(), Path.cwd().resolve(),
    }
    if any(resolved == path or path.is_relative_to(resolved) for path in protected):
        raise ValueError(f"refusing to clear a protected run directory: {destination}")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"run directory is not a directory: {destination}")
    if destination.exists():
        for child in destination.iterdir():
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
    else:
        destination.mkdir(parents=True, exist_ok=True)


def _public_actions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        source = [dict(descriptor, id=key) if isinstance(descriptor, Mapping) else {"id": key}
                  for key, descriptor in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        source = list(value)
    else:
        return []
    result = []
    for item in source:
        if isinstance(item, str):
            result.append({"id": item, "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}})
        elif isinstance(item, Mapping):
            public = {key: child for key, child in item.items() if not key.startswith("_")}
            result.append(public)
    return result


def _random_schema_value(schema: Mapping[str, Any], rng: random.Random) -> tuple[bool, Any]:
    """Produce a simple JSON-schema value for a required action argument."""
    if "const" in schema:
        return True, schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return True, rng.choice(enum)
    kind = schema.get("type")
    if kind == "integer":
        lower = int(schema.get("minimum", 0))
        upper = int(schema.get("maximum", lower))
        if "exclusiveMinimum" in schema:
            lower = int(schema["exclusiveMinimum"]) + 1
        if "exclusiveMaximum" in schema:
            upper = int(schema["exclusiveMaximum"]) - 1
        return (True, rng.randint(lower, upper)) if lower <= upper else (False, None)
    if kind == "number":
        lower = float(schema.get("minimum", 0))
        upper = float(schema.get("maximum", lower))
        if "exclusiveMinimum" in schema:
            lower = float(schema["exclusiveMinimum"]) + 1e-6
        if "exclusiveMaximum" in schema:
            upper = float(schema["exclusiveMaximum"]) - 1e-6
        return (True, rng.uniform(lower, upper)) if lower <= upper else (False, None)
    if kind == "boolean":
        return True, bool(rng.getrandbits(1))
    if kind == "string" and "pattern" not in schema:
        length = max(0, int(schema.get("minLength", 0)))
        return True, "x" * length
    if kind == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            return False, None
        count = max(0, int(schema.get("minItems", 0)))
        values = []
        for _ in range(count):
            valid, value = _random_schema_value(items, rng)
            if not valid:
                return False, None
            values.append(value)
        return True, values
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return False, None
        result = {}
        for name in required:
            child = properties.get(name)
            if not isinstance(child, Mapping):
                return False, None
            valid, value = _random_schema_value(child, rng)
            if not valid:
                return False, None
            result[name] = value
        return True, result
    return False, None


def _random_legal_action(legal_actions: Any, rng: random.Random) -> dict[str, Any] | None:
    """Select a legal action and generate arguments for its required fields."""
    candidates = []
    for action in _public_actions(legal_actions):
        action_id = action.get("id")
        parameters = action.get("parameters", action.get("args_schema", {}))
        if not isinstance(action_id, str) or not action_id or not isinstance(parameters, Mapping):
            continue
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            continue
        args = {}
        valid = True
        for name in required:
            schema = properties.get(name)
            if not isinstance(schema, Mapping):
                valid = False
                break
            value_valid, value = _random_schema_value(schema, rng)
            if not value_valid:
                valid = False
                break
            args[name] = value
        if valid:
            candidates.append({"action": action_id, "args": args})
    return rng.choice(candidates) if candidates else None


_REVIEW_SYSTEM_PROMPT = """# ez01 incremental memory review

Review this run's ordered observations against the existing structured patterns. You do not control the game and must not request actions or resets. Return only a JSON object with a `changes` array and boolean `legacy_memory_migrated`. Set the latter true only when all useful legacy MEMORY.md findings have been converted to structured patterns or strategies; otherwise keep the legacy note active. Do not rewrite the whole memory.

Each change is one of:
- An upsert with `operation`, optional existing `pattern_id`, `pattern_type` (`mechanic`, `strategy`, `exception`, or `hypothesis`), a concise `claim`, optional `applicability_conditions`, optional `state_selector`, and only newly observed `supporting_evidence` / `contradicting_evidence` IDs from this review's events.
- A retirement with `operation: "retire"` and an existing `pattern_id`.

When revising an existing pattern, reuse its ID. Use distinct conditional variants when different conditions explain different outcomes; retire or supersede a broad pattern only when justified. Include evidence references only in the structured evidence arrays, never in prose. Do not invent facts or confidence/status values; the harness computes confidence and status deterministically.

For useful successful action sequences, use `pattern_type: "strategy"` and provide `goal`, `preconditions` (string list), `action_policy`, `expected_observations`, and `success_conditions` (string list). Each expected observation has `action`, optional `state_selector` over game_state/level/levels_completed/terminal, and a measurable prediction in the supported all-of predicate format. Do not produce strategies that are not supported by a successful transition.

Distinguish direct observations from explanations. Direction words name commands; infer movement from frames and measurements. Keep claims game-specific and concise. Return `{"changes": [], "legacy_memory_migrated": true}` when no useful legacy content remains."""


def _validate_review_payload(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Mapping) or set(value) - {"changes", "legacy_memory_migrated"} or "changes" not in value:
        raise ValueError("review response must be a JSON object containing changes")
    legacy_migrated = value.get("legacy_memory_migrated", False)
    if not isinstance(legacy_migrated, bool):
        raise ValueError("legacy_memory_migrated must be a boolean")
    changes = value["changes"]
    if not isinstance(changes, list):
        raise ValueError("review changes must be a list")
    if any(not isinstance(change, Mapping) for change in changes):
        raise ValueError("each review change must be an object")
    return [dict(change) for change in changes], legacy_migrated


def _review_events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in records:
        kind = row.get("type")
        if kind == "state":
            events.append({
                "type": "initial_state", "evidence_id": row.get("evidence_id"),
                "level": row.get("level"), "levels_completed": row.get("levels_completed"),
                "reset_generation": row.get("reset_generation"), "game_state": row.get("game_state"),
            })
        elif kind == "reset":
            events.append({
                "type": "reset_boundary", "evidence_id": row.get("evidence_id"),
                "level": row.get("level"), "levels_completed": row.get("levels_completed"),
                "reset_generation": row.get("reset_generation"),
                "justification": row.get("justification"),
            })
        elif kind == "audit":
            if row.get("kind") == "pattern_prediction":
                events.append({
                    "type": "pattern_prediction", "evidence_id": row.get("evidence_id"),
                    "pattern_id": row.get("pattern_id"), "transition_id": row.get("transition_id"),
                    "verdict": row.get("verdict"), "action": model_action_name(row.get("action")),
                    "expected": row.get("expected"), "actual": row.get("actual"),
                    "reason": row.get("reason"), "reset_generation": row.get("reset_generation"),
                })
            else:
                events.append({
                    "type": "action_error", "evidence_id": row.get("evidence_id"),
                    "kind": row.get("kind"), "action": model_action_name(row.get("action")),
                    "error": row.get("error"), "reset_generation": row.get("reset_generation"),
                })
        elif kind == "transition":
            events.append({
                "type": "transition", "evidence_id": row.get("evidence_id"),
                "action": model_action_name(row.get("action")), "args": row.get("args", {}),
                "measurements": row.get("measurements"), "result": row.get("result"),
                "level": row.get("level"), "levels_completed": row.get("levels_completed"),
                "game_state": row.get("game_state"), "terminal": row.get("terminal"),
                "reset_generation": row.get("reset_generation"),
            })
    return events


def _review_frame_selection(
    transitions: Sequence[Mapping[str, Any]],
    frames: FrameStore,
    *,
    max_images: int = 8,
) -> tuple[list[Path], list[str]]:
    """Choose at most four informative before/after pairs in chronological order."""
    if max_images < 2 or not transitions:
        return [], []
    frame_entries = {entry["frame_id"]: entry for entry in frames.entries()}
    pair_limit = max_images // 2
    scores: list[tuple[int, int]] = []
    previous_completed: int | None = 0
    for index, row in enumerate(transitions):
        score = 0
        completed = row.get("levels_completed")
        if completed is not None:
            completed = int(completed)
            if previous_completed is not None and completed > previous_completed:
                score += 10
            previous_completed = completed
        if row.get("terminal"):
            score += 9
        measurements = row.get("measurements")
        if isinstance(measurements, Mapping) and measurements.get("changed_pixels") == 0:
            score += 4
        if index in {0, len(transitions) - 1}:
            score += 2
        scores.append((score, index))
    ranked = [index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))]
    selected = ranked[:pair_limit]
    if len(selected) < pair_limit:
        remaining = [index for index in range(len(transitions)) if index not in selected]
        while remaining and len(selected) < pair_limit:
            target = round((len(selected) + 1) * (len(transitions) - 1) / (pair_limit + 1))
            index = min(remaining, key=lambda item: (abs(item - target), item))
            selected.append(index)
            remaining.remove(index)
    selected.sort()

    image_paths: list[Path] = []
    image_labels: list[str] = []
    for index in selected:
        row = transitions[index]
        evidence_id = str(row.get("evidence_id", f"transition-{index + 1}"))
        pair = []
        for field, label in (("pre_frame_image_id", "before"), ("post_frame_image_id", "after")):
            image_id = row.get(field)
            entry = frame_entries.get(str(image_id)) if image_id else None
            if entry is None or entry.get("kind") != IMAGE_KIND:
                pair = []
                break
            pair.append((frames.path_of(str(image_id)), f"{label.title()} frame for {evidence_id}"))
        for path, label in pair:
            image_paths.append(path)
            image_labels.append(label)
    return image_paths[:max_images], image_labels[:max_images]


def _perform_review(
    tools: GameTools,
    client: Any,
    *,
    game_id: str,
    run_id: str,
    memory_store_dir: Path,
    trigger_reasons: Sequence[str],
) -> dict[str, Any]:
    """Review the frozen run prefix and persist a consolidated Markdown summary."""
    transition_ids: list[str] = []
    first_id: str | None = None
    last_id: str | None = None
    image_labels: list[str] = []
    usage: dict[str, Any] = {}
    latency = 0.0
    attempts = 0
    try:
        records = tools.history.records()
        transitions = [row for row in records if row.get("type") == "transition"]
        transition_ids = [str(row["evidence_id"]) for row in transitions if row.get("evidence_id")]
        first_id = transition_ids[0] if transition_ids else None
        last_id = transition_ids[-1] if transition_ids else None
        memory = GameMemory(tools.run_dir, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
        memory_text = memory.write_view().read_text(encoding="utf-8")
        legacy_review = memory.get("review:latest")
        legacy_review_text = str(legacy_review.get("claim", "")) if legacy_review else ""
        if memory.has_review_coverage(game_id, transition_ids):
            row = {
                "schema": "v10.live_review/2", "status": "skipped_duplicate_evidence",
                "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
                "covered_transitions": {"count": len(transition_ids), "first": first_id, "last": last_id},
                "selected_image_labels": [], "request_attempts": 0, "usage": {},
                "latency_seconds": 0.0, "duplicate_review": True,
            }
            append_jsonl(tools.run_dir / "reviews.jsonl", row)
            return {"row": row, "attempts": 0, "usage": {}, "latency_seconds": 0.0}
        frames = FrameStore(tools.run_dir)
        image_paths, image_labels = _review_frame_selection(transitions, frames)
        events = _review_events(records)
        patterns = [
            {
                "pattern_id": entry["id"], "pattern_type": entry.get("pattern_type"),
                "claim": entry.get("claim"), "applicability_conditions": entry.get("applicability_conditions", []),
                "state_selector": entry.get("state_selector", {}), "status": entry.get("status"),
                "confidence": entry.get("confidence"),
                "support_count": len(entry.get("supporting_evidence", [])),
                "contradiction_count": len(entry.get("contradicting_evidence", [])),
                **{
                    key: entry[key] for key in (
                        "goal", "preconditions", "action_policy", "expected_observations", "success_conditions",
                    ) if key in entry
                },
            }
            for entry in memory.patterns(game_id, include_retired=True)
        ]
        user_prompt = (
            f"Review all recorded experience for game {game_id} through action {tools.action_count}. "
            f"This review was triggered by: {', '.join(trigger_reasons)}. "
            "Update only patterns affected by observations since their last review. Add only new evidence IDs; never repeat an ID already attached to a pattern. "
            "When evidence conflicts, consider whether applicability conditions distinguish the observations; refine or split patterns if supported. "
            "Keep evidence IDs in the structured evidence arrays only.\n\n"
            "Existing patterns (authoritative ledger projection):\n"
            + json.dumps(patterns, ensure_ascii=False, separators=(",", ":")) + "\n\n"
            "Existing MEMORY.md view (legacy content may be migrated into patterns):\n"
            + memory_text + "\n\n"
            "Prior unstructured review, if any:\n" + legacy_review_text + "\n\n"
            "New ordered run observations (evidence IDs are valid only in the evidence arrays):\n"
            + json.dumps(events, ensure_ascii=False, separators=(",", ":"))
        )
        attempts = 1
        response = client.complete(
            _REVIEW_SYSTEM_PROMPT, user_prompt, image_paths, image_labels,
            response_format="json_object",
        )
        usage = dict(response.get("usage", {}))
        latency = float(response.get("latency_seconds", 0.0))
        attempts = max(1, int(response.get("request_attempts", 1)))
        content = response.get("content")
        if response.get("finish_reason") == "length":
            raise ValueError("review response was truncated at the output-token limit")
        changes, legacy_migrated = _validate_review_payload(_parse_content(content))
        applied = memory.apply_review_changes(
            changes, game_id=game_id, evidence_ids=transition_ids,
            current_step=tools.action_count,
        )
        if not applied.get("committed", True):
            raise ValueError(f"review contained no valid changes: {applied['rejected']}")
        legacy_review_retired = memory.retire_legacy_review(current_step=tools.action_count) if legacy_migrated else False
        canonical = json.dumps(changes, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row = {
            "schema": "v10.live_review/2",
            "status": "saved_with_rejections" if applied["rejected"] else "saved",
            "trigger_reasons": list(trigger_reasons),
            "through_action": tools.action_count,
            "covered_transitions": {"count": len(transition_ids), "first": first_id, "last": last_id},
            "selected_image_labels": image_labels,
            "model": response.get("model", getattr(client, "model", None)),
            "request_attempts": attempts, "usage": usage,
            "latency_seconds": round(latency, 6),
            "memory_revision": memory.revision(), "memory_path": str(memory.view_path),
            "patterns_added": applied["added"], "patterns_updated": applied["updated"],
            "patterns_retired": applied["retired"], "unchanged_patterns": applied["unchanged"],
            "rejected_changes": applied["rejected"], "duplicate_review": applied["duplicate_review"],
            "legacy_memory_migrated": legacy_migrated, "legacy_review_retired": legacy_review_retired,
            "result_characters": len(canonical),
            "result_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
    except Exception as exc:
        attempts = max(attempts, int(getattr(exc, "request_attempts", 0)))
        if getattr(exc, "latency_seconds", None) is not None:
            latency = max(latency, float(exc.latency_seconds))
        row = {
            "schema": "v10.live_review/2", "status": "failed", "trigger_reasons": list(trigger_reasons),
            "through_action": tools.action_count,
            "covered_transitions": {"count": len(transition_ids), "first": first_id, "last": last_id},
            "selected_image_labels": image_labels,
            "request_attempts": attempts, "usage": usage,
            "latency_seconds": round(latency, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }
    append_jsonl(tools.run_dir / "reviews.jsonl", row)
    return {"row": row, "attempts": attempts, "usage": usage, "latency_seconds": latency}


def _live_context(
    tools: GameTools,
    current: Mapping[str, Any],
    game_id: str,
    horizon: int,
    skill_path: Path,
    memory_store_dir: Path,
    run_id: str,
    memory_context_entries: int,
) -> dict[str, Any]:
    records = tools.history.records()
    transitions = [
        row for row in records
        if row.get("type") == "transition"
        and int(row.get("reset_generation", 0)) == int(tools.reset_generation)
    ][-horizon:]
    first = next((row for row in records if row.get("type") == "state"), None)
    visible_ids = {str(current["evidence_id"])} if current.get("evidence_id") else set()
    if first and first.get("evidence_id"):
        visible_ids.add(str(first["evidence_id"]))
    visible_ids.update(str(row["evidence_id"]) for row in transitions if row.get("evidence_id"))
    legal_actions = _public_actions(current.get("legal_actions", []))
    if tools.reset_count < tools.max_resets:
        legal_actions.append({
            "id": "RESET",
            "description": "Reset the game when a controlled restart is justified.",
            "parameters": {
                "type": "object", "properties": {"justification": {"type": "string"}},
                "required": ["justification"], "additionalProperties": False,
            },
        })
    visible_step = int(current.get("step", max((int(row.get("step", 0)) for row in records), default=0)))
    memory_store = GameMemory(tools.run_dir, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    memory = memory_store.compact(
        max_entries=memory_context_entries, through_step=visible_step,
        game_id=game_id, current_state=current,
    )
    for entry in memory["entries"]:
        visible_ids.update(str(value) for value in entry.get("evidence_ids", []))
    context = {
        "game_id": game_id,
        "current_state": {
            "game_state": current.get("game_state"), "level": current.get("level"),
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "terminal": current.get("terminal", False),
            "legal_actions": legal_actions,
        },
        "recent_transitions": [
            {
                "action": row.get("action"), "args": row.get("args", {}),
                "measurements": row.get("measurements"), "level": row.get("level"),
                "levels_completed": row.get("levels_completed"), "game_state": row.get("game_state"),
            }
            for row in transitions
        ],
        "memory": memory,
        "budgets": {
            "actions_used": tools.action_count, "actions_remaining": max(0, tools.max_actions - tools.action_count),
            "resets_used": tools.reset_count, "resets_remaining": max(0, tools.max_resets - tools.reset_count),
            "plan_horizon": horizon,
        },
    }
    image_ids: list[tuple[str, str]] = []
    current_image_id = current.get("frame_image_id")
    previous_image_id = transitions[-1].get("pre_frame_image_id") if transitions else None
    if previous_image_id and previous_image_id != current_image_id:
        image_ids.append(("Previous actual game frame", str(previous_image_id)))
    if current_image_id:
        image_ids.append(("Current actual game frame", str(current_image_id)))
    frames = FrameStore(tools.run_dir)
    entries = {entry["frame_id"]: entry for entry in frames.entries()}
    image_paths: list[Path] = []
    image_labels: list[str] = []
    for label, frame_id in image_ids:
        entry = entries.get(frame_id)
        if entry is None or entry.get("kind") not in {IMAGE_KIND, DELTA_KIND}:
            continue
        image_paths.append(frames.path_of(frame_id))
        image_labels.append(label)
    system_prompt = skill_path.read_text(encoding="utf-8")
    model_context = dict(context)
    model_current_state = dict(context["current_state"])
    model_current_state["legal_actions"] = model_facing_action_specs(legal_actions)
    model_context["current_state"] = model_current_state
    model_context["recent_transitions"] = [
        {**row, "action": model_action_name(row.get("action"))}
        for row in context["recent_transitions"]
    ]
    prompt = (
        f"Choose up to {horizon} legal actions from the current state, recent outcomes, and read-only memory. "
        "Return only an actions list. Each action must contain only action and args. "
        "Do not include analysis, memory updates, predictions, goals, or narrative. "
        "If RESET is listed, use it alone and include args.justification.\n\n"
        "Required response shape:\n" + json.dumps(_decision_contract_description(), ensure_ascii=False, separators=(",", ":"))
        + "\n\nVisible context:\n" + json.dumps(model_context, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "system_prompt": system_prompt, "user_prompt": prompt, "context": context,
        "image_paths": image_paths, "image_labels": image_labels,
        "visible_evidence_ids": visible_ids,
        "legal_action_specs": legal_actions, "visible_step": visible_step,
        "skill_sha256": hashlib.sha256(skill_path.read_bytes()).hexdigest(),
    }


def run_live(
    env: Any,
    run_dir: str | Path,
    client: Any,
    *,
    game_id: str,
    seed: int = 0,
    max_actions: int = 30,
    max_resets: int = 0,
    plan_horizon: int = 4,
    review_interval: int = 50,
    skill_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
) -> dict[str, Any]:
    """Run a bounded live episode, persisting evidence before and after actions."""
    if not 1 <= int(plan_horizon) <= 4:
        raise ValueError("plan_horizon must be between 1 and 4")
    if int(review_interval) < 1:
        raise ValueError("review_interval must be positive")
    if int(max_actions) < 1 or int(max_resets) < 0:
        raise ValueError("action budget must be positive and reset budget non-negative")
    if not 0 <= int(memory_context_entries) <= 5:
        raise ValueError("memory_context_entries must be between 0 and 5")
    destination = Path(run_dir)
    source_skill = Path(skill_path) if skill_path is not None else Path(__file__).with_name("SKILL.md")
    run_id = uuid.uuid4().hex
    memory_store_dir = game_memory_dir(state_dir or default_state_dir(), game_id)
    # Old versions kept the only copy of knowledge inside the disposable run directory.
    GameMemory.migrate_run_ledger(destination, memory_store_dir, game_id)
    _prepare_live_run_directory(destination)
    memory = GameMemory(destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    memory.write_view()
    skill_sha256 = hashlib.sha256(source_skill.read_bytes()).hexdigest()
    write_json(destination / "run_manifest.json", {
        "schema": "v10.live_run/1", "run_id": run_id, "game_id": game_id,
        "seed": int(seed), "skill_sha256": skill_sha256,
        "persistent_memory_dir": str(memory_store_dir),
    })
    tools = GameTools(env, destination, max_actions=max_actions, max_resets=max_resets)
    current = tools.observe()
    memory.record_run_state(game_id, current, step=tools.action_count)
    model_calls = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    action_usage = dict(total_usage)
    review_usage = dict(total_usage)
    total_latency = 0.0
    review_latency = 0.0
    review_model_calls = 0
    review_count = 0
    pattern_predictions = 0
    pattern_prediction_matches = 0
    pattern_prediction_contradictions = 0
    pattern_prediction_inconclusive = 0
    actions_since_review = 0
    interrupted_plans = 0
    stop_reason = "running"
    decisions_path = destination / "decisions.jsonl"
    start_time = time.monotonic()
    fallback_rng = random.Random(int(seed))

    def update_metrics() -> dict[str, Any]:
        elapsed = round(time.monotonic() - start_time, 3)
        terminal = bool(current.get("terminal"))
        state = current.get("game_state")
        result = {
            "schema": "v10.live_metrics/2", "game_id": game_id, "model": getattr(client, "model", None),
            "run_id": run_id, "seed": int(seed),
            "actions": tools.action_count, "action_budget": tools.max_actions,
            "resets": tools.reset_count, "reset_budget": tools.max_resets,
            "model_calls": model_calls, "usage": dict(total_usage),
            "action_usage": dict(action_usage), "review_usage": dict(review_usage),
            "review_model_calls": review_model_calls, "review_count": review_count,
            "pattern_predictions": pattern_predictions,
            "pattern_prediction_matches": pattern_prediction_matches,
            "pattern_prediction_contradictions": pattern_prediction_contradictions,
            "pattern_prediction_inconclusive": pattern_prediction_inconclusive,
            "review_latency_seconds": round(review_latency, 6),
            "review_interval": int(review_interval),
            "latency_seconds": round(total_latency, 6), "elapsed_seconds": elapsed,
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "game_state": state, "terminal": terminal, "won": state == "WIN",
            "interrupted_plans": interrupted_plans, "stop_reason": stop_reason,
            "memory_entries": GameMemory(
                destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
            ).read().get("count", 0),
            "memory_context_entries": int(memory_context_entries),
        }
        write_json(destination / "metrics.json", result)
        return result

    def progress(message: str) -> None:
        """Write concise live progress so long inference calls remain visible."""
        level = current.get("level")
        completed = current.get("levels_completed")
        total = current.get("win_levels")
        print(
            f"[v10 live] level={level if level is not None else '?'} "
            f"({completed if completed is not None else '?'} completed"
            f"/{total if total is not None else '?'}), "
            f"actions={tools.action_count}/{tools.max_actions} | {message}",
            file=sys.stderr, flush=True,
        )

    update_metrics()
    progress(f"started game={game_id}")
    try:
        while tools.action_count < tools.max_actions and not current.get("terminal"):
            inputs = _live_context(
                tools, current, game_id, int(plan_horizon), source_skill,
                memory_store_dir, run_id, int(memory_context_entries),
            )
            responses: list[dict[str, Any]] = []
            validation_error: str | None = None
            decision: dict[str, Any] | None = None
            for attempt in range(2):
                model_calls += 1
                try:
                    response = client.complete(
                        inputs["system_prompt"], inputs["user_prompt"],
                        inputs["image_paths"], inputs["image_labels"],
                    )
                except StrataRequestError as exc:
                    model_calls += max(0, int(getattr(exc, "request_attempts", 1)) - 1)
                    total_latency += float(getattr(exc, "latency_seconds", 0.0))
                    raise
                model_calls += max(0, int(response.get("request_attempts", 1)) - 1)
                responses.append(response)
                for key in total_usage:
                    value = response.get("usage", {}).get(key)
                    if value is not None:
                        total_usage[key] += int(value)
                        action_usage[key] += int(value)
                total_latency += float(response.get("latency_seconds", 0.0))
                try:
                    decision = validate_model_decision(
                        _parse_content(response.get("content")), inputs["legal_action_specs"],
                        max_actions=plan_horizon,
                    )
                    break
                except (DecisionValidationError, ActionError) as exc:
                    # Never retain a partially accepted candidate across repair.
                    decision = None
                    validation_error = str(exc)
                    if attempt == 0:
                        inputs["user_prompt"] = _decision_repair_prompt(
                            inputs["user_prompt"], validation_error,
                        )
            if decision is None:
                append_jsonl(decisions_path, {
                    "schema": "v10.live_decision/2", "status": "rejected",
                    "game_id": game_id, "visible_evidence_ids": sorted(inputs["visible_evidence_ids"]),
                    "validation_error": validation_error, "repair_attempted": len(responses) > 1,
                    "model": responses[-1].get("model"), "usage": _sum_usage(responses),
                    "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                    "latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in responses), 6),
                })
                fallback = _random_legal_action(current.get("legal_actions", []), fallback_rng)
                if fallback is None:
                    stop_reason = "invalid_decision_no_legal_fallback"
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_fallback/2", "status": "unavailable",
                        "reason": "no legal action has arguments that can be generated safely",
                        "validation_error": validation_error,
                    })
                    break
                action_number = tools.action_count + 1
                progress(
                    f"decision invalid after repair; taking random legal fallback action "
                    f"{action_number}: {fallback['action']}"
                )
                result = tools.act(fallback["action"], fallback["args"])
                current = result["observation"]
                memory.record_run_state(game_id, current, step=tools.action_count)
                append_jsonl(decisions_path, {
                    "schema": "v10.live_fallback/2", "status": "executed",
                    "cause": "invalid_decision", "validation_error": validation_error,
                    "action": fallback["action"], "args": fallback["args"],
                    "evidence_id": result["evidence"]["evidence_id"],
                })
                progress(
                    f"fallback action {tools.action_count}: {fallback['action']} -> "
                    f"{current.get('game_state', '?')}; observation captured"
                )
                update_metrics()
                continue

            decision_row = {
                "schema": "v10.live_decision/2", "status": "accepted",
                "game_id": game_id, "decision_schema": DECISION_SCHEMA,
                "prompt_version": "v10.live_prompt/3", "skill_sha256": inputs["skill_sha256"],
                "model": responses[-1].get("model"),
                "visible_evidence_ids": sorted(inputs["visible_evidence_ids"]),
                "input_image_labels": inputs["image_labels"], "decision": decision,
                "repair_attempted": len(responses) > 1,
                "initial_validation_error": validation_error, "usage": _sum_usage(responses),
                "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                "latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in responses), 6),
            }
            append_jsonl(decisions_path, decision_row)

            plan = decision
            progress(
                f"decision ready in {decision_row['latency_seconds']:.1f}s; "
                f"planned={[item['action'] for item in plan['actions']]}"
            )
            reset_action = next((item for item in plan["actions"] if item["action"] == "RESET"), None)
            if reset_action is not None:
                try:
                    reset_result = tools.reset(reset_action["args"]["justification"])
                except BudgetExceeded:
                    stop_reason = "reset_budget_exhausted"
                    append_jsonl(decisions_path, {"schema": "v10.live_plan_result/2", "status": stop_reason})
                    break
                current = reset_result["observation"]
                progress("reset executed")
                interrupted_plans += 1
                append_jsonl(decisions_path, {
                    "schema": "v10.live_plan_result/2", "status": "reset",
                    "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                })
                update_metrics()
                continue
            actions = plan["actions"]
            if not actions:
                stop_reason = "model_returned_no_actions"
                append_jsonl(decisions_path, {"schema": "v10.live_plan_result/2", "status": stop_reason})
                break

            executed: list[dict[str, Any]] = []
            interruption: str | None = None
            for action in actions[:plan_horizon]:
                if tools.action_count >= tools.max_actions:
                    interruption = "action_budget_exhausted"
                    break
                legal_now = {item.get("id") for item in _public_actions(current.get("legal_actions", []))}
                if action["action"] not in legal_now:
                    interruption = "planned_action_no_longer_legal"
                    break
                previous_level = current.get("level")
                previous_levels_completed = current.get("levels_completed")
                prediction_memory = GameMemory(
                    destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
                )
                expected_pattern = prediction_memory.expected_observation(game_id, current, action["action"])
                try:
                    next_number = tools.action_count + 1
                    progress(f"taking action {next_number}: {action['action']}")
                    result = tools.act(
                        action["action"], action.get("args", {}),
                    )
                except (ActionError, BudgetExceeded) as exc:
                    interruption = "action_error" if isinstance(exc, ActionError) else "action_budget_exhausted"
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_plan_result/2", "status": "action_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    break
                current = result["observation"]
                memory = GameMemory(
                    destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
                )
                memory.record_run_state(game_id, current, step=tools.action_count)
                progress(
                    f"action {tools.action_count}: {action['action']} -> "
                    f"{current.get('game_state', '?')}"
                )
                executed_action = {"evidence_id": result["evidence"]["evidence_id"]}
                prediction_failed = False
                if expected_pattern is not None:
                    actual = {
                        "measurements": result.get("measurements", {}),
                        "game_state": current.get("game_state"), "level": current.get("level"),
                        "levels_completed": current.get("levels_completed"),
                        "win_levels": current.get("win_levels"), "terminal": current.get("terminal", False),
                    }
                    verdict_row = evaluate_prediction(expected_pattern["measurable_prediction"], actual)
                    verdict = str(verdict_row.get("verdict", "inconclusive"))
                    audit = tools.history.append({
                        "type": "audit", "kind": "pattern_prediction",
                        "pattern_id": expected_pattern["pattern_id"],
                        "transition_id": result["evidence"]["evidence_id"],
                        "action": action["action"], "expected": expected_pattern["expected_observation"],
                        "actual": actual, "verdict": verdict, "reason": verdict_row.get("reason"),
                        "reset_generation": tools.reset_generation,
                    })
                    memory.record_pattern_prediction(
                        expected_pattern["pattern_id"], result["evidence"]["evidence_id"], verdict,
                        audit["evidence_id"], step=tools.action_count,
                    )
                    pattern_predictions += 1
                    if verdict == "matched":
                        pattern_prediction_matches += 1
                    elif verdict == "failed":
                        pattern_prediction_contradictions += 1
                        prediction_failed = True
                    else:
                        pattern_prediction_inconclusive += 1
                    executed_action["pattern_prediction"] = {
                        "pattern_id": expected_pattern["pattern_id"],
                        "audit_id": audit["evidence_id"], "verdict": verdict,
                    }
                    progress(f"pattern prediction {verdict}: {expected_pattern['pattern_id']}")
                executed.append(executed_action)
                actions_since_review += 1
                trigger_reasons = []
                if prediction_failed:
                    trigger_reasons.append("unexpected_transition")
                level_completed = (
                    previous_levels_completed is not None
                    and current.get("levels_completed") is not None
                    and int(current["levels_completed"]) > int(previous_levels_completed)
                )
                if level_completed:
                    trigger_reasons.append("level_completed")
                if actions_since_review >= int(review_interval):
                    trigger_reasons.append("action_interval")
                if trigger_reasons:
                    review_count += 1
                    progress(f"review starting; triggers={trigger_reasons}")
                    review_result = _perform_review(
                        tools, client, game_id=game_id, run_id=run_id,
                        memory_store_dir=memory_store_dir,
                        trigger_reasons=trigger_reasons,
                    )
                    # One attempt, successful or not, starts the next interval.
                    actions_since_review = 0
                    attempts = int(review_result["attempts"])
                    review_model_calls += attempts
                    model_calls += attempts
                    elapsed = float(review_result["latency_seconds"])
                    review_latency += elapsed
                    total_latency += elapsed
                    for key in total_usage:
                        value = review_result["usage"].get(key)
                        if value is not None:
                            total_usage[key] += int(value)
                            review_usage[key] += int(value)
                    progress(f"review {review_result['row']['status']}")
                if current.get("terminal"):
                    interruption = "terminal_state"
                elif current.get("level") != previous_level or current.get("levels_completed") != previous_levels_completed:
                    interruption = "level_change"
                elif trigger_reasons:
                    interruption = "review"
                if interruption:
                    break
            if interruption:
                interrupted_plans += 1
            append_jsonl(decisions_path, {
                "schema": "v10.live_plan_result/2", "status": "interrupted" if interruption else "completed",
                "executed": executed, "interrupt_reason": interruption,
            })
            update_metrics()
            if interruption == "action_budget_exhausted":
                stop_reason = interruption
                break
            if interruption == "action_error":
                stop_reason = interruption
                break
            if current.get("terminal"):
                stop_reason = "win" if current.get("game_state") == "WIN" else "game_over"
                break
        if current.get("terminal"):
            stop_reason = "win" if current.get("game_state") == "WIN" else "game_over"
        elif tools.action_count >= tools.max_actions and stop_reason not in {
            "invalid_decision", "invalid_decision_no_legal_fallback", "reset_budget_exhausted",
            "model_returned_no_actions", "action_error",
        }:
            stop_reason = "action_budget_exhausted"
        return update_metrics()
    except (StrataRequestError, OSError, RuntimeError, ValueError) as exc:
        stop_reason = "model_or_runtime_error"
        append_jsonl(decisions_path, {
            "schema": "v10.live_error/1", "error": f"{type(exc).__name__}: {exc}",
            "request_attempts": int(getattr(exc, "request_attempts", 1)),
            "latency_seconds": round(float(getattr(exc, "latency_seconds", 0.0)), 6),
        })
        update_metrics()
        raise


def _sum_usage(responses: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [response.get("usage", {}).get(key) for response in responses]
        result[key] = sum(int(value) for value in values if value is not None) if any(value is not None for value in values) else None
    return result


__all__ = ["run_live"]
