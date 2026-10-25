"""Live single-agent loop over GameTools and a Strata-compatible model client."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import time
import uuid
from collections import Counter, deque
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
from .frames import IMAGE_KIND, FrameStore
from .runlog import append_jsonl, write_json
from .strata_client import StrataRequestError
from .tools.game import ActionError, BudgetExceeded, GameTools, evaluate_prediction
from .tools.history import History
from .tools.memory import (
    DEFAULT_CONTEXT_ENTRIES,
    MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
    MAX_MEMORY_MARKDOWN_CHARS,
    GameMemory,
    default_state_dir,
    game_memory_dir,
    global_memory_path,
    read_global_memory,
    save_game_memory,
    save_global_memory,
    shared_skill_path,
)
from .tools.solutions import LevelSolutionStore, level_key, state_fingerprint


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


_REVIEW_SYSTEM_PROMPT = """# V10 memory consolidation

Review existing game MEMORY.md and shared global MEMORY.md, new ordered observations, and verified successful routes. Return only one JSON object with exactly two non-empty Markdown strings: `game_memory_markdown` and `global_memory_markdown`.

Write very concise, decision-useful guidance. Merge repetition, qualify uncertainty, correct contradictions, and preserve demonstrated strategies with level or observed-state scope. Use short prose bullets under brief headings; game memory should contain only a few bullets. Do not include any digits, tables, timelines, raw observation narration, coordinates, pixel distances, displacement values, per-level action counts, or action sequences. Do not copy route details into memory; the exact-state route store handles precise route lengths and replay.

Routes are supplied only when the environment recorded level completion or WIN. Treat their outcome as fixed facts. You may merge equivalent routes into one high-level strategy while retaining scope in words (such as “on matching level start states”), but never derive or reproduce route lengths, exact actions, counts, or arguments. Game memory may contain lasting mechanics, useful approaches, repeated failures, and unresolved experiments. Describe a tested approach qualitatively (for example: “When the goal is above the player, move upward toward it; use the verified route when the starting state matches.”).

Global memory is at most a few bullets and contains only methods that transfer to unrelated games. Do not mention this game's colors, objects, directions, levels, pixel values, or specific rule. Generalize a reusable method (for example: “Compare each object's position before and after one action to infer action effects.”); if no transferable lesson was learned, retain only prior reusable guidance. Remove game-specific details from existing global memory rather than carrying them forward.

Separate observations from explanations. Unchanged pixels show only that an action had no visible effect in that state; do not infer walls, total reachability, or completion rules from proximity. Distinguish lasting mechanics from historical positions. Follow current observations when they conflict with prior memory. Do not put current run state in durable memory. Return complete replacement Markdown, retaining only concise findings that can change a future decision."""


def _validate_review_payload(
    value: Any,
    *,
    game_memory_max_chars: int = MAX_MEMORY_MARKDOWN_CHARS,
    global_memory_max_chars: int = MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
) -> tuple[str, str]:
    expected = {"game_memory_markdown", "global_memory_markdown"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("review response must contain exactly game_memory_markdown and global_memory_markdown")
    game_memory = value.get("game_memory_markdown")
    global_memory = value.get("global_memory_markdown")
    for name, markdown, limit in (
        ("game_memory_markdown", game_memory, game_memory_max_chars),
        ("global_memory_markdown", global_memory, global_memory_max_chars),
    ):
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError(f"{name} must be non-empty Markdown text")
        if len(markdown.strip()) > int(limit):
            raise ValueError(f"{name} exceeds the {int(limit)} character limit")
    return game_memory.strip(), global_memory.strip()


def _review_repair_prompt(base_prompt: str, errors: Any) -> str:
    """Ask for both complete Markdown fields after validation fails."""
    return (
        base_prompt
        + "\n\nYour prior review response was invalid. Return both complete replacement Markdown fields "
        "with concise content within the specified limits. Return only the required JSON object.\n"
        "Validation errors:\n" + json.dumps(errors, ensure_ascii=False, separators=(",", ":"))
    )


def _review_events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in records:
        kind = row.get("type")
        if kind == "state":
            events.append({
                "type": "initial_state",
                "level": row.get("level"), "levels_completed": row.get("levels_completed"),
                "reset_generation": row.get("reset_generation"), "game_state": row.get("game_state"),
            })
        elif kind == "reset":
            events.append({
                "type": "reset_boundary",
                "level": row.get("level"), "levels_completed": row.get("levels_completed"),
                "reset_generation": row.get("reset_generation"),
                "justification": row.get("justification"),
            })
        elif kind == "audit":
            if row.get("kind") == "pattern_prediction":
                events.append({
                    "type": "pattern_prediction",
                    "pattern_id": row.get("pattern_id"),
                    "verdict": row.get("verdict"), "action": model_action_name(row.get("action")),
                    "expected": row.get("expected"), "actual": row.get("actual"),
                    "reason": row.get("reason"), "reset_generation": row.get("reset_generation"),
                })
            else:
                events.append({
                    "type": "action_error",
                    "kind": row.get("kind"), "action": model_action_name(row.get("action")),
                    "error": row.get("error"), "reset_generation": row.get("reset_generation"),
                })
        elif kind == "transition":
            events.append({
                "type": "transition",
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
    initial_completed: int | None = 0,
) -> tuple[list[Path], list[str]]:
    """Choose at most four informative before/after pairs in chronological order."""
    if max_images < 2 or not transitions:
        return [], []
    frame_entries = {entry["frame_id"]: entry for entry in frames.entries()}
    pair_limit = max_images // 2
    categories: dict[str, list[int]] = {name: [] for name in ("progress", "interaction", "blocked", "movement")}
    previous_completed = initial_completed
    for index, row in enumerate(transitions):
        completed = row.get("levels_completed")
        if completed is not None:
            completed = int(completed)
            if previous_completed is not None and completed > previous_completed:
                categories["progress"].append(index)
            previous_completed = completed
        measurements = row.get("measurements")
        changed = measurements.get("changed_pixels") if isinstance(measurements, Mapping) else None
        if changed == 0:
            categories["blocked"].append(index)
        elif isinstance(changed, (int, float)) and changed > 0:
            categories["movement"].append(index)
        transitions_by_color = measurements.get("color_transitions", []) if isinstance(measurements, Mapping) else []
        if isinstance(transitions_by_color, list) and len(transitions_by_color) >= 3:
            categories["interaction"].append(index)
    selected: list[int] = []
    # Reserve image pairs for distinct evidence types before filling remaining slots.
    for category in ("progress", "interaction", "blocked", "movement"):
        options = categories[category]
        if options and len(selected) < pair_limit:
            selected.append(options[-1] if category == "progress" else options[0])
    ranked = sorted(
        set(range(len(transitions))) - set(selected),
        key=lambda index: (
            0 if transitions[index].get("terminal") else 1,
            index,
        ),
    )
    selected.extend(ranked[:max(0, pair_limit - len(selected))])
    selected.sort()

    image_paths: list[Path] = []
    image_labels: list[str] = []
    for index in selected:
        row = transitions[index]
        pair = []
        for field, label in (("pre_frame_image_id", "before"), ("post_frame_image_id", "after")):
            image_id = row.get(field)
            entry = frame_entries.get(str(image_id)) if image_id else None
            if entry is None or entry.get("kind") != IMAGE_KIND:
                pair = []
                break
            pair.append((frames.path_of(str(image_id)), f"{label.title()} frame for transition {index + 1}"))
        for path, label in pair:
            image_paths.append(path)
            image_labels.append(label)
    return image_paths[:max_images], image_labels[:max_images]


def _repeated_outcome_summary(
    transitions: Sequence[Mapping[str, Any]], frames: FrameStore,
) -> dict[str, Any]:
    """Summarize repeated visible states and state/action pairs with no pixel change."""
    state_counts: dict[str, int] = {}
    ineffective: dict[tuple[str, str], int] = {}
    for row in transitions:
        for field in ("pre_frame_id", "post_frame_id"):
            frame_id = row.get(field)
            if not isinstance(frame_id, str) or not frames.exists(frame_id):
                continue
            fingerprint = hashlib.sha256(frames.get(frame_id).tobytes()).hexdigest()[:12]
            state_counts[fingerprint] = state_counts.get(fingerprint, 0) + 1
            if field == "pre_frame_id" and (row.get("measurements") or {}).get("changed_pixels") == 0:
                pair = (fingerprint, str(model_action_name(row.get("action"))))
                ineffective[pair] = ineffective.get(pair, 0) + 1
    return {
        "repeated_states": [
            {"frame_fingerprint": key, "observations": count}
            for key, count in sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))
            if count > 1
        ][:6],
        "ineffective_actions": [
            {"state_fingerprint": state, "action": action, "unchanged_count": count}
            for (state, action), count in sorted(ineffective.items(), key=lambda item: (-item[1], item[0]))
        ][:8],
    }


def _action_screen_sequence(
    transitions: Sequence[Mapping[str, Any]],
    current_image_id: Any,
    frames: FrameStore,
    *,
    label_prefix: str = "Recent",
    current_label: str = "Current screen",
) -> tuple[list[Path], list[str], list[dict[str, Any]]]:
    """Return every observed screen in action order, including intermediate frames."""
    entries = {entry["frame_id"]: entry for entry in frames.entries()}
    ordered: list[tuple[str, str | None, str]] = []

    def add(image_id: Any, action: str | None, stage: str) -> None:
        if image_id is not None:
            ordered.append((str(image_id), action, stage))

    if transitions:
        first_action = str(model_action_name(transitions[0].get("action")))
        add(transitions[0].get("pre_frame_image_id"), None, f"initial before {first_action}")
        for row in transitions:
            action = str(model_action_name(row.get("action")))
            result = row.get("result")
            result = result if isinstance(result, Mapping) else {}
            intermediate_ids = result.get("intermediate_frame_image_ids")
            if not isinstance(intermediate_ids, list):
                intermediate_ids = row.get("intermediate_frame_image_ids", [])
            if not isinstance(intermediate_ids, list):
                intermediate_ids = []
            for index, image_id in enumerate(intermediate_ids, start=1):
                add(image_id, action, f"during {action}, intermediate screen {index}/{len(intermediate_ids)}")
            add(row.get("post_frame_image_id"), action, f"after {action}")
    add(current_image_id, None, current_label)

    image_paths: list[Path] = []
    image_labels: list[str] = []
    sequence: list[dict[str, Any]] = []
    last_image_id: str | None = None
    for image_id, action, stage in ordered:
        if image_id == last_image_id:
            continue
        entry = entries.get(image_id)
        if entry is None or entry.get("kind") != IMAGE_KIND:
            continue
        label = f"{label_prefix} screen {len(sequence) + 1}: {stage}" if transitions else stage
        image_paths.append(frames.path_of(image_id))
        image_labels.append(label)
        sequence.append({
            "screen": len(sequence) + 1, "action": action,
            "stage": stage, "label": label,
        })
        last_image_id = image_id
    return image_paths, image_labels, sequence


def _verified_route_summaries(solution_store: LevelSolutionStore) -> list[dict[str, Any]]:
    """Provide success facts for prose consolidation; the store retains exact routes."""
    summaries: list[dict[str, Any]] = []
    for route in solution_store.solutions():
        try:
            scope = json.loads(str(route.get("level_key", "{}")))
        except (TypeError, json.JSONDecodeError):
            scope = {"level_key": route.get("level_key")}
        summaries.append({
            "scope": scope,
            "environment_recorded_success": route.get("success"),
            "actions": [model_action_name(step.get("action")) for step in route.get("steps", [])],
            "step_count": len(route.get("steps", [])),
        })
    return summaries


def _perform_review(
    tools: GameTools,
    client: Any,
    *,
    game_id: str,
    run_id: str,
    memory_store_dir: Path,
    trigger_reasons: Sequence[str],
    game_memory_max_chars: int = MAX_MEMORY_MARKDOWN_CHARS,
    global_memory_max_chars: int = MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
) -> dict[str, Any]:
    """Consolidate observations into model-authored game and global Markdown."""
    transition_count = 0
    image_labels: list[str] = []
    usage: dict[str, Any] = {}
    latency = 0.0
    attempts = 0
    repair_history: list[dict[str, Any]] = []
    memory_saved = False
    global_saved = False
    memory = GameMemory(tools.run_dir, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    try:
        records = tools.history.records()
        transitions = [row for row in records if row.get("type") == "transition"]
        transition_count = len(transitions)
        state_root = memory_store_dir.parent.parent
        view_path = memory_store_dir / "MEMORY.md"
        existing_memory = view_path.read_text(encoding="utf-8").strip() if view_path.is_file() else ""
        legacy_review = memory.get("review:latest")
        if not existing_memory and legacy_review:
            existing_memory = str(legacy_review.get("claim", "")).strip()
        shared_memory = read_global_memory(state_root)
        events = _review_events(records)
        covered_event_count = len(events)
        if memory.has_review_coverage(game_id, covered_event_count):
            row = {
                "schema": "v10.live_review/3", "status": "skipped_duplicate_review",
                "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
                "covered_transition_count": transition_count, "selected_image_labels": [],
                "request_attempts": 0, "usage": {}, "latency_seconds": 0.0,
                "duplicate_review": True,
            }
            append_jsonl(tools.run_dir / "reviews.jsonl", row)
            return {"row": row, "attempts": 0, "usage": {}, "latency_seconds": 0.0}

        prior_event_count = min(memory.reviewed_event_count(game_id), len(events))
        new_events = events[prior_event_count:]
        frames = FrameStore(tools.run_dir)
        previous_transition_count = sum(event.get("type") == "transition" for event in events[:prior_event_count])
        if previous_transition_count:
            initial_completed = transitions[previous_transition_count - 1].get("levels_completed")
        else:
            initial_state = next((row for row in records if row.get("type") == "state"), {})
            initial_completed = initial_state.get("levels_completed", 0)
        new_transitions = transitions[previous_transition_count:]
        image_paths, image_labels = _review_frame_selection(
            new_transitions, frames,
            initial_completed=int(initial_completed) if initial_completed is not None else None,
        )
        solution_store = LevelSolutionStore(memory_store_dir / "level_solutions.jsonl", game_id)
        verified_routes = _verified_route_summaries(solution_store)
        user_prompt = (
            f"Consolidate memory for game {game_id} after action {tools.action_count}. "
            f"Review trigger: {', '.join(trigger_reasons)}. Current progress belongs only in observation context, not either memory file. "
            "Preserve level/state scope when summarizing successful approaches. Existing game MEMORY.md:\n"
            + (existing_memory or "(none)")
            + "\n\nExisting shared global MEMORY.md:\n"
            + (shared_memory or "(none)")
            + "\n\nNew ordered observations since the last successful review checkpoint:\n"
            + json.dumps(new_events, ensure_ascii=False, separators=(",", ":"))
            + "\n\nRepeated states and ineffective state/action summary:\n"
            + json.dumps(_repeated_outcome_summary(transitions, tools.frames), ensure_ascii=False, separators=(",", ":"))
            + "\n\nVerified successful routes (outcomes and scope are environment facts; summarize strategies without copying routes):\n"
            + json.dumps(verified_routes, ensure_ascii=False, separators=(",", ":"))
            + f"\n\nLength limits: game Markdown at most {int(game_memory_max_chars)} characters; "
            + f"global Markdown at most {int(global_memory_max_chars)} characters. Keep game memory to a few concise bullets "
            "and global memory to only broadly reusable bullets."
        )
        responses: list[dict[str, Any]] = []
        review_prompt = user_prompt
        game_markdown = ""
        global_markdown = ""
        for repair_index in range(2):
            attempts += 1
            try:
                response = client.complete(
                    _REVIEW_SYSTEM_PROMPT, review_prompt, image_paths, image_labels,
                    response_format="json_object",
                )
            except StrataRequestError as exc:
                attempts += max(0, int(getattr(exc, "request_attempts", 1)) - 1)
                raise
            responses.append(response)
            attempts += max(0, int(response.get("request_attempts", 1)) - 1)
            latency += float(response.get("latency_seconds", 0.0))
            usage = _sum_usage(responses)
            try:
                if response.get("finish_reason") == "length":
                    raise ValueError("review response was truncated at the output-token limit")
                game_markdown, global_markdown = _validate_review_payload(
                    _parse_content(response.get("content")),
                    game_memory_max_chars=game_memory_max_chars,
                    global_memory_max_chars=global_memory_max_chars,
                )
                break
            except (DecisionValidationError, ValueError) as exc:
                repair_history.append({"attempt": repair_index + 1, "error": str(exc)})
                if repair_index == 0:
                    review_prompt = _review_repair_prompt(user_prompt, str(exc))
                    continue
                raise

        # Each destination is atomically replaced only after both outputs validate.
        memory_saved = save_game_memory(
            memory_store_dir, game_markdown, max_characters=game_memory_max_chars,
        )
        global_saved = save_global_memory(
            state_root, global_markdown, max_characters=global_memory_max_chars,
        )
        memory.record_review_checkpoint(
            game_id=game_id, current_step=tools.action_count,
            covered_event_count=covered_event_count,
        )
        digest = hashlib.sha256(
            (game_markdown + "\0" + global_markdown).encode("utf-8")
        ).hexdigest()
        row = {
            "schema": "v10.live_review/3", "status": "saved",
            "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
            "covered_transition_count": transition_count,
            "selected_image_labels": image_labels,
            "model": response.get("model", getattr(client, "model", None)),
            "request_attempts": attempts, "usage": _sum_usage(responses),
            "latency_seconds": round(latency, 6),
            "game_memory_path": str(view_path), "game_memory_updated": memory_saved,
            "game_memory_characters": len(game_markdown),
            "global_memory_path": str(global_memory_path(state_root)),
            "global_memory_updated": global_saved,
            "global_memory_characters": len(global_markdown),
            "verified_routes_supplied": len(verified_routes),
            "review_checkpoint_saved": True, "repair_history": repair_history,
            "result_sha256": digest,
        }
    except Exception as exc:
        attempts = max(attempts, int(getattr(exc, "request_attempts", 0)))
        if getattr(exc, "latency_seconds", None) is not None:
            latency = max(latency, float(exc.latency_seconds))
        row = {
            "schema": "v10.live_review/3", "status": "failed",
            "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
            "covered_transition_count": transition_count,
            "selected_image_labels": image_labels,
            "request_attempts": attempts, "usage": usage,
            "latency_seconds": round(latency, 6),
            "game_memory_saved": memory_saved, "global_memory_saved": global_saved,
            "review_checkpoint_saved": False, "repair_history": repair_history,
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
    *,
    starting_routine: bool = False,
    bootstrap_observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    records = tools.history.records()
    transitions = [
        row for row in records
        if row.get("type") == "transition"
        and int(row.get("reset_generation", 0)) == int(tools.reset_generation)
    ][-12:]
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
    # The saved model-authored Markdown is authoritative; never render ledger
    # records into action guidance here.
    memory = memory_store.view_path.read_text(encoding="utf-8") if memory_store.view_path.is_file() else ""
    global_memory = read_global_memory(memory_store_dir.parent.parent)
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
                "measurements": _compact_transition_measurements(row.get("measurements")),
                "level": row.get("level"),
                "levels_completed": row.get("levels_completed"), "game_state": row.get("game_state"),
            }
            for row in transitions
        ],
        "screen_sequence": [],
        "memory": memory,
        "repeated_outcomes": _repeated_outcome_summary(transitions, FrameStore(tools.run_dir)),
        "global_memory": global_memory,
        "budgets": {
            "actions_used": tools.action_count, "actions_remaining": max(0, tools.max_actions - tools.action_count),
            "resets_used": tools.reset_count, "resets_remaining": max(0, tools.max_resets - tools.reset_count),
            "plan_horizon": horizon,
        },
    }
    if starting_routine:
        context["bootstrap_observations"] = [
            {
                "action": model_action_name(row.get("action")),
                "measurements": _compact_transition_measurements(row.get("measurements")),
                "level": row.get("level"),
                "levels_completed": row.get("levels_completed"),
                "game_state": row.get("game_state"),
            }
            for row in bootstrap_observations
        ]
    frames = FrameStore(tools.run_dir)
    if starting_routine and bootstrap_observations:
        action_images, image_labels, screen_sequence = _action_screen_sequence(
            bootstrap_observations, current.get("frame_image_id"), frames,
            label_prefix="Bootstrap", current_label="restored initial screen after reset",
        )
    else:
        action_images, image_labels, screen_sequence = _action_screen_sequence(
            transitions, current.get("frame_image_id"), frames,
            label_prefix="Action context",
            current_label="Current actual game frame",
        )
    image_paths = action_images
    context["screen_sequence"] = screen_sequence
    system_prompt = skill_path.read_text(encoding="utf-8")
    model_context = dict(context)
    model_current_state = dict(context["current_state"])
    model_current_state["legal_actions"] = model_facing_action_specs(legal_actions)
    model_context["current_state"] = model_current_state
    model_context["recent_transitions"] = [
        {**row, "action": model_action_name(row.get("action"))}
        for row in context["recent_transitions"]
    ]
    start_instruction = (
        "The listed bootstrap observations came from a short exploratory sequence, and the game was then reset. "
        "The current image and current_state are the initial screen after that reset. Use the bootstrap observations and memory to choose "
        "a short exploratory action sequence that must start from this initial screen; do not continue from the bootstrap's final screen. "
        f"Return up to {horizon} legal lowercase directional actions. Do not reset. This instruction applies only to this one startup decision.\n\n"
        if starting_routine else ""
    )
    prompt = (
        start_instruction
        + f"Choose up to {horizon} legal actions from the current state, recent outcomes, and read-only memory. "
        "The supplied screen images are in chronological order. Labels identify the action before/after which each screen was observed; include every intermediate screen in that order. "
        "Use game memory first for this game's established mechanics and strategies. Use global memory only as reusable exploration guidance; it must not override game-specific observations. "
        "Prefer applicable demonstrated strategies, but follow the current image and outcomes if memory conflicts. Distinguish lasting mechanics from historical positions. "
        "Avoid repeating actions that had no visible effect in the same observed state and avoid cycles. Use one action for an uncertain experiment; use a longer plan only for understood behavior. "
        "Keep reasoning internal. "
        "Return only an actions list. Each action must contain only action and args. "
        "Do not include analysis, memory updates, predictions, goals, or narrative. "
        "If RESET is listed, use it alone and include args.justification.\n\n"
        "Required response shape:\n" + json.dumps(_decision_contract_description(), ensure_ascii=False, separators=(",", ":"))
        + "\n\nVisible context:\n" + json.dumps(model_context, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "system_prompt": system_prompt, "user_prompt": prompt, "context": context,
        "image_paths": image_paths, "image_labels": image_labels,
        "legal_action_specs": legal_actions, "visible_step": visible_step,
        "skill_sha256": hashlib.sha256(skill_path.read_bytes()).hexdigest(),
    }


def _standard_start_routine(legal_action_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the fixed up/left/down/right opening, limited to currently legal moves."""
    directions = ("up", "left", "down", "right")
    legal_directions = {
        str(model_action_name(spec.get("id"))) for spec in legal_action_specs
    }
    candidate = {
        "actions": [
            {"action": direction, "args": {}}
            for direction in directions if direction in legal_directions
        ]
    }
    if not candidate["actions"]:
        return []
    try:
        return validate_model_decision(candidate, legal_action_specs, max_actions=4)["actions"]
    except (DecisionValidationError, ActionError):
        # An unusual movement schema may require arguments; avoid fabricating them.
        accepted = []
        for direction in directions:
            if direction not in legal_directions:
                continue
            try:
                accepted.extend(validate_model_decision(
                    {"actions": [{"action": direction, "args": {}}]},
                    legal_action_specs, max_actions=1,
                )["actions"])
            except (DecisionValidationError, ActionError):
                continue
        return accepted


def _compact_transition_measurements(value: Any) -> dict[str, Any]:
    """Keep recent action context small while retaining directly useful effects."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    if "changed_pixels" in value:
        result["changed_pixels"] = value["changed_pixels"]
    displacement = value.get("candidate_displacement")
    if isinstance(displacement, Mapping) and isinstance(displacement.get("best"), Mapping):
        best = displacement["best"]
        result["best_displacement"] = {
            key: best[key] for key in ("dx", "dy", "score") if key in best
        }
    transitions = value.get("color_transitions")
    if isinstance(transitions, list):
        result["color_change_count"] = len(transitions)
    return result


def run_live(
    env: Any,
    run_dir: str | Path,
    client: Any,
    *,
    game_id: str,
    seed: int = 0,
    max_actions: int = 30,
    max_resets: int = 1,
    plan_horizon: int = 4,
    review_interval: int = 50,
    skill_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
    starting_routine: bool = True,
    game_memory_max_chars: int = MAX_MEMORY_MARKDOWN_CHARS,
    global_memory_max_chars: int = MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
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
    if int(game_memory_max_chars) < 1 or int(global_memory_max_chars) < 1:
        raise ValueError("memory Markdown limits must be positive")
    destination = Path(run_dir)
    source_skill = Path(skill_path) if skill_path is not None else shared_skill_path()
    run_id = uuid.uuid4().hex
    state_root = Path(state_dir) if state_dir is not None else default_state_dir()
    memory_store_dir = game_memory_dir(state_root, game_id)
    # Old versions kept the only copy of knowledge inside the disposable run directory.
    GameMemory.migrate_run_ledger(destination, memory_store_dir, game_id)
    _prepare_live_run_directory(destination)
    memory = GameMemory(destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    skill_sha256 = hashlib.sha256(source_skill.read_bytes()).hexdigest()
    write_json(destination / "run_manifest.json", {
        "schema": "v10.live_run/1", "run_id": run_id, "game_id": game_id,
        "seed": int(seed), "skill_sha256": skill_sha256,
        "persistent_memory_dir": str(memory_store_dir),
        "global_memory_path": str(global_memory_path(state_root)),
        "level_solutions_path": str(memory_store_dir / "level_solutions.jsonl"),
    })
    tools = GameTools(env, destination, max_actions=max_actions, max_resets=max_resets)
    current = tools.observe()
    memory.record_run_state(game_id, current, step=tools.action_count)
    solution_store = LevelSolutionStore(memory_store_dir / "level_solutions.jsonl", game_id)
    historical_run = Path(__file__).resolve().parents[2] / "runs" / "v10" / "ez01_20"
    if game_id == "ez01" and (historical_run / "evidence.jsonl").is_file():
        solution_store.import_successful_run(historical_run)
    initial_routine_pending = bool(starting_routine and tools.max_resets > 0)
    bootstrap_probe_active = bool(starting_routine and tools.max_resets > 0)
    bootstrap_plan_pending = False
    bootstrap_observations: list[dict[str, Any]] = []
    starter_queue: list[dict[str, Any]] = []
    starter_queue_source = "standard_start_routine"
    level_start_observation: Mapping[str, Any] = dict(current)
    level_path_edges: list[dict[str, Any]] = []
    cached_solution_plans = 0
    cached_solution_actions = 0
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
    consecutive_unchanged_actions = 0
    no_progress_actions = 0
    fresh_actions_since_review = 0
    recent_state_actions: deque[tuple[str, str, str]] = deque(maxlen=16)
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
            "cached_solution_plans": cached_solution_plans,
            "cached_solution_actions": cached_solution_actions,
            "available_level_solutions": len(solution_store.solutions()),
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

    def perform_triggered_review(reasons: Sequence[str]) -> None:
        nonlocal review_count, actions_since_review, fresh_actions_since_review
        nonlocal consecutive_unchanged_actions, review_model_calls, model_calls
        nonlocal review_latency, total_latency
        review_count += 1
        progress(f"review starting; triggers={list(reasons)}")
        result = _perform_review(
            tools, client, game_id=game_id, run_id=run_id,
            memory_store_dir=memory_store_dir, trigger_reasons=reasons,
            game_memory_max_chars=game_memory_max_chars,
            global_memory_max_chars=global_memory_max_chars,
        )
        actions_since_review = 0
        fresh_actions_since_review = 0
        consecutive_unchanged_actions = 0
        attempts = int(result["attempts"])
        review_model_calls += attempts
        model_calls += attempts
        elapsed = float(result["latency_seconds"])
        review_latency += elapsed
        total_latency += elapsed
        for key in total_usage:
            value = result["usage"].get(key)
            if value is not None:
                total_usage[key] += int(value)
                review_usage[key] += int(value)
        progress(f"review {result['row']['status']}")

    def learn_transition(
        before: Mapping[str, Any], action: Mapping[str, Any], result: Mapping[str, Any],
    ) -> dict[str, Any]:
        nonlocal level_start_observation, level_path_edges
        after = result["observation"]
        edge = solution_store.record_transition(
            before, str(action["action"]), action.get("args", {}), after,
            run_id=run_id,
        )
        if bootstrap_probe_active:
            return edge
        level_path_edges.append(edge)
        previous_completed = before.get("levels_completed")
        current_completed = after.get("levels_completed")
        completed = (
            previous_completed is not None and current_completed is not None
            and int(current_completed) > int(previous_completed)
        )
        won = str(after.get("game_state", "")).upper() == "WIN"
        if completed or won:
            solution_store.record_successful_trajectory(
                level_start_observation, level_path_edges, run_id=run_id,
                success="win" if won else "level_completed",
            )
            level_start_observation = dict(after)
            level_path_edges = []
        elif level_key(before) != level_key(after):
            # A nonterminal level discontinuity is not a completed route.
            level_start_observation = dict(after)
            level_path_edges = []
        return edge

    update_metrics()
    progress(f"started game={game_id}")
    try:
        while tools.action_count < tools.max_actions and not current.get("terminal"):
            responses: list[dict[str, Any]] = []
            validation_error: str | None = None
            cached_path_error: str | None = None
            decision: dict[str, Any] | None = None
            decision_source = "model"
            solution_edges: list[dict[str, Any]] = []
            starter_chunk = False

            if starter_queue:
                decision_source = starter_queue_source
                decision = {"actions": starter_queue[:int(plan_horizon)]}
                starter_chunk = True
                inputs = {
                    "skill_sha256": skill_sha256,
                    "image_labels": [],
                    "legal_action_specs": _public_actions(current.get("legal_actions", [])),
                }
            elif initial_routine_pending and solution_store.shortest_known_solution(current):
                # Exact, level-scoped success is better startup evidence than a probe.
                initial_routine_pending = False
                bootstrap_probe_active = False
            elif initial_routine_pending:
                initial_routine_pending = False
                inputs = {"skill_sha256": skill_sha256, "image_labels": [],
                          "legal_action_specs": _public_actions(current.get("legal_actions", []))}
                starter_queue = _standard_start_routine(inputs["legal_action_specs"])
                starter_queue_source = "bootstrap_probe"
                if starter_queue:
                    decision_source = starter_queue_source
                    decision = {"actions": starter_queue[:int(plan_horizon)]}
                    starter_chunk = True
            elif bootstrap_plan_pending:
                bootstrap_plan_pending = False
                inputs = _live_context(
                    tools, current, game_id, int(plan_horizon), source_skill,
                    memory_store_dir, run_id, int(memory_context_entries),
                    starting_routine=True, bootstrap_observations=bootstrap_observations,
                )
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
                        validation_error = f"bootstrap planning request failed: {exc}"
                        break
                    model_calls += max(0, int(response.get("request_attempts", 1)) - 1)
                    responses.append(response)
                    for key in total_usage:
                        value = response.get("usage", {}).get(key)
                        if value is not None:
                            total_usage[key] += int(value)
                            action_usage[key] += int(value)
                    total_latency += float(response.get("latency_seconds", 0.0))
                    try:
                        candidate = validate_model_decision(
                            _parse_content(response.get("content")),
                            inputs["legal_action_specs"], max_actions=plan_horizon,
                        )
                        directions = {"up", "left", "down", "right"}
                        if not candidate["actions"] or any(
                            str(model_action_name(item["action"])) not in directions
                            for item in candidate["actions"]
                        ):
                            raise DecisionValidationError(
                                "bootstrap plan must contain legal lowercase directional actions"
                            )
                        decision = candidate
                        starter_queue_source = "bootstrap_model_start"
                        break
                    except (DecisionValidationError, ActionError) as exc:
                        decision = None
                        validation_error = str(exc)
                        if attempt == 0:
                            inputs["user_prompt"] = _decision_repair_prompt(
                                inputs["user_prompt"], validation_error,
                            )
                if decision is not None:
                    starter_queue = list(decision["actions"])
                else:
                    starter_queue = _standard_start_routine(inputs["legal_action_specs"])
                    starter_queue_source = "bootstrap_start_fallback"
                if starter_queue:
                    decision_source = starter_queue_source
                    decision = {"actions": starter_queue[:int(plan_horizon)]}
                    starter_chunk = True

            if decision is None:
                known_path = solution_store.shortest_known_solution(current)
                solution_edges = known_path[:plan_horizon] if known_path else []
            if decision is None and solution_edges:
                inputs = {
                    "skill_sha256": skill_sha256,
                    "image_labels": [],
                    "legal_action_specs": current.get("legal_actions", []),
                }
                cached_decision = {"actions": [
                    {"action": model_action_name(edge["action"]), "args": edge.get("args", {})}
                    for edge in solution_edges
                ]}
                try:
                    decision = validate_model_decision(
                        cached_decision, inputs["legal_action_specs"], max_actions=plan_horizon,
                    )
                    decision_source = "verified_level_path"
                except (DecisionValidationError, ActionError) as exc:
                    # A stored edge may no longer be legal; ask the model from the current state.
                    cached_path_error = f"cached path is no longer valid: {exc}"
                    solution_edges = []
            if decision is None and not solution_edges:
                validation_error = None
                inputs = _live_context(
                    tools, current, game_id, int(plan_horizon), source_skill,
                    memory_store_dir, run_id, int(memory_context_entries),
                )
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
            if starter_chunk:
                progress(f"using {decision_source} ({len(starter_queue)} starter actions queued)")
            if decision is None:
                append_jsonl(decisions_path, {
                    "schema": "v10.live_decision/2", "status": "rejected",
                    "game_id": game_id,
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
                fallback_name = model_action_name(fallback["action"])
                progress(
                    f"decision invalid after repair; taking random legal fallback action "
                    f"{action_number}: {fallback_name}"
                )
                before_observation = current
                previous_level = current.get("level")
                previous_levels_completed = current.get("levels_completed")
                result = tools.act(
                    fallback["action"], fallback["args"], action_name=fallback_name,
                )
                current = result["observation"]
                observed_edge = learn_transition(before_observation, fallback, result)
                memory.record_run_state(game_id, current, step=tools.action_count)
                append_jsonl(decisions_path, {
                    "schema": "v10.live_fallback/2", "status": "executed",
                    "cause": "invalid_decision", "validation_error": validation_error,
                    "action": fallback_name, "args": fallback["args"],
                    "evidence_id": result["evidence"]["evidence_id"],
                })
                progress(
                    f"fallback action {tools.action_count}: {fallback_name} -> "
                    f"{current.get('game_state', '?')}; observation captured"
                )
                recent_state_actions.append((
                    str(observed_edge["source_state"]), str(fallback["action"]),
                    str(observed_edge["target_state"]),
                ))
                made_progress = (
                    current.get("levels_completed") != previous_levels_completed
                    or current.get("level") != previous_level
                    or str(current.get("game_state", "")).upper() == "WIN"
                )
                no_progress_actions = 0 if made_progress else no_progress_actions + 1
                actions_since_review += 1
                fresh_actions_since_review += 1
                changed_pixels = (result.get("measurements") or {}).get("changed_pixels")
                consecutive_unchanged_actions = consecutive_unchanged_actions + 1 if changed_pixels == 0 else 0
                fallback_reasons = []
                if previous_levels_completed is not None and current.get("levels_completed") is not None and int(current["levels_completed"]) > int(previous_levels_completed):
                    fallback_reasons.append("level_completed")
                if str(current.get("game_state", "")).upper() == "WIN":
                    fallback_reasons.append("win")
                repeated_states = Counter(
                    state for source, _action, target in recent_state_actions for state in (source, target)
                )
                repeated_pairs = Counter((source, action) for source, action, _target in recent_state_actions)
                if (
                    no_progress_actions >= 3 and fresh_actions_since_review >= 3
                    and (max(repeated_states.values(), default=0) >= 3 or max(repeated_pairs.values(), default=0) >= 2)
                ):
                    fallback_reasons.append("stagnation")
                if actions_since_review >= int(review_interval):
                    fallback_reasons.append("action_interval")
                if fallback_reasons and not bootstrap_probe_active:
                    perform_triggered_review(fallback_reasons)
                update_metrics()
                continue

            logged_decision = {
                **decision,
                "actions": [
                    {**item, "action": model_action_name(item["action"])}
                    for item in decision["actions"]
                ],
            }
            decision_row = {
                "schema": "v10.live_decision/2", "status": "accepted",
                "game_id": game_id, "decision_schema": DECISION_SCHEMA,
                "prompt_version": "v10.live_prompt/3", "skill_sha256": inputs["skill_sha256"],
                "model": responses[-1].get("model") if responses else None,
                "source": "verified_level_path" if solution_edges else decision_source,
                "cached_path_edge_ids": [edge["id"] for edge in solution_edges],
                "cached_path_error": cached_path_error,
                "input_image_labels": inputs["image_labels"], "decision": logged_decision,
                "repair_attempted": len(responses) > 1,
                "initial_validation_error": validation_error, "usage": _sum_usage(responses),
                "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                "latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in responses), 6),
            }
            if solution_edges:
                cached_solution_plans += 1
                progress(f"replaying verified known path ({len(solution_edges)} actions in this plan)")
            append_jsonl(decisions_path, decision_row)

            plan = decision
            progress(
                f"decision ready in {decision_row['latency_seconds']:.1f}s; "
                f"planned={[model_action_name(item['action']) for item in plan['actions']]}"
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
                starter_queue.clear()
                level_start_observation = dict(current)
                level_path_edges = []
                no_progress_actions = 0
                consecutive_unchanged_actions = 0
                recent_state_actions.clear()
                actions_since_review = 0
                fresh_actions_since_review = 0
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
            for action_index, action in enumerate(actions[:plan_horizon]):
                if tools.action_count >= tools.max_actions:
                    interruption = "action_budget_exhausted"
                    break
                expected_cached_edge = solution_edges[action_index] if solution_edges else None
                if expected_cached_edge and state_fingerprint(current) != expected_cached_edge["source_state"]:
                    interruption = "solution_diverged"
                    progress("cached path state mismatch; returning to model decisions")
                    break
                legal_now = {item.get("id") for item in _public_actions(current.get("legal_actions", []))}
                if action["action"] not in legal_now:
                    interruption = "planned_action_no_longer_legal"
                    if starter_chunk:
                        starter_queue.clear()
                    break
                previous_level = current.get("level")
                previous_levels_completed = current.get("levels_completed")
                prediction_memory = GameMemory(
                    destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
                )
                expected_pattern = prediction_memory.expected_observation(game_id, current, action["action"])
                before_observation = current
                try:
                    next_number = tools.action_count + 1
                    action_name = model_action_name(action["action"])
                    progress(f"taking action {next_number}: {action_name}")
                    result = tools.act(
                        action["action"], action.get("args", {}), action_name=action_name,
                    )
                except (ActionError, BudgetExceeded) as exc:
                    interruption = "action_error" if isinstance(exc, ActionError) else "action_budget_exhausted"
                    if starter_chunk:
                        starter_queue.clear()
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_plan_result/2", "status": "action_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    break
                current = result["observation"]
                observed_edge = learn_transition(before_observation, action, result)
                if starter_chunk and starter_queue:
                    starter_queue.pop(0)
                if expected_cached_edge:
                    cached_solution_actions += 1
                    if observed_edge["target_state"] != expected_cached_edge["target_state"]:
                        interruption = "solution_diverged"
                        progress("cached path observation differed; returning to model decisions")
                source_state = str(observed_edge["source_state"])
                target_state = str(observed_edge["target_state"])
                recent_state_actions.append((source_state, str(action["action"]), target_state))
                made_progress = (
                    current.get("levels_completed") != previous_levels_completed
                    or current.get("level") != previous_level
                    or str(current.get("game_state", "")).upper() == "WIN"
                )
                no_progress_actions = 0 if made_progress else no_progress_actions + 1
                changed_pixels = (result.get("measurements") or {}).get("changed_pixels")
                if changed_pixels == 0:
                    consecutive_unchanged_actions += 1
                else:
                    consecutive_unchanged_actions = 0
                memory = GameMemory(
                    destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
                )
                memory.record_run_state(game_id, current, step=tools.action_count)
                progress(
                    f"action {tools.action_count}: {model_action_name(action['action'])} -> "
                    f"{current.get('game_state', '?')}"
                )
                executed_action = {
                    "action": model_action_name(action["action"]),
                    "evidence_id": result["evidence"]["evidence_id"],
                }
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
                        "action": action_name,
                        "expected": expected_pattern["expected_observation"],
                        "actual": actual, "verdict": verdict, "reason": verdict_row.get("reason"),
                        "reset_generation": tools.reset_generation,
                    })
                    memory.record_pattern_prediction(
                        expected_pattern["pattern_id"], verdict, step=tools.action_count,
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
                fresh_actions_since_review += 1
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
                if str(current.get("game_state", "")).upper() == "WIN":
                    trigger_reasons.append("win")
                repeated_state_count = Counter(
                    state for source, _action, target in recent_state_actions for state in (source, target)
                )
                repeated_pair_count = Counter(
                    (source, action_name) for source, action_name, _target in recent_state_actions
                )
                repeated_without_progress = (
                    no_progress_actions >= 3
                    and fresh_actions_since_review >= 3
                    and (
                        max(repeated_state_count.values(), default=0) >= 3
                        or max(repeated_pair_count.values(), default=0) >= 2
                    )
                )
                if repeated_without_progress:
                    trigger_reasons.append("stagnation")
                if actions_since_review >= int(review_interval):
                    trigger_reasons.append("action_interval")
                if trigger_reasons and not bootstrap_probe_active:
                    perform_triggered_review(trigger_reasons)
                if current.get("terminal"):
                    interruption = "terminal_state"
                elif current.get("level") != previous_level or current.get("levels_completed") != previous_levels_completed:
                    interruption = "level_change"
                elif trigger_reasons:
                    interruption = "review"
                if starter_chunk and interruption in {
                    "terminal_state", "level_change", "planned_action_no_longer_legal",
                    "action_error", "solution_diverged",
                }:
                    starter_queue.clear()
                if interruption == "solution_diverged":
                    break
                if interruption:
                    break
            if interruption:
                interrupted_plans += 1
            append_jsonl(decisions_path, {
                "schema": "v10.live_plan_result/2", "status": "interrupted" if interruption else "completed",
                "executed": executed, "interrupt_reason": interruption,
            })
            if bootstrap_probe_active and (not starter_queue or interruption is not None):
                bootstrap_generation = tools.reset_generation
                bootstrap_observations = [
                    dict(row) for row in tools.history.records()
                    if row.get("type") == "transition"
                    and int(row.get("reset_generation", -1)) == int(bootstrap_generation)
                ]
                try:
                    reset_result = tools.reset(
                        "Return to the initial screen after collecting startup bootstrap observations."
                    )
                except BudgetExceeded:
                    stop_reason = "bootstrap_reset_budget_exhausted"
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_bootstrap_reset/1", "status": stop_reason,
                        "required_reset_budget": 1,
                    })
                    update_metrics()
                    break
                current = reset_result["observation"]
                bootstrap_probe_active = False
                bootstrap_plan_pending = True
                starter_queue.clear()
                level_start_observation = dict(current)
                level_path_edges = []
                no_progress_actions = 0
                consecutive_unchanged_actions = 0
                recent_state_actions.clear()
                append_jsonl(decisions_path, {
                    "schema": "v10.live_bootstrap_reset/1", "status": "reset",
                    "bootstrap_transitions": [row.get("evidence_id") for row in bootstrap_observations],
                    "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                    "reset_generation": tools.reset_generation,
                })
                progress("bootstrap observations captured; reset to initial screen")
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
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        append_jsonl(decisions_path, {
            "schema": "v10.live_interruption/1", "stop_reason": stop_reason,
            "actions_completed": tools.action_count,
            "detail": "Interrupted by user; no review or route finalization was started.",
        })
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
