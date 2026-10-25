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
    validate_model_decision,
)
from .frames import DELTA_KIND, IMAGE_KIND, FrameStore
from .runlog import append_jsonl, write_json
from .strata_client import StrataRequestError
from .tools.game import ActionError, BudgetExceeded, GameTools
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
    transitions = [row for row in records if row.get("type") == "transition"][-horizon:]
    first = next((row for row in records if row.get("type") == "state"), None)
    visible_ids = {str(current["evidence_id"])} if current.get("evidence_id") else set()
    if first and first.get("evidence_id"):
        visible_ids.add(str(first["evidence_id"]))
    visible_ids.update(str(row["evidence_id"]) for row in transitions if row.get("evidence_id"))
    relevant_ids = {str(row["evidence_id"]) for row in transitions}
    prior_audits = [
        {
            "evidence_id": row.get("evidence_id"), "prediction_id": row.get("prediction_id"),
            "prediction_evidence_id": row.get("prediction_evidence_id"),
            "transition_evidence_id": row.get("transition_evidence_id"), "verdict": row.get("verdict"),
            "details": row.get("details"),
        }
        for row in records
        if row.get("type") == "audit" and row.get("kind") == "prediction_result"
        and row.get("transition_evidence_id") in relevant_ids
    ]
    visible_predictions = {
        str(row.get("prediction", {}).get("prediction_id")) for row in transitions
        if isinstance(row.get("prediction"), Mapping) and row.get("prediction", {}).get("prediction_id")
    }
    visible_prediction_rows = [
        row for row in records
        if row.get("type") == "audit" and row.get("kind") == "prediction"
        and isinstance(row.get("prediction"), Mapping)
        and row.get("prediction", {}).get("prediction_id") in visible_predictions
    ]
    for row in [*visible_prediction_rows, *prior_audits]:
        if row.get("evidence_id"):
            visible_ids.add(str(row["evidence_id"]))
        for field in ("prediction_evidence_id", "transition_evidence_id"):
            if row.get(field):
                visible_ids.add(str(row[field]))
    legal_actions = _public_actions(current.get("legal_actions", []))
    visible_step = int(current.get("step", max((int(row.get("step", 0)) for row in records), default=0)))
    memory_store = GameMemory(tools.run_dir, store_dir=memory_store_dir, run_id=run_id)
    memory = memory_store.compact(max_entries=memory_context_entries, through_step=visible_step)
    for entry in memory["entries"]:
        visible_ids.update(str(value) for value in entry.get("evidence_ids", []))
    context = {
        "analysis_schema": DECISION_SCHEMA,
        "game_id": game_id,
        "initial_state_evidence_id": first.get("evidence_id") if first else None,
        "current_evidence_id": current.get("evidence_id"),
        "citable_evidence_ids": sorted(visible_ids),
        "visible_prediction_ids": sorted(visible_predictions),
        "frame_ids_are_image_handles_not_evidence": True,
        "current_state": {
            "game_state": current.get("game_state"), "level": current.get("level"),
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "terminal": current.get("terminal", False), "frame_shape": list(current["frame"].shape),
            "legal_actions": legal_actions,
        },
        "recent_transitions": [
            {
                "evidence_id": row["evidence_id"], "action": row.get("action"), "args": row.get("args", {}),
                "pre_frame_id": row.get("pre_frame_id"), "post_frame_id": row.get("post_frame_id"),
                "prediction_evidence_id": row.get("prediction_evidence_id"),
                "measurements": row.get("measurements"), "level": row.get("level"),
                "levels_completed": row.get("levels_completed"), "game_state": row.get("game_state"),
                "prediction": row.get("prediction"),
            }
            for row in transitions
        ],
        "prediction_audits": prior_audits,
        "memory": memory,
        "budgets": {
            "actions_used": tools.action_count, "actions_remaining": max(0, tools.max_actions - tools.action_count),
            "resets_used": tools.reset_count, "resets_remaining": max(0, tools.max_resets - tools.reset_count),
            "plan_horizon": horizon,
        },
    }
    image_ids: list[tuple[str, str]] = []
    if first and first.get("frame_image_id"):
        image_ids.append(("Initial state image (frame handle, not an evidence ID)", str(first["frame_image_id"])))
    for row in transitions:
        if row.get("delta_image_id"):
            image_ids.append((f"Delta visualization (not a game frame) after {row.get('action')} ({row['evidence_id']})", str(row["delta_image_id"])))
    if current.get("frame_image_id"):
        image_ids.append(("Current state image (actual game frame; frame handle, not an evidence ID)", str(current["frame_image_id"])))
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
    prompt = (
        "Analyze only the visible current state, recent transitions, audits, and memory below. "
        "Return compact JSON matching the contract. Keep each claim brief and omit unsupported list entries. "
        "Do not invent mechanics or outcomes. Recommend up to four legal actions when the sequence remains safe; "
        "otherwise recommend one informative action. Give concise predictions. "
        "Continue by default; reset only with an explicit evidence-based reason. "
        "Cite only IDs listed in citable_evidence_ids; F-prefixed frame handles are not evidence IDs. "
        "Delta images are overlays of changed pixels, not game states; use the current-state image for object positions.\n\n"
        "Decision contract:\n" + json.dumps(_decision_contract_description(), ensure_ascii=False, separators=(",", ":"))
        + "\n\nVisible context:\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "system_prompt": system_prompt, "user_prompt": prompt, "context": context,
        "image_paths": image_paths, "image_labels": image_labels,
        "visible_evidence_ids": visible_ids, "visible_prediction_ids": visible_predictions,
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
    skill_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
) -> dict[str, Any]:
    """Run a bounded live episode, persisting evidence before and after actions."""
    if not 1 <= int(plan_horizon) <= 4:
        raise ValueError("plan_horizon must be between 1 and 4")
    if int(max_actions) < 1 or int(max_resets) < 0:
        raise ValueError("action budget must be positive and reset budget non-negative")
    if int(memory_context_entries) < 0:
        raise ValueError("memory_context_entries must be non-negative")
    destination = Path(run_dir)
    source_skill = Path(skill_path) if skill_path is not None else Path(__file__).with_name("SKILL.md")
    run_id = uuid.uuid4().hex
    memory_store_dir = game_memory_dir(state_dir or default_state_dir(), game_id)
    # Old versions kept the only copy of knowledge inside the disposable run directory.
    GameMemory.migrate_run_ledger(destination, memory_store_dir, game_id)
    _prepare_live_run_directory(destination)
    memory = GameMemory(destination, store_dir=memory_store_dir, run_id=run_id)
    if not memory.view_path.exists():
        memory.write_view()
    skill_sha256 = hashlib.sha256(source_skill.read_bytes()).hexdigest()
    write_json(destination / "run_manifest.json", {
        "schema": "v10.live_run/1", "run_id": run_id, "game_id": game_id,
        "seed": int(seed), "skill_sha256": skill_sha256,
        "persistent_memory_dir": str(memory_store_dir),
    })
    tools = GameTools(env, destination, max_actions=max_actions, max_resets=max_resets)
    current = tools.observe()
    model_calls = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_latency = 0.0
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
            "schema": "v10.live_metrics/1", "game_id": game_id, "model": getattr(client, "model", None),
            "run_id": run_id, "seed": int(seed),
            "actions": tools.action_count, "action_budget": tools.max_actions,
            "resets": tools.reset_count, "reset_budget": tools.max_resets,
            "model_calls": model_calls, "usage": dict(total_usage),
            "latency_seconds": round(total_latency, 6), "elapsed_seconds": elapsed,
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "game_state": state, "terminal": terminal, "won": state == "WIN",
            "interrupted_plans": interrupted_plans, "stop_reason": stop_reason,
            "memory_entries": GameMemory(
                destination, store_dir=memory_store_dir, run_id=run_id,
            ).read().get("count", 0),
            "memory_context_entries": int(memory_context_entries),
            "prediction_verdicts": {
                verdict: sum(
                    row.get("type") == "audit" and row.get("kind") == "prediction_result"
                    and row.get("verdict") == verdict
                    for row in tools.history.records()
                )
                for verdict in ("matched", "failed", "inconclusive")
            },
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
                total_latency += float(response.get("latency_seconds", 0.0))
                try:
                    decision = validate_model_decision(
                        _parse_content(response.get("content")), inputs["legal_action_specs"],
                        inputs["visible_evidence_ids"], inputs["visible_prediction_ids"],
                    )
                    used_ids = {
                        row.get("prediction", {}).get("prediction_id")
                        for row in tools.history.records()
                        if row.get("type") == "audit" and row.get("kind") == "prediction"
                        and isinstance(row.get("prediction"), Mapping)
                    }
                    duplicate = next((item["prediction_id"] for item in decision["decision"]["actions"]
                                      if item["prediction_id"] in used_ids), None)
                    if duplicate:
                        raise DecisionValidationError(f"prediction_id already used: {duplicate}")
                    break
                except (DecisionValidationError, ActionError) as exc:
                    validation_error = str(exc)
                    if attempt == 0:
                        inputs["user_prompt"] = _decision_repair_prompt(
                            inputs["user_prompt"], validation_error, inputs["visible_evidence_ids"],
                        )
            if decision is None:
                append_jsonl(decisions_path, {
                    "schema": "v10.live_decision/1", "status": "rejected",
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
                        "schema": "v10.live_fallback/1", "status": "unavailable",
                        "reason": "no legal action has arguments that can be generated safely",
                        "validation_error": validation_error,
                    })
                    break
                action_number = tools.action_count + 1
                fallback_prediction_id = f"FALLBACK_{run_id}_{action_number}"
                progress(
                    f"decision invalid after repair; taking random legal fallback action "
                    f"{action_number}: {fallback['action']}"
                )
                result = tools.act(
                    fallback["action"], fallback["args"],
                    prediction_id=fallback_prediction_id,
                    expected_observation="No reliable model prediction; observe the result of this fallback action.",
                    measurable_prediction=None,
                    unmeasurable_reason="The model decision failed validation; this random fallback has no model prediction.",
                )
                current = result["observation"]
                append_jsonl(decisions_path, {
                    "schema": "v10.live_fallback/1", "status": "executed",
                    "cause": "invalid_decision", "validation_error": validation_error,
                    "action": fallback["action"], "args": fallback["args"],
                    "prediction_id": fallback_prediction_id,
                    "evidence_id": result["evidence"]["evidence_id"],
                    "prediction_verdict": result["prediction_audit"].get("verdict"),
                })
                progress(
                    f"fallback action {tools.action_count}: {fallback['action']} -> "
                    f"{current.get('game_state', '?')}; observation captured"
                )
                update_metrics()
                continue

            memory_result = GameMemory(
                destination, store_dir=memory_store_dir, run_id=run_id,
            ).propose_update(
                decision["memory_patch"], known_evidence_ids=inputs["visible_evidence_ids"],
                current_step=inputs["visible_step"],
            )
            decision_row = {
                "schema": "v10.live_decision/1", "status": "accepted",
                "game_id": game_id, "decision_schema": DECISION_SCHEMA,
                "prompt_version": "v10.live_prompt/1", "skill_sha256": inputs["skill_sha256"],
                "model": responses[-1].get("model"),
                "visible_evidence_ids": sorted(inputs["visible_evidence_ids"]),
                "input_image_labels": inputs["image_labels"], "decision": decision,
                "memory_updates": memory_result, "repair_attempted": len(responses) > 1,
                "initial_validation_error": validation_error, "usage": _sum_usage(responses),
                "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                "latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in responses), 6),
            }
            append_jsonl(decisions_path, decision_row)

            plan = decision["decision"]
            progress(
                f"decision ready in {decision_row['latency_seconds']:.1f}s; "
                f"planned={[item['action'] for item in plan['actions']]}"
            )
            if plan["continue_or_reset"] == "reset":
                try:
                    reset_result = tools.reset(plan["reset_justification"])
                except BudgetExceeded:
                    stop_reason = "reset_budget_exhausted"
                    append_jsonl(decisions_path, {"schema": "v10.live_plan_result/1", "status": stop_reason})
                    break
                current = reset_result["observation"]
                progress("reset executed")
                interrupted_plans += 1
                append_jsonl(decisions_path, {
                    "schema": "v10.live_plan_result/1", "status": "reset",
                    "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                })
                update_metrics()
                continue
            actions = plan["actions"]
            if not actions:
                stop_reason = "model_returned_no_actions"
                append_jsonl(decisions_path, {"schema": "v10.live_plan_result/1", "status": stop_reason})
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
                try:
                    next_number = tools.action_count + 1
                    progress(f"taking action {next_number}: {action['action']}")
                    result = tools.act(
                        action["action"], action.get("args", {}),
                        prediction_id=action["prediction_id"],
                        expected_observation=action["expected_observation"],
                        measurable_prediction=action.get("measurable_prediction"),
                        unmeasurable_reason=action.get("unmeasurable_reason"),
                    )
                except (ActionError, BudgetExceeded) as exc:
                    interruption = "action_error" if isinstance(exc, ActionError) else "action_budget_exhausted"
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_plan_result/1", "status": "action_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    break
                current = result["observation"]
                progress(
                    f"action {tools.action_count}: {action['action']} -> "
                    f"{current.get('game_state', '?')}; "
                    f"prediction={result['prediction_audit'].get('verdict', '?')}"
                )
                executed.append({
                    "prediction_id": action["prediction_id"],
                    "evidence_id": result["evidence"]["evidence_id"],
                    "verdict": result["prediction_audit"].get("verdict"),
                })
                if current.get("terminal"):
                    interruption = "terminal_state"
                elif current.get("level") != previous_level or current.get("levels_completed") != previous_levels_completed:
                    interruption = "level_change"
                elif result["prediction_audit"].get("verdict") == "failed":
                    interruption = "prediction_mismatch"
                if interruption:
                    break
            if interruption:
                interrupted_plans += 1
            append_jsonl(decisions_path, {
                "schema": "v10.live_plan_result/1", "status": "interrupted" if interruption else "completed",
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
