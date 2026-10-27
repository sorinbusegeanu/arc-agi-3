"""Bounded game adapter with durable state and transition evidence.

The environment contract is intentionally small: ``observe()`` returns a mapping
with a pixel grid in ``frame`` (or ``image``), legal action descriptors in
``legal_actions``, and optional state metadata. ``act(action, args)`` performs
exactly one action. It may return intermediate frames in ``frames``. ``reset()``
is only called through this adapter and requires an explicit justification.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..frames import DELTA_KIND, IMAGE_KIND, FrameStore, to_frame_array
from ..tools.history import History


PREDICTION_OPERATORS = ("eq", "ne", "gt", "gte", "lt", "lte", "between")
PREDICTION_PATHS: dict[str, str] = {
    "measurements.schema": "string",
    "measurements.changed_pixels": "number",
    "measurements.total_pixels": "number",
    "measurements.change_ratio": "number",
    "measurements.region": "array",
    "measurements.changed_box": "array",
    "measurements.color_transitions": "array",
    "measurements.transitions_truncated": "boolean",
    "measurements.candidate_displacement.schema": "string",
    "measurements.candidate_displacement.background": "number",
    "measurements.candidate_displacement.search_radius": "number",
    "measurements.candidate_displacement.appeared_pixels": "number",
    "measurements.candidate_displacement.disappeared_pixels": "number",
    "measurements.candidate_displacement.best.dx": "number",
    "measurements.candidate_displacement.best.dy": "number",
    "measurements.candidate_displacement.best.score": "number",
    "measurements.candidate_displacement.second.dx": "number",
    "measurements.candidate_displacement.second.dy": "number",
    "measurements.candidate_displacement.second.score": "number",
    "game_state": "string",
    "level": "number",
    "levels_completed": "number",
    "win_levels": "number",
    "terminal": "boolean",
}
_LEGACY_MEASUREMENT_FIELDS = {
    path.removeprefix("measurements."): path for path in PREDICTION_PATHS
    if path.startswith("measurements.")
}


def normalize_measurable_prediction(value: Any) -> Any:
    """Convert legacy flat pixel measurements to the canonical predicate form."""
    if value is None or not isinstance(value, Mapping):
        return value
    if set(value) == {"all"}:
        return dict(value)

    measurements = value
    if set(value) == {"pixel_diff"} and isinstance(value.get("pixel_diff"), Mapping):
        measurements = value["pixel_diff"]

    predicates: list[dict[str, Any]] = []

    def flatten(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), child)
            return
        path = _LEGACY_MEASUREMENT_FIELDS.get(prefix)
        if path is None:
            predicates.append({"path": f"measurements.{prefix}", "op": "eq", "value": item})
        else:
            predicates.append({"path": path, "op": "eq", "value": item})

    for key, child in measurements.items():
        flatten(str(key), child)
    return {"all": predicates}


def validate_measurable_prediction(value: Any, field_name: str = "action.measurable_prediction") -> None:
    """Validate canonical all-of comparison predicates against known outputs."""
    if not isinstance(value, Mapping) or set(value) != {"all"}:
        raise ActionError(
            f"{field_name} must use {{'all': [{{'path','op','value'}}]}}; "
            "flat pixel_diff fields are legacy input only"
        )
    predicates = value.get("all")
    if not isinstance(predicates, list) or not predicates:
        raise ActionError(f"{field_name}.all must be a non-empty list")
    for index, predicate in enumerate(predicates):
        name = f"{field_name}.all[{index}]"
        if not isinstance(predicate, Mapping) or set(predicate) != {"path", "op", "value"}:
            raise ActionError(f"{name} must contain exactly path, op, and value")
        path = predicate["path"]
        if not isinstance(path, str) or path not in PREDICTION_PATHS:
            raise ActionError(
                f"{name}.path is unsupported: {path}; allowed paths: {', '.join(PREDICTION_PATHS)}"
            )
        op = predicate["op"]
        if op not in PREDICTION_OPERATORS:
            raise ActionError(f"{name}.op must be one of {', '.join(PREDICTION_OPERATORS)}")
        kind = PREDICTION_PATHS[path]
        expected = predicate["value"]
        if op in {"gt", "gte", "lt", "lte", "between"} and kind != "number":
            raise ActionError(f"{name}.op {op} requires a numeric measurement path")
        if op == "between":
            if (
                not isinstance(expected, list) or len(expected) != 2
                or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in expected)
                or expected[0] > expected[1]
            ):
                raise ActionError(f"{name}.value for between must be [minimum, maximum] in ascending order")
        elif kind == "number" and (
            isinstance(expected, bool) or not isinstance(expected, (int, float))
        ):
            raise ActionError(f"{name}.value must be numeric for {path}")
        elif kind == "string" and not isinstance(expected, str):
            raise ActionError(f"{name}.value must be a string for {path}")
        elif kind == "boolean" and not isinstance(expected, bool):
            raise ActionError(f"{name}.value must be a boolean for {path}")
        elif kind == "array" and not isinstance(expected, list):
            raise ActionError(f"{name}.value must be an array for {path}")
        try:
            json.dumps(expected, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ActionError(f"{name}.value must be finite JSON data") from exc


class ActionError(ValueError):
    """An action is malformed, illegal, or cannot safely be executed."""


class BudgetExceeded(RuntimeError):
    """The requested environment operation exceeds the configured run budget."""


class ActionBudget:
    """Shared cap for live actions in a run."""

    def __init__(self, max_actions: int, *, used: int = 0) -> None:
        self.max_actions = int(max_actions)
        self.used = int(used)
        self.categories: dict[str, int] = {}

    @property
    def remaining(self) -> int:
        return max(0, self.max_actions - self.used)

    def ensure_capacity(self, count: int = 1) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("action reservation must be non-negative")
        if count > self.remaining:
            raise BudgetExceeded(
                f"action budget exhausted ({self.used}+{count}>{self.max_actions})"
            )

    def consume(self, category: str = "live") -> None:
        self.ensure_capacity()
        self.used += 1
        self.categories[category] = self.categories.get(category, 0) + 1


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"environment {name}() must return a mapping-like value")


def _frame_from(value: Mapping[str, Any], name: str = "observation") -> np.ndarray:
    frame = value.get("frame", value.get("image"))
    if frame is None:
        raise ValueError(f"{name} must include a frame or image")
    return to_frame_array(frame, name)


def _action_table(legal_actions: Any) -> dict[str, dict[str, Any]]:
    if isinstance(legal_actions, Mapping):
        items = legal_actions.items()
        table = {}
        for key, descriptor in items:
            if isinstance(descriptor, Mapping):
                item = dict(descriptor)
                item.setdefault("id", str(key))
            else:
                item = {"id": str(key)}
            item["_environment_id"] = key
            table[str(key)] = item
        return table
    if isinstance(legal_actions, Sequence) and not isinstance(legal_actions, (str, bytes)):
        table = {}
        for descriptor in legal_actions:
            if isinstance(descriptor, str):
                action_id, item = descriptor, {"id": descriptor}
            elif isinstance(descriptor, int) and not isinstance(descriptor, bool):
                action_id, item = str(descriptor), {"id": str(descriptor), "_environment_id": descriptor}
            elif isinstance(descriptor, Mapping):
                item = dict(descriptor)
                action_id = item.get("id", item.get("action"))
            else:
                raise ValueError("legal_actions entries must be strings or mappings")
            if not isinstance(action_id, str) or not action_id:
                if isinstance(action_id, int) and not isinstance(action_id, bool):
                    action_id = str(action_id)
                else:
                    raise ValueError("each legal action descriptor needs a non-empty id")
            item.setdefault("_environment_id", action_id)
            table[action_id] = item
        return table
    raise ValueError("legal_actions must be a mapping or sequence")


def _validate_args(action_id: str, args: Any, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise ActionError("action args must be an object")
    result = dict(args)

    def is_json_value(value: Any) -> bool:
        if value is None or isinstance(value, (str, bool, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(is_json_value(item) for item in value)
        if isinstance(value, Mapping):
            return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
        return False

    try:
        # Validate the complete argument tree before consulting the schema or
        # allowing GameTools to call the environment. This rejects nested NaN,
        # infinities, non-string object keys, and non-JSON Python values too.
        if not is_json_value(result):
            raise ValueError("unsupported JSON value")
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ActionError(f"{action_id} args must be finite JSON data") from exc
    schema = descriptor.get("parameters", descriptor.get("args_schema"))
    if not isinstance(schema, Mapping):
        return result
    required = schema.get("required", ())
    properties = schema.get("properties", {})
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise ValueError(f"invalid argument schema for {action_id}: required must be a list")
    if not isinstance(properties, Mapping):
        raise ValueError(f"invalid argument schema for {action_id}: properties must be an object")
    missing = [key for key in required if key not in result]
    if missing:
        raise ActionError(f"{action_id} requires argument(s): {', '.join(map(str, missing))}")
    if schema.get("additionalProperties") is False:
        extra = sorted(set(result) - set(properties))
        if extra:
            raise ActionError(f"{action_id} does not accept argument: {extra[0]}")
    for key, spec in properties.items():
        if key not in result or not isinstance(spec, Mapping):
            continue
        value = result[key]
        kind = spec.get("type")
        if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ActionError(f"{action_id}.{key} must be an integer")
        if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ActionError(f"{action_id}.{key} must be a number")
        if kind == "string" and not isinstance(value, str):
            raise ActionError(f"{action_id}.{key} must be a string")
        if "enum" in spec and value not in spec["enum"]:
            raise ActionError(f"{action_id}.{key} is not an allowed value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                raise ActionError(f"{action_id}.{key} is below its minimum")
            if "maximum" in spec and value > spec["maximum"]:
                raise ActionError(f"{action_id}.{key} is above its maximum")
    return result


def validate_decision(payload: Any, legal_actions: Any, max_actions: int = 4) -> dict[str, Any]:
    """Validate an action-only plan against the current legal action schemas."""
    if not isinstance(payload, Mapping):
        raise ActionError("response must be an object")
    if set(payload) != {"actions"}:
        raise ActionError("response must contain only an actions list")
    actions = payload["actions"]
    if not isinstance(actions, list) or len(actions) > int(max_actions):
        raise ActionError(f"actions must contain 0 to {int(max_actions)} actions")
    table = _action_table(legal_actions)
    validated = []
    for index, item in enumerate(actions):
        if not isinstance(item, Mapping):
            raise ActionError(f"actions[{index}] must be an object")
        if set(item) - {"action", "args"}:
            raise ActionError(f"actions[{index}] may contain only action and args")
        action_id = item.get("action")
        if not isinstance(action_id, str) or action_id not in table:
            raise ActionError(f"actions[{index}] has illegal action: {action_id}")
        validated.append({
            "action": action_id,
            "args": _validate_args(action_id, item.get("args", {}), table[action_id]),
        })
    if any(item["action"] == "RESET" for item in validated) and len(validated) != 1:
        raise ActionError("RESET must be the only action in the response")
    return {"actions": validated}


def evaluate_prediction(
    expected: Any,
    actual: Any,
    unmeasurable_reason: str | None = None,
) -> dict[str, Any]:
    """Score explicit path predicates; prose-only expectations remain unscorable."""
    if expected is None:
        details = {"verdict": "inconclusive", "reason": "prediction marked unmeasurable"}
        if unmeasurable_reason:
            details["unmeasurable_reason"] = unmeasurable_reason
        return details
    if not isinstance(actual, Mapping):
        return {"verdict": "inconclusive", "reason": "actual measurements are unavailable"}
    expected = normalize_measurable_prediction(expected)
    try:
        validate_measurable_prediction(expected)
    except ActionError as exc:
        return {"verdict": "inconclusive", "reason": f"invalid prediction: {exc}"}

    results: list[dict[str, Any]] = []
    unavailable: list[str] = []
    failed = False
    for predicate in expected["all"]:
        current: Any = actual
        for component in predicate["path"].split("."):
            if not isinstance(current, Mapping) or component not in current:
                current = None
                break
            current = current[component]
        if current is None:
            unavailable.append(predicate["path"])
            continue

        target = predicate["value"]
        op = predicate["op"]
        kind = PREDICTION_PATHS[predicate["path"]]
        if kind == "number" and (isinstance(current, bool) or not isinstance(current, (int, float))):
            unavailable.append(predicate["path"])
            continue
        if kind == "string" and not isinstance(current, str):
            unavailable.append(predicate["path"])
            continue
        if kind == "boolean" and not isinstance(current, bool):
            unavailable.append(predicate["path"])
            continue
        if kind == "array" and not isinstance(current, list):
            unavailable.append(predicate["path"])
            continue
        if op in {"eq", "ne"}:
            equal = current == target if kind == "number" else (
                type(current) is type(target) and current == target
            )
            passed = equal if op == "eq" else not equal
        elif op == "gt":
            passed = current > target
        elif op == "gte":
            passed = current >= target
        elif op == "lt":
            passed = current < target
        elif op == "lte":
            passed = current <= target
        else:  # between is inclusive at both ends.
            passed = target[0] <= current <= target[1]
        failed = failed or not passed
        results.append({
            "path": predicate["path"], "op": op, "expected": target,
            "actual": current, "matched": passed,
        })

    if failed:
        return {"verdict": "failed", "compared": results, "unavailable_paths": unavailable}
    if unavailable:
        return {
            "verdict": "inconclusive", "reason": "predicted measurement path is unavailable",
            "unavailable_paths": unavailable, "compared": results,
        }
    return {"verdict": "matched", "compared": results}


class GameTools:
    """Environment adapter with an action budget and append-only evidence."""

    def __init__(
        self,
        env: Any,
        run_dir: Path | str,
        max_actions: int = 1000,
        *,
        action_budget: ActionBudget | None = None,
        action_budget_category: str = "live",
    ) -> None:
        if not callable(getattr(env, "observe", None)) or not callable(getattr(env, "act", None)):
            raise TypeError("env must implement observe() and act(action, args)")
        self.env = env
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames = FrameStore(self.run_dir)
        self.history = History(self.run_dir)
        self.max_actions = int(max_actions)
        self.action_count = sum(record.get("type") == "transition" for record in self.history.records())
        self.action_budget = action_budget or ActionBudget(self.max_actions, used=self.action_count)
        if self.action_budget.max_actions != self.max_actions:
            raise ValueError("shared action budget must use the GameTools max_actions limit")
        self.action_budget_category = str(action_budget_category)
        self.reset_count = sum(record.get("type") == "reset" for record in self.history.records())
        self.reset_generation = max((int(record.get("reset_generation", 0)) for record in self.history.records()), default=0)
        self._current: dict[str, Any] | None = None

    @property
    def supports_reset(self) -> bool:
        return callable(getattr(self.env, "reset", None))

    @property
    def actions_remaining(self) -> int:
        return self.action_budget.remaining

    def _capture(self, raw: Any) -> dict[str, Any]:
        observation = _mapping(raw, "observe")
        array = _frame_from(observation)
        step = self.history.next_step()
        frame_id = self.frames.put(array, step=step)
        from PIL import Image
        from vlm_v2.frame_writer import observation_to_image

        image = observation_to_image(array)
        image = image.resize((image.width * 10, image.height * 10), resample=Image.Resampling.NEAREST)
        image_id = f"{frame_id}_rgb"
        self.frames.put_image(np.asarray(image, dtype=np.uint8), image_id, source=frame_id, step=step, kind=IMAGE_KIND)
        return {
            **observation, "frame": array, "frame_id": frame_id,
            "frame_image_id": image_id, "legal_actions": observation.get("legal_actions", []),
        }

    def observe(self) -> dict[str, Any]:
        if self._current is not None:
            return dict(self._current)
        current = self._capture(self.env.observe())
        evidence = self.history.append({
            "type": "state", "frame_id": current["frame_id"],
            "frame_image_id": current["frame_image_id"], "legal_actions": current["legal_actions"],
            "level": current.get("level"), "levels_completed": current.get("levels_completed"),
            "terminal": current.get("terminal"), "game_state": current.get("game_state"),
            "reset_generation": self.reset_generation,
            "game_state": current.get("game_state"), "win_levels": current.get("win_levels"),
            "game_id": current.get("game_id"),
        })
        current["evidence_id"] = evidence["evidence_id"]
        self._current = current
        return dict(current)

    def act(
        self,
        action: str,
        args: Mapping[str, Any] | None = None,
        *,
        action_name: str | None = None,
    ) -> dict[str, Any]:
        self.action_budget.ensure_capacity()
        before = self._current or self.observe()
        table = _action_table(before.get("legal_actions", []))
        if action not in table:
            raise ActionError(f"illegal action: {action}")
        safe_args = _validate_args(action, args or {}, table[action])
        pre_step = self.history.next_step()
        try:
            # An exception after dispatch may still have changed environment state.
            self.action_budget.consume(self.action_budget_category)
            result_raw = self.env.act(table[action].get("_environment_id", action), safe_args)
        except Exception as exc:
            self.history.append({
                "type": "audit", "kind": "action_error",
                "action": str(action_name) if action_name else action,
                "error": f"{type(exc).__name__}: {exc}", "reset_generation": self.reset_generation,
            })
            raise
        result = dict(result_raw) if isinstance(result_raw, Mapping) else {"result": result_raw}
        result_frames = result.get("frames", [])
        if result_frames is not None:
            if not isinstance(result_frames, Sequence) or isinstance(result_frames, (str, bytes)):
                raise ValueError("env.act result frames must be a sequence")
            from PIL import Image
            from vlm_v2.frame_writer import observation_to_image

            for intermediate in result_frames:
                image = intermediate.get("frame", intermediate.get("image")) if isinstance(intermediate, Mapping) else intermediate
                grid = to_frame_array(image, "intermediate frame")
                frame_id = self.frames.put(grid, source="intermediate", step=pre_step)
                result.setdefault("intermediate_frame_ids", []).append(frame_id)
                rendered = observation_to_image(grid)
                rendered = rendered.resize(
                    (rendered.width * 10, rendered.height * 10),
                    resample=Image.Resampling.NEAREST,
                )
                image_id = f"{frame_id}_rgb"
                self.frames.put_image(
                    np.asarray(rendered, dtype=np.uint8), image_id,
                    source=frame_id, step=pre_step, kind=IMAGE_KIND,
                )
                result.setdefault("intermediate_frame_image_ids", []).append(image_id)
        after = self._capture(self.env.observe())
        before_frame, after_frame = before["frame"], after["frame"]
        from .vision import candidate_displacement, pixel_diff
        measurements = pixel_diff(before_frame, after_frame)
        measurements["candidate_displacement"] = candidate_displacement(before_frame, after_frame)
        from PIL import Image
        from .vision import delta_image

        delta = Image.fromarray(delta_image(before_frame, after_frame), mode="RGB")
        delta = delta.resize((delta.width * 10, delta.height * 10), resample=Image.Resampling.NEAREST)
        transition_id = self.history.next_evidence_id()
        delta_image_id = f"D{transition_id}"
        self.frames.put_image(
            np.asarray(delta, dtype=np.uint8), delta_image_id,
            source=transition_id, step=pre_step, kind=DELTA_KIND,
        )
        self.action_count += 1
        evidence = self.history.append({
            "evidence_id": transition_id,
            "type": "transition",
            "action": str(action_name) if action_name else action,
            "args": safe_args,
            "pre_frame_id": before["frame_id"], "post_frame_id": after["frame_id"],
            "pre_frame_image_id": before["frame_image_id"], "post_frame_image_id": after["frame_image_id"],
            "pre_evidence_id": before.get("evidence_id"),
            "delta_image_id": delta_image_id,
            "result": {key: value for key, value in result.items() if key not in {"frames", "frame", "image"}},
            "measurements": measurements, "legal_actions": after["legal_actions"],
            "level": after.get("level"), "levels_completed": after.get("levels_completed"),
            "pre_level": before.get("level"), "post_level": after.get("level"),
            "terminal": after.get("terminal"), "game_state": after.get("game_state"),
            "reset_generation": self.reset_generation,
        })
        after["evidence_id"] = evidence["evidence_id"]
        self._current = after
        return {
            "evidence": evidence,
            "observation": dict(after), "result": result,
            "measurements": measurements, "action_count": self.action_count,
        }

    def reset(self, justification: str) -> dict[str, Any]:
        if not isinstance(justification, str) or not justification.strip():
            raise ActionError("reset requires a non-empty justification")
        reset_method = getattr(self.env, "reset", None)
        if not callable(reset_method):
            raise ActionError("environment does not support reset")
        raw = reset_method()
        self.reset_count += 1
        self.reset_generation += 1
        self._current = None
        current = self._capture(raw if isinstance(raw, Mapping) and ("frame" in raw or "image" in raw) else self.env.observe())
        evidence = self.history.append({
            "type": "reset", "justification": justification.strip(), "frame_id": current["frame_id"],
            "frame_image_id": current["frame_image_id"],
            "legal_actions": current["legal_actions"], "level": current.get("level"),
            "levels_completed": current.get("levels_completed"), "terminal": current.get("terminal"),
            "game_state": current.get("game_state"), "win_levels": current.get("win_levels"),
            "game_id": current.get("game_id"),
            "reset_generation": self.reset_generation,
        })
        current["evidence_id"] = evidence["evidence_id"]
        self._current = current
        return {"observation": dict(current), "evidence": evidence, "reset_count": self.reset_count}


__all__ = ["ActionBudget", "ActionError", "BudgetExceeded", "GameTools", "evaluate_prediction", "validate_decision"]
