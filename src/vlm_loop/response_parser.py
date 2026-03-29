from __future__ import annotations

import json
from typing import Any

from .action_schema import canonicalize_action_name
from .models import ModelAnalysisResult


ANALYSIS_STAGE_IDS = {"start_poi", "update_poi"}
ACTION_STAGE_IDS = {"start_poi_actions", "update_poi_actions"}
REVIEW_STAGE_IDS = {"episode_review"}


def parse_backend_contract_object(raw_wrapper: Any, *, backend: str) -> dict[str, Any]:
    wrapper = _coerce_wrapper_object(raw_wrapper)
    response_payload = wrapper.get("response") if isinstance(wrapper.get("response"), dict) else wrapper
    extraction_path, inner_payload = _extract_backend_inner_payload(response_payload, backend=backend)
    if inner_payload is None:
        raise ValueError(f"missing backend contract payload at {extraction_path} for backend={backend}")
    if isinstance(inner_payload, dict):
        return inner_payload
    if not isinstance(inner_payload, str):
        raise ValueError(
            f"backend contract payload at {extraction_path} for backend={backend} must be string or object, "
            f"got {type(inner_payload).__name__}"
        )
    try:
        return _extract_json_object(inner_payload)
    except ValueError as exc:
        preview = inner_payload[:500]
        raise ValueError(
            "failed to parse backend contract payload: "
            f"backend={backend} extraction_path={extraction_path} payload_preview={preview!r}"
        ) from exc


def validate_stage_contract(payload: dict[str, Any], *, stage_id: str) -> dict[str, Any]:
    if stage_id in ANALYSIS_STAGE_IDS:
        payload = _unwrap_stage_payload(payload, stage_id=stage_id, required_keys={"sprite", "poi"})
        sprite = payload.get("sprite")
        poi = payload.get("poi")
        if not isinstance(sprite, dict):
            raise ValueError(f"{stage_id} must return exactly one sprite object")
        if not isinstance(poi, dict):
            raise ValueError(f"{stage_id} must return exactly one poi object")
        return payload
    if stage_id in ACTION_STAGE_IDS:
        payload = _unwrap_stage_payload(payload, stage_id=stage_id, required_keys={"actions"})
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError(f"{stage_id} must return exactly one actions list")
        return payload
    if stage_id in REVIEW_STAGE_IDS:
        payload = _unwrap_stage_payload(
            payload,
            stage_id=stage_id,
            required_keys={"outcome_review"},
            allow_outcome_review_passthrough=True,
        )
        review = payload.get("outcome_review") if isinstance(payload.get("outcome_review"), dict) else payload
        if not isinstance(review, dict):
            raise ValueError(f"{stage_id} must return outcome_review object")
        for key in ("likely_goal_or_mechanic", "key_observation", "failure_reason_or_success_reason"):
            if not isinstance(review.get(key), str):
                raise ValueError(f"{stage_id} must return string field: {key}")
        next_run_hint = review.get("next_run_hint")
        if next_run_hint is not None:
            if not isinstance(next_run_hint, dict):
                raise ValueError(f"{stage_id} next_run_hint must be an object when present")
            required_keys = {"sprite_description", "target", "hud", "avoid"}
            missing = required_keys - set(next_run_hint)
            if missing:
                raise ValueError(f"{stage_id} next_run_hint missing keys: {sorted(missing)}")
            for key in sorted(required_keys):
                if not isinstance(next_run_hint.get(key), str):
                    raise ValueError(f"{stage_id} next_run_hint.{key} must be a string")
        return review
    raise ValueError(f"unsupported stage_id for contract validation: {stage_id}")


def _unwrap_stage_payload(
    payload: dict[str, Any],
    *,
    stage_id: str,
    required_keys: set[str],
    allow_outcome_review_passthrough: bool = False,
) -> dict[str, Any]:
    if required_keys.issubset(payload.keys()):
        return payload

    candidates: list[dict[str, Any]] = []
    for key in (stage_id, "result", "output", "analysis", "review", "data", "contract"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    if allow_outcome_review_passthrough:
        outcome_review = payload.get("outcome_review")
        if isinstance(outcome_review, dict):
            return payload

    dict_values = [value for value in payload.values() if isinstance(value, dict)]
    if len(dict_values) == 1:
        candidates.append(dict_values[0])

    for candidate in candidates:
        if required_keys.issubset(candidate.keys()):
            return candidate
        if allow_outcome_review_passthrough and "outcome_review" in required_keys:
            outcome_review = candidate.get("outcome_review")
            if isinstance(outcome_review, dict):
                return candidate
    return payload


def extract_action_sequence(
    payload: dict[str, Any],
    *,
    field: str,
    allowed_actions: list[str],
    min_length: int,
    max_length: int,
) -> tuple[list[str] | None, str | None]:
    candidate = _lookup_path(payload, field)
    if not isinstance(candidate, list):
        return None, "actions_missing_or_not_list"
    if len(candidate) < min_length:
        return None, "too_short"
    if len(candidate) > max_length:
        return None, "too_long"
    normalized = [canonicalize_action_name(item) for item in candidate]
    allowed = set(allowed_actions)
    if any(action not in allowed for action in normalized):
        return None, "unknown_action"
    return normalized, None


def build_model_analysis_result(
    *,
    stage_outputs: dict[str, Any],
    raw_text: str,
) -> ModelAnalysisResult:
    latest_analysis = _latest_analysis_output(stage_outputs)
    return ModelAnalysisResult(
        sprite_summary=_sprite_summary(latest_analysis.get("sprite")),
        selected_poi_summary=_poi_summary(latest_analysis.get("poi")),
        pattern_match_summary=_pattern_match_summary(latest_analysis.get("pattern_matches")),
        screen_change_summary=_screen_change_summary(latest_analysis),
        ui_summary=_ui_summary(latest_analysis.get("ui_elements")),
        proposed_actions_start=_action_summary(stage_outputs.get("start_poi_actions")),
        proposed_actions_update=_action_summary(stage_outputs.get("update_poi_actions")),
        raw_text=raw_text,
    )


def _latest_analysis_output(stage_outputs: dict[str, Any]) -> dict[str, Any]:
    for stage_id in ("update_poi", "start_poi"):
        value = stage_outputs.get(stage_id)
        if isinstance(value, dict):
            return value
    return {}


def _sprite_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts = [str(value.get("description") or "").strip()]
    for key in ("start_position", "current_position"):
        token = str(value.get(key) or "").strip()
        if token:
            parts.append(token)
    return " | ".join(part for part in parts if part)


def _poi_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return " | ".join(
        part
        for part in (
            str(value.get("name") or "").strip(),
            str(value.get("description") or "").strip(),
            str(value.get("location") or "").strip(),
        )
        if part
    )


def _pattern_match_summary(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    tokens: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        hud_description = str(item.get("hud_description") or "").strip()
        world_description = str(item.get("world_object_description") or "").strip()
        location = str(item.get("world_object_location") or "").strip()
        text = " -> ".join(part for part in (hud_description, world_description, location) if part)
        if text:
            tokens.append(text)
    return "; ".join(tokens)


def _screen_change_summary(payload: dict[str, Any]) -> str:
    value = payload.get("screen_change_summary")
    if isinstance(value, dict):
        return str(value.get("description") or "").strip()
    pattern = payload.get("screen_changes")
    if isinstance(pattern, list):
        return "; ".join(
            str(item.get("description") or "").strip()
            for item in pattern
            if isinstance(item, dict) and str(item.get("description") or "").strip()
        )
    return ""


def _ui_summary(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "; ".join(
        " | ".join(
            part
            for part in (
                str(item.get("description") or "").strip(),
                str(item.get("location") or "").strip(),
            )
            if part
        )
        for item in value
        if isinstance(item, dict) and (item.get("description") or item.get("location"))
    )


def _action_summary(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    actions = value.get("actions")
    if not isinstance(actions, list):
        return []
    return [canonicalize_action_name(item) for item in actions]


def _lookup_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for token in [part for part in path.split(".") if part]:
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict):
                        return payload
        start = text.find("{", start + 1)
    partial_payload = _extract_partial_root_object(text)
    if partial_payload is not None:
        return partial_payload
    raise ValueError("no JSON object found in model response")


def _extract_partial_root_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    if start < 0:
        return None
    text = raw_text[start:]
    decoder = json.JSONDecoder()
    index = 1
    result: dict[str, Any] = {}
    parsed_any = False

    while index < len(text):
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text):
            break
        if text[index] == "}":
            return result if parsed_any else None
        try:
            key_obj, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if not isinstance(key_obj, str):
            break
        index = next_index
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index >= len(text) or text[index] != ":":
            break
        index += 1
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index >= len(text):
            break
        try:
            value_obj, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if key_obj not in result:
            result[key_obj] = value_obj
        parsed_any = True
        index = next_index

    return result if parsed_any else None


def _coerce_wrapper_object(raw_wrapper: Any) -> dict[str, Any]:
    if isinstance(raw_wrapper, dict):
        return raw_wrapper
    if isinstance(raw_wrapper, str):
        payload = json.loads(raw_wrapper)
        if isinstance(payload, dict):
            return payload
    raise ValueError("backend wrapper must be a JSON object or a JSON string containing an object")


def _extract_backend_inner_payload(response_payload: dict[str, Any], *, backend: str) -> tuple[str, Any]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "ollama":
        message = response_payload.get("message")
        if not isinstance(message, dict):
            return "response.message.content", None
        return "response.message.content", message.get("content")
    if normalized_backend == "vllm":
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return "response.choices[0].message.content", None
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return "response.choices[0].message.content", None
        message = first_choice.get("message")
        if not isinstance(message, dict):
            return "response.choices[0].message.content", None
        content = message.get("content")
        if isinstance(content, list):
            text_chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "response.choices[0].message.content", "\n".join(chunk for chunk in text_chunks if chunk)
        return "response.choices[0].message.content", content
    raise ValueError(f"unsupported backend for contract extraction: {backend}")
