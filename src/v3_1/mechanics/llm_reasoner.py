from __future__ import annotations

import json
from typing import Any

from v3_1.llm.local_adapter_base import LocalLLMAdapter, LocalLLMRequest
from v3_1.mechanics.hypothesis_types import HypothesisBundle
from v3_1.mechanics.llm_prompt_builder import build_focused_llm_payload
from v3_1.mechanics.llm_schema import LLMHypothesisInput, LLMHypothesisOutput
from v3_1.runtime.hypothesis_gating import llm_skip_reason


def _preset_for_task(*, task_role: str, hypothesis_config) -> dict:
    if task_role == "ambiguity_resolution":
        return dict(getattr(hypothesis_config, "llm_mode_ambiguity_resolver", {}) or {})
    if task_role == "experiment_suggestion":
        return dict(getattr(hypothesis_config, "llm_mode_experiment_suggester", {}) or {})
    return dict(getattr(hypothesis_config, "llm_mode_hypothesis_generator", {}) or {})


def _prompt_mode_for_task(task_role: str) -> str:
    if task_role == "ambiguity_resolution":
        return "resolve_contradiction"
    if task_role == "experiment_suggestion":
        return "suggest_experiment"
    return "hypothesis_for_exit"


def _approx_tokens(text: str) -> int:
    return max(1, int((len(text) / 4.0) + 0.999))


def _safe_metadata(
    *,
    reason: str,
    attempted: bool,
    succeeded: bool,
    adapter,
    response=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "disabled": not succeeded,
        "reason": str(reason or ""),
        "llm_call_attempted": bool(attempted),
        "llm_call_succeeded": bool(succeeded),
        "llm_adapter_name": type(adapter).__name__ if adapter is not None else None,
        "llm_model_name": str(getattr(response, "model_name", "") or ""),
        "llm_latency_ms": int(getattr(response, "latency_ms", 0) or 0),
    }
    if response is not None:
        metadata["error_code"] = getattr(response, "error_code", None)
        metadata["error_message"] = getattr(response, "error_message", None)
        metadata["raw_text"] = getattr(response, "raw_text", "")
    metadata.update(dict(extra or {}))
    return metadata


def _focused_limits(hypothesis_config) -> dict[str, Any]:
    return {
        "max_nodes": int(getattr(hypothesis_config, "llm_prompt_max_nodes", 16) or 16),
        "max_edges": int(getattr(hypothesis_config, "llm_prompt_max_edges", 24) or 24),
        "max_paths": int(getattr(hypothesis_config, "llm_prompt_max_paths", 5) or 5),
        "max_contradictions": int(getattr(hypothesis_config, "llm_prompt_max_contradictions", 5) or 5),
        "max_exit_attempts": int(getattr(hypothesis_config, "llm_prompt_max_exit_attempts", 5) or 5),
        "max_pattern_relations": int(getattr(hypothesis_config, "llm_prompt_max_pattern_relations", 5) or 5),
        "max_allowed_node_ids": int(getattr(hypothesis_config, "llm_prompt_max_allowed_node_ids", 24) or 24),
    }


def _infer_query_target(
    *,
    mechanic_graph_snapshot: dict,
    deterministic_hypothesis_bundle: HypothesisBundle,
    contradictions: list[dict],
    prompt_mode: str,
) -> dict[str, Any]:
    graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    nodes_by_id = dict(graph_state.get("nodes_by_id", {}))
    if str(prompt_mode) == "resolve_contradiction":
        ranked = sorted(
            [dict(row) for row in list(contradictions or [])],
            key=lambda row: (-int(row.get("count", row.get("contradiction_count", 1)) or 1), str(row.get("contradiction_id") or row.get("id") or "")),
        )
        for row in ranked:
            for key in ("affected_ids", "affected_node_ids", "node_ids"):
                for node_id in list(row.get(key, []) or []):
                    if str(node_id) in nodes_by_id:
                        return {"node_id": str(node_id), "target_kind": str(dict(nodes_by_id[str(node_id)]).get("node_kind") or "")}
    if str(prompt_mode) == "suggest_experiment":
        best = sorted(
            list(getattr(deterministic_hypothesis_bundle, "path_proposals", ()) or []),
            key=lambda row: (
                -len(list(getattr(row, "contradiction_refs", ()) or [])),
                -float(getattr(row, "confidence", 0.0) or 0.0),
            ),
        )
        for proposal in best:
            for node_id in list(getattr(proposal, "metadata", {}).get("node_ids", ()) or []):
                if str(node_id) in nodes_by_id:
                    return {"node_id": str(node_id), "target_kind": str(dict(nodes_by_id[str(node_id)]).get("node_kind") or "")}
    exits = [dict(row) for row in list(nodes_by_id.values()) if str(dict(row).get("node_kind") or "") == "exit"]
    exits.sort(key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0)))
    if exits:
        return {"node_id": str(exits[0].get("node_id") or ""), "target_kind": "exit"}
    if nodes_by_id:
        first = next(iter(nodes_by_id.values()))
        return {"node_id": str(dict(first).get("node_id") or ""), "target_kind": str(dict(first).get("node_kind") or "")}
    return {"node_id": "", "target_kind": "unknown"}


def _focused_payload_from_args(
    *,
    hypothesis_input: LLMHypothesisInput | None,
    mechanic_graph_snapshot: dict | None,
    deterministic_hypothesis_bundle: HypothesisBundle | None,
    exit_attempt_summary: list[dict] | None,
    contradictions: list[dict] | None,
    pattern_relation_summary: list[dict] | None,
    hypothesis_config,
    task_role: str,
) -> tuple[dict[str, Any], bool]:
    prompt_mode = _prompt_mode_for_task(task_role)
    if mechanic_graph_snapshot is None or deterministic_hypothesis_bundle is None:
        payload = {
            "system_instruction": str(getattr(hypothesis_input, "system_instruction", "") or ""),
            "prompt_mode": prompt_mode,
            "query_target": {"node_id": "", "target_kind": "unknown"},
            "graph_nodes": list(getattr(hypothesis_input, "graph_nodes", ()) or ()),
            "graph_edges": list(getattr(hypothesis_input, "graph_edges", ()) or ()),
            "top_deterministic_edges": list(getattr(hypothesis_input, "top_deterministic_edges", ()) or ()),
            "top_deterministic_paths": list(getattr(hypothesis_input, "top_deterministic_paths", ()) or ()),
            "open_questions": list(getattr(hypothesis_input, "open_questions", ()) or ()),
            "contradictions": list(getattr(hypothesis_input, "contradictions", ()) or ()),
            "exit_attempts": list(getattr(hypothesis_input, "exit_attempts", ()) or ()),
            "pattern_relations": list(getattr(hypothesis_input, "pattern_relations", ()) or ()),
            "allowed_node_ids": list(getattr(hypothesis_input, "allowed_node_ids", ()) or ()),
            "allowed_edge_kinds": list(getattr(hypothesis_input, "allowed_edge_kinds", ()) or ()),
            "allowed_path_kinds": list(getattr(hypothesis_input, "allowed_path_kinds", ()) or ()),
            "payload_section_counts": {
                "graph_nodes": len(list(getattr(hypothesis_input, "graph_nodes", ()) or ())),
                "graph_edges": len(list(getattr(hypothesis_input, "graph_edges", ()) or ())),
                "top_deterministic_paths": len(list(getattr(hypothesis_input, "top_deterministic_paths", ()) or ())),
                "contradictions": len(list(getattr(hypothesis_input, "contradictions", ()) or ())),
                "exit_attempts": len(list(getattr(hypothesis_input, "exit_attempts", ()) or ())),
                "pattern_relations": len(list(getattr(hypothesis_input, "pattern_relations", ()) or ())),
                "allowed_node_ids": len(list(getattr(hypothesis_input, "allowed_node_ids", ()) or ())),
            },
        }
        return payload, True
    limits = _focused_limits(hypothesis_config)
    query_target = _infer_query_target(
        mechanic_graph_snapshot=mechanic_graph_snapshot,
        deterministic_hypothesis_bundle=deterministic_hypothesis_bundle,
        contradictions=list(contradictions or []),
        prompt_mode=prompt_mode,
    )
    payload = build_focused_llm_payload(
        mechanic_graph_snapshot=mechanic_graph_snapshot,
        deterministic_hypothesis_bundle=deterministic_hypothesis_bundle,
        exit_attempt_summary=list(exit_attempt_summary or []),
        contradictions=list(contradictions or []),
        query_target=query_target,
        pattern_relation_summary=list(pattern_relation_summary or []),
        prompt_mode=prompt_mode,
        **limits,
    )
    return payload, False


def _trim_payload(payload: dict[str, Any], *, trim_strategy: str) -> tuple[dict[str, Any], bool]:
    trimmed = dict(payload)
    trim_applied = False
    optional_sections = ["pattern_relations", "exit_attempts", "contradictions"]
    if str(trim_strategy) == "aggressive":
        optional_sections = ["pattern_relations", "exit_attempts", "contradictions", "top_deterministic_paths"]
    for section in optional_sections:
        if list(trimmed.get(section, ()) or []):
            trimmed[section] = ()
            trim_applied = True
            break
    if not trim_applied:
        for section, target in (
            ("graph_edges", max(4, int(len(list(trimmed.get("graph_edges", ()) or ())) / 2))),
            ("graph_nodes", max(4, int(len(list(trimmed.get("graph_nodes", ()) or ())) / 2))),
            ("top_deterministic_paths", max(2, int(len(list(trimmed.get("top_deterministic_paths", ()) or ())) / 2))),
            ("allowed_node_ids", max(8, int(len(list(trimmed.get("allowed_node_ids", ()) or ())) / 2))),
        ):
            values = list(trimmed.get(section, ()) or ())
            if len(values) > target:
                trimmed[section] = tuple(values[:target])
                trim_applied = True
                break
    section_counts = dict(trimmed.get("payload_section_counts", {}) or {})
    for key in ("graph_nodes", "graph_edges", "top_deterministic_paths", "contradictions", "exit_attempts", "pattern_relations", "allowed_node_ids"):
        section_counts[key] = len(list(trimmed.get(key, ()) or ()))
    trimmed["payload_section_counts"] = section_counts
    return trimmed, trim_applied


def generate_llm_hypotheses(
    hypothesis_input: LLMHypothesisInput | None,
    *,
    adapter: LocalLLMAdapter | None = None,
    hypothesis_config: object | None = None,
    task_role: str = "mechanic_hypothesis_generation",
    session_id: str = "",
    round_id: int = 0,
    max_output_tokens: int = 1200,
    temperature: float = 0.3,
    mechanic_graph_snapshot: dict | None = None,
    deterministic_hypothesis_bundle: HypothesisBundle | None = None,
    exit_attempt_summary: list[dict] | None = None,
    contradictions: list[dict] | None = None,
    pattern_relation_summary: list[dict] | None = None,
) -> LLMHypothesisOutput:
    if adapter is None:
        return LLMHypothesisOutput(metadata=_safe_metadata(reason="no_adapter", attempted=False, succeeded=False, adapter=None))
    prompt_payload, used_compatibility_builder = _focused_payload_from_args(
        hypothesis_input=hypothesis_input,
        mechanic_graph_snapshot=mechanic_graph_snapshot,
        deterministic_hypothesis_bundle=deterministic_hypothesis_bundle,
        exit_attempt_summary=exit_attempt_summary,
        contradictions=contradictions,
        pattern_relation_summary=pattern_relation_summary,
        hypothesis_config=hypothesis_config,
        task_role=task_role,
    )
    prompt_mode = str(prompt_payload.get("prompt_mode") or _prompt_mode_for_task(task_role))
    trim_strategy = str(getattr(hypothesis_config, "llm_prompt_trim_strategy", "optional_then_limits") or "optional_then_limits")
    max_chars = int(getattr(hypothesis_config, "llm_prompt_max_chars", 9000) or 9000)
    max_approx_tokens = int(getattr(hypothesis_config, "llm_prompt_max_approx_tokens", 2200) or 2200)
    serialized_user_payload = json.dumps(
        {key: value for key, value in dict(prompt_payload).items() if key != "system_instruction"},
        sort_keys=True,
    )
    system_instruction = str(prompt_payload.get("system_instruction") or "")
    prompt_char_count = len(serialized_user_payload)
    prompt_approx_token_count = _approx_tokens(serialized_user_payload)
    prompt_trim_applied = False
    if prompt_char_count > max_chars or prompt_approx_token_count > max_approx_tokens:
        prompt_payload, prompt_trim_applied = _trim_payload(prompt_payload, trim_strategy=trim_strategy)
        serialized_user_payload = json.dumps(
            {key: value for key, value in dict(prompt_payload).items() if key != "system_instruction"},
            sort_keys=True,
        )
        prompt_char_count = len(serialized_user_payload)
        prompt_approx_token_count = _approx_tokens(serialized_user_payload)
    if prompt_char_count > max_chars or prompt_approx_token_count > max_approx_tokens:
        skip_reason = llm_skip_reason(
            config=type("HypothesisConfigHolder", (), {"hypothesis_generation": hypothesis_config})(),
            mechanic_graph_snapshot=mechanic_graph_snapshot,
            deterministic_bundle=deterministic_hypothesis_bundle,
            repeated_failures=0,
            contradiction_level=len(list(contradictions or [])),
            deterministic_tied=False,
            graph_ambiguity=0.0,
            current_call_count=0,
            prompt_too_large_after_trimming=True,
        ) or "prompt_budget_exceeded"
        return LLMHypothesisOutput(
            metadata=_safe_metadata(
                reason=skip_reason,
                attempted=False,
                succeeded=False,
                adapter=adapter,
                extra={
                    "parameter_preset_used": str(task_role),
                    "task_role": str(task_role),
                    "schema_valid_response": False,
                    "prompt_char_count": prompt_char_count,
                    "prompt_approx_token_count": prompt_approx_token_count,
                    "prompt_trim_applied": prompt_trim_applied,
                    "prompt_mode": prompt_mode,
                    "query_target_id": str(dict(prompt_payload.get("query_target", {}) or {}).get("node_id") or ""),
                    "payload_section_counts": dict(prompt_payload.get("payload_section_counts", {}) or {}),
                    "used_compatibility_builder": used_compatibility_builder,
                },
            )
        )
    payload = dict(prompt_payload)
    payload["prompt_trim_applied"] = bool(prompt_trim_applied)
    emit_raw_debug = bool(getattr(hypothesis_config, "llm_emit_raw_debug", False)) if hypothesis_config is not None else False
    preset = _preset_for_task(task_role=task_role, hypothesis_config=hypothesis_config)
    request_temperature = float(preset.get("temperature", temperature) or temperature)
    request_top_p = float(preset.get("top_p", getattr(hypothesis_config, "llm_top_p", 0.8) if hypothesis_config is not None else 0.8) or 0.8)
    request_top_k = int(preset.get("top_k", getattr(hypothesis_config, "llm_top_k", 20) if hypothesis_config is not None else 20) or 20)
    request_presence_penalty = float(preset.get("presence_penalty", getattr(hypothesis_config, "llm_presence_penalty", 0.0) if hypothesis_config is not None else 0.0) or 0.0)
    request_max_output_tokens = int(preset.get("max_output_tokens", max_output_tokens) or max_output_tokens)
    request_repetition_penalty = float(getattr(hypothesis_config, "llm_repetition_penalty", 1.0) if hypothesis_config is not None else 1.0)
    response = adapter.generate_structured_json(
        LocalLLMRequest(
            payload_json=payload,
            schema_name="llm_hypothesis_output",
            schema_version="v1",
            max_output_tokens=request_max_output_tokens,
            temperature=request_temperature,
            top_p=request_top_p,
            top_k=request_top_k,
            presence_penalty=request_presence_penalty,
            repetition_penalty=request_repetition_penalty,
            enable_thinking=False,
            stream=False,
            round_id=int(round_id),
            session_id=str(session_id),
        )
    )
    response_metadata = {
        "parameter_preset_used": str(task_role),
        "task_role": str(task_role),
        "prompt_char_count": prompt_char_count,
        "prompt_approx_token_count": prompt_approx_token_count,
        "prompt_trim_applied": prompt_trim_applied,
        "prompt_mode": prompt_mode,
        "query_target_id": str(dict(prompt_payload.get("query_target", {}) or {}).get("node_id") or ""),
        "payload_section_counts": dict(prompt_payload.get("payload_section_counts", {}) or {}),
        "temperature": request_temperature,
        "top_p": request_top_p,
        "top_k": request_top_k,
        "presence_penalty": request_presence_penalty,
        "repetition_penalty": request_repetition_penalty,
        "max_output_tokens": request_max_output_tokens,
        "enable_thinking": False,
        "stream": False,
        "used_compatibility_builder": used_compatibility_builder,
    }
    if emit_raw_debug:
        response_metadata["debug_prompt_payload"] = payload
        response_metadata["debug_system_instruction"] = system_instruction
        response_metadata["debug_prompt_string"] = serialized_user_payload
    if not bool(response.ok) or not isinstance(response.parsed_json, dict):
        return LLMHypothesisOutput(
            metadata=_safe_metadata(
                reason=str(response.error_code or "llm_call_failed"),
                attempted=True,
                succeeded=False,
                adapter=adapter,
                response=response,
                extra={**response_metadata, "schema_valid_response": False},
            )
        )
    raw = response.parsed_json
    adapter_metadata = dict(raw.get("metadata", {}) or {})
    return LLMHypothesisOutput(
        edge_proposals=tuple(raw.get("edge_proposals", ()) or ()),
        path_proposals=tuple(raw.get("path_proposals", ()) or ()),
        test_proposals=tuple(raw.get("test_proposals", ()) or ()),
        metadata={
            **adapter_metadata,
            **_safe_metadata(
                reason="",
                attempted=True,
                succeeded=True,
                adapter=adapter,
                response=response,
                extra={**response_metadata, "schema_valid_response": True, "raw_output_keys": sorted(raw.keys())},
            ),
        },
    )
