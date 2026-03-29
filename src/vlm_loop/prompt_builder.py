from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_config(path: str) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prompt config must be a JSON object")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("prompt config must contain a non-empty stages list")
    return payload


def build_prompt(
    *,
    stage_config: dict[str, Any],
    action_set: list[str],
    prior_stage_outputs: dict[str, Any],
    extra_context: dict[str, Any] | None = None,
    task_prompt: str = "",
) -> str:
    task_text = task_prompt.strip() or str(stage_config.get("task_prompt") or "").strip()
    template = str(stage_config.get("user_prompt_template") or "")
    json_contract = str(stage_config.get("json_contract") or "")
    prior_outputs_json = json.dumps(prior_stage_outputs, indent=2, ensure_ascii=False)
    context = dict(extra_context or {})
    next_run_hint = context.get("next_run_hint")
    next_run_hint_text = "" if next_run_hint is None else json.dumps(next_run_hint, ensure_ascii=False, separators=(",", ":"))
    try:
        prompt = template.format(
            task_prompt=task_text,
            action_set=", ".join(action_set),
            json_contract=json_contract,
            prior_stage_outputs_json=prior_outputs_json,
            prior_stage_outputs_compact=json.dumps(prior_stage_outputs, ensure_ascii=False),
            previous_target_json=json.dumps(context.get("previous_target_json"), ensure_ascii=False),
            next_run_hint=next_run_hint_text,
            episode_index=context.get("episode_index", 0),
            episode_outcome=str(context.get("episode_outcome") or ""),
        ).strip()
    except KeyError as exc:
        raise ValueError(f"unsupported prompt template variable: {exc.args[0]}") from exc
    if next_run_hint is None:
        prompt = "\n".join(line for line in prompt.splitlines() if line.strip() != "Next run hint:").strip()
    if not prompt:
        raise ValueError("prompt config produced an empty prompt")
    return prompt


def build_prompt_record(
    *,
    sequence_id: str,
    stage_id: str,
    system_prompt: str,
    prompt: str,
    action_set: list[str],
    prior_stage_outputs: dict[str, Any],
    extra_context: dict[str, Any] | None = None,
    stage_role: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "stage_id": stage_id,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "action_set": list(action_set),
        "prior_stage_outputs_json": json.loads(json.dumps(prior_stage_outputs)),
        "extra_context": json.loads(json.dumps(extra_context or {})),
        "stage_role": json.loads(json.dumps(stage_role or {})),
    }
