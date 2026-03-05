from __future__ import annotations

import base64
import importlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_GEN_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 800,
    "repetition_penalty": 1.05,
    "do_sample": False,
}


PROMPT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "prompts", "visual_describer.json")
GPT_OSS_PROMPT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "prompts", "visual_describer_gpt_oss.json")


def run_visual_describer(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    probe_trace: Dict[str, Any],
    max_pois: int = 5,
    frame_diffs: Optional[Any] = None,
    model_config: Optional[Dict[str, Any]] = None,
    logging_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_config = model_config or {}
    logging_config = logging_config or {}
    errors: List[str] = []

    frames = _extract_frames(probe_trace, max_frames=4)
    if len(frames) < 2:
        errors.append("insufficient_frames")
    h, w = _infer_hw(frames)
    if h <= 0 or w <= 0:
        errors.append("invalid_frame_dimensions")

    prompt = _build_prompt(
        frames,
        w,
        h,
        frame_diffs=frame_diffs,
        max_pois=max_pois,
        include_grids=True,
        include_system=True,
    )
    model_name = str(model_config.get("model_name") or "unknown")
    gen_params = dict(DEFAULT_GEN_PARAMS)
    gen_params.update(model_config.get("generation_params") or {})
    gen_params = _apply_env_generation_overrides(gen_params)
    # When thinking mode is active the model needs non-zero temperature;
    # default force_deterministic to False in that case.
    default_force_det = not bool(model_config.get("use_thinking", False))
    if model_config.get("force_deterministic", default_force_det):
        gen_params["temperature"] = 0.0
        gen_params["top_p"] = 1.0
        gen_params["top_k"] = 0
        gen_params["do_sample"] = False

    # GPT OSS text-grid path: grids sent as JSON text, dedicated prompt & schema.
    if bool(model_config.get("text_grids", False)):
        diffs = frame_diffs if frame_diffs is not None else (probe_trace.get("frame_diffs") or [])
        return _run_gpt_oss_path(
            frames=frames, h=h, w=w, frame_diffs_input=diffs,
            model_config=model_config, gen_params=gen_params, model_name=model_name,
            errors=list(errors), episode_id=episode_id, game_id=game_id, seed=seed,
            logging_config=logging_config,
        )

    text_output = None
    if not errors:
        text_output = _call_model(prompt, frames, model_config, gen_params, errors, logging_config)

    if text_output is None:
        return _error_report(
            episode_id=episode_id,
            game_id=game_id,
            seed=seed,
            model_name=model_name,
            gen_params=gen_params,
            errors=errors or ["no_model_output"],
            trace_id=logging_config.get("trace_id"),
            timestamp_step=logging_config.get("timestamp_step"),
        )

    parsed = _parse_json(text_output, errors)
    if parsed is None:
        return _error_report(
            episode_id=episode_id,
            game_id=game_id,
            seed=seed,
            model_name=model_name,
            gen_params=gen_params,
            errors=errors or ["parse_failed"],
            trace_id=logging_config.get("trace_id"),
            timestamp_step=logging_config.get("timestamp_step"),
        )

    if logging_config and logging_config.get("debug") and logging_config.get("debug_log_path"):
        _append_debug_log(
            logging_config["debug_log_path"],
            {
                "event": "visual_describer_json_parsed",
                "ts": time.time(),
                "keys": sorted(list(parsed.keys())),
                "content": parsed,
            },
        )

    valid, validation_errors = _validate_output(parsed, w, h, max_pois)
    if not valid:
        errors.extend(validation_errors)
        retry_output = _call_model(prompt, frames, model_config, gen_params, errors, logging_config)
        if retry_output:
            parsed_retry = _parse_json(retry_output, errors)
            if parsed_retry is not None:
                valid_retry, validation_errors_retry = _validate_output(parsed_retry, w, h, max_pois)
                if valid_retry:
                    parsed = parsed_retry
                    errors = [e for e in errors if e not in validation_errors]
                else:
                    errors.extend(validation_errors_retry)

    if errors:
        return _error_report(
            episode_id=episode_id,
            game_id=game_id,
            seed=seed,
            model_name=model_name,
            gen_params=gen_params,
            errors=errors,
            trace_id=logging_config.get("trace_id"),
            timestamp_step=logging_config.get("timestamp_step"),
        )

    return _build_report(
        episode_id=episode_id,
        game_id=game_id,
        seed=seed,
        model_name=model_name,
        gen_params=gen_params,
        payload=parsed,
        trace_id=logging_config.get("trace_id"),
        timestamp_step=logging_config.get("timestamp_step"),
    )


def _extract_frames(probe_trace: Dict[str, Any], *, max_frames: int) -> List[List[List[int]]]:
    frames: List[List[List[int]]] = []
    seen = set()
    steps = probe_trace.get("steps") or []
    for step in steps:
        obs = step.get("obs") or {}
        grid = obs.get("grid")
        if isinstance(grid, list):
            key = _grid_key(grid)
            if key not in seen:
                seen.add(key)
                frames.append(grid)
                if len(frames) >= max_frames:
                    break
    return frames


def _grid_key(grid: List[List[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def _infer_hw(frames: List[List[List[int]]]) -> Tuple[int, int]:
    if not frames:
        return 0, 0
    h = len(frames[0])
    w = len(frames[0][0]) if h > 0 else 0
    return h, w


def _build_prompt(
    frames: List[List[List[int]]],
    w: int,
    h: int,
    *,
    frame_diffs: Optional[Any],
    max_pois: int,
    include_grids: bool,
    include_system: bool = True,
) -> str:
    cfg = _load_prompt_config()
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    task_lines = cfg.get("task_lines") or []
    frames_section = cfg.get("frames_section") or {}
    schema_lines = cfg.get("schema_lines") or []
    rules_lines = cfg.get("rules_lines") or []

    prompt_parts = []
    if include_system and system_prompt:
        prompt_parts.extend([system_prompt, ""])
    for line in task_lines:
        prompt_parts.append(
            str(line)
            .replace("{W}", str(w))
            .replace("{H}", str(h))
        )
    if include_grids:
        prompt_parts.append(str(frames_section.get("grids_header", "Frames (as JSON grids):")))
        for idx, grid in enumerate(frames):
            prompt_parts.append(f"Frame {idx}: {json.dumps(grid)}")
    else:
        prompt_parts.append(str(frames_section.get("images_header", "Frames provided as images in order:")))
        for idx in range(len(frames)):
            prompt_parts.append(f"- Frame {idx}")
    if frame_diffs is not None:
        prompt_parts.append(f"Frame diffs (optional): {json.dumps(frame_diffs)}")

    prompt_parts.append("")
    for line in schema_lines:
        prompt_parts.append(str(line))
    prompt_parts.append("")
    for line in rules_lines:
        prompt_parts.append(str(line).replace("{K}", str(max_pois)))
    return "\n".join(prompt_parts)


def _call_model(
    prompt: str,
    frames: List[List[List[int]]],
    model_config: Dict[str, Any],
    gen_params: Dict[str, Any],
    errors: List[str],
    logging_config: Optional[Dict[str, Any]],
) -> Optional[str]:
    debug = bool(logging_config.get("debug")) if logging_config else False
    debug_log_path = logging_config.get("debug_log_path") if logging_config else None
    backend = str(model_config.get("backend") or "")
    # Write early sentinel so the log is non-empty even if the model call fails.
    if debug and debug_log_path:
        _append_debug_log(debug_log_path, {
            "event": "call_model_invoked",
            "backend": backend,
            "model_name": model_config.get("model_name"),
            "ts": time.time(),
        })
    if backend == "python_module":
        path = model_config.get("callable")
        if not path or ":" not in str(path):
            errors.append("invalid_model_callable")
            return None
        module_name, attr = str(path).split(":", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            errors.append("model_callable_not_found")
            return None
        fn = getattr(module, attr)
        return fn(prompt=prompt, frames=frames, generation_params=gen_params)
    if backend == "transformers_vl":
        try:
            return _transformers_vl_generate(prompt, frames, model_config, gen_params, logging_config)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"transformers_vl_error:{exc}")
            return None
    if backend == "hf_server":
        try:
            return _hf_server_generate(
                prompt,
                frames,
                model_config,
                gen_params,
                debug=debug,
                debug_log_path=debug_log_path,
                logging_config=logging_config,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hf_server_error:{exc}")
            return None
    if backend == "vllm":
        try:
            return _vllm_generate(
                prompt,
                frames,
                model_config,
                gen_params,
                debug=debug,
                debug_log_path=debug_log_path,
                logging_config=logging_config,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vllm_error:{exc}")
            return None

    errors.append("no_model_backend")
    return None


def _parse_json(text: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            errors.append("output_not_object")
            return None
        return parsed
    except Exception:  # noqa: BLE001
        errors.append("json_parse_error")
        return None


def _validate_output(payload: Dict[str, Any], w: int, h: int, max_pois: int) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    required_keys = ["game_description", "sprite_character", "ignore_regions", "poi_list", "exit_hypotheses"]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing_{key}")

    def _check_region(item: Dict[str, Any], label: str) -> None:
        for k in ("x0", "y0", "x1", "y1", "confidence"):
            if k not in item:
                errors.append(f"{label}_missing_{k}")
                return
        if not _in_bounds(item["x0"], item["y0"], w, h) or not _in_bounds(item["x1"], item["y1"], w, h):
            errors.append(f"{label}_out_of_bounds")
        if not _confidence_valid(item["confidence"]):
            errors.append(f"{label}_bad_confidence")

    sprite = payload.get("sprite_character")
    if isinstance(sprite, dict):
        for k in ("x", "y", "width", "height", "confidence"):
            if k not in sprite:
                errors.append("sprite_missing_fields")
                break
        if not _in_bounds(sprite.get("x"), sprite.get("y"), w, h):
            errors.append("sprite_out_of_bounds")
        if not isinstance(sprite.get("width"), int) or not isinstance(sprite.get("height"), int):
            errors.append("sprite_bad_size")
        if not _confidence_valid(sprite.get("confidence")):
            errors.append("sprite_bad_confidence")

    for item in payload.get("ignore_regions", []) or []:
        if isinstance(item, dict):
            _check_region(item, "ignore_region")

    if len(payload.get("ignore_regions", []) or []) > 5:
        errors.append("ignore_region_count_exceeds_max")

    poi_list = payload.get("poi_list", []) or []
    if len(poi_list) > max_pois:
        errors.append("poi_count_exceeds_max")
    for item in poi_list:
        if not isinstance(item, dict):
            errors.append("poi_invalid")
            continue
        for k in ("id", "x", "y", "intent", "rationale", "priority", "confidence"):
            if k not in item:
                errors.append("poi_missing_fields")
                break
        if not _in_bounds(item.get("x"), item.get("y"), w, h):
            errors.append("poi_out_of_bounds")
        if not isinstance(item.get("priority"), int) or not (1 <= item["priority"] <= max_pois):
            errors.append("poi_bad_priority")
        if not _confidence_valid(item.get("confidence")):
            errors.append("poi_bad_confidence")

    for item in payload.get("exit_hypotheses", []) or []:
        if not isinstance(item, dict):
            errors.append("exit_invalid")
            continue
        for k in ("x", "y", "type", "confidence"):
            if k not in item:
                errors.append("exit_missing_fields")
                break
        if not _in_bounds(item.get("x"), item.get("y"), w, h):
            errors.append("exit_out_of_bounds")
        if not _confidence_valid(item.get("confidence")):
            errors.append("exit_bad_confidence")

    if len(payload.get("exit_hypotheses", []) or []) > 3:
        errors.append("exit_count_exceeds_max")

    return len(errors) == 0, errors


def _build_report(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    model_name: str,
    gen_params: Dict[str, Any],
    payload: Dict[str, Any],
    trace_id: Optional[str],
    timestamp_step: Optional[int],
) -> Dict[str, Any]:
    report = {
        "schema_version": "EnvReportV1",
        "agent_name": "lsa_visual_describer",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
        "model_name": model_name,
        "generation_params": gen_params,
        "game_description": payload.get("game_description", ""),
        "sprite_character": payload.get("sprite_character", {}),
        "ignore_regions": payload.get("ignore_regions", []) or [],
        "poi_list": payload.get("poi_list", []) or [],
        "exit_hypotheses": payload.get("exit_hypotheses", []) or [],
        "errors": [],
        "trace_id": trace_id,
        "timestamp_step": timestamp_step,
    }
    return report


def _error_report(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    model_name: str,
    gen_params: Dict[str, Any],
    errors: List[str],
    trace_id: Optional[str],
    timestamp_step: Optional[int],
) -> Dict[str, Any]:
    return {
        "schema_version": "EnvReportV1",
        "agent_name": "lsa_visual_describer",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
        "model_name": model_name,
        "generation_params": gen_params,
        "game_description": "",
        "sprite_character": {},
        "ignore_regions": [],
        "poi_list": [],
        "exit_hypotheses": [],
        "errors": errors,
        "trace_id": trace_id,
        "timestamp_step": timestamp_step,
    }


def _in_bounds(x: Any, y: Any, w: int, h: int) -> bool:
    if not isinstance(x, int) or not isinstance(y, int):
        return False
    return 0 <= x < w and 0 <= y < h


def _confidence_valid(value: Any) -> bool:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return False
    return 0.0 <= v <= 1.0


def _apply_env_generation_overrides(gen_params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(gen_params)
    greedy = os.getenv("greedy")
    if greedy is not None:
        greedy_norm = str(greedy).strip().lower()
        if greedy_norm in ("true", "1", "yes"):
            out["do_sample"] = False
        elif greedy_norm in ("false", "0", "no"):
            out["do_sample"] = True
    for key in ("top_p", "top_k", "temperature", "repetition_penalty"):
        val = os.getenv(key)
        if val is None:
            continue
        if key == "top_k":
            out[key] = int(float(val))
        else:
            out[key] = float(val)
    presence_penalty = os.getenv("presence_penalty")
    if presence_penalty is not None:
        out["presence_penalty"] = float(presence_penalty)
    out_seq_length = os.getenv("out_seq_length")
    if out_seq_length is not None:
        out["max_new_tokens"] = int(float(out_seq_length))
    return out


def _safe_generate(model: Any, inputs: Dict[str, Any], gen_params: Dict[str, Any]):
    params = dict(gen_params)
    for _ in range(3):
        try:
            return model.generate(**inputs, **params)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise
            key = message.split("unexpected keyword argument")[-1].strip()
            key = key.strip(":").strip().strip("'").strip('"')
            if key in params:
                params.pop(key)
                continue
            raise
    return model.generate(**inputs, **params)


def _transformers_vl_generate(
    prompt: str,
    frames: List[List[List[int]]],
    model_config: Dict[str, Any],
    gen_params: Dict[str, Any],
    logging_config: Optional[Dict[str, Any]],
) -> str:
    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_name is required for transformers_vl backend")

    include_grids = bool(model_config.get("include_grids_in_prompt", False))
    h, w = _infer_hw(frames)
    prompt = _build_prompt(
        frames,
        w,
        h,
        frame_diffs=None,
        max_pois=5,
        include_grids=include_grids,
        include_system=False,
    )

    try:
        import torch
        from transformers import AutoProcessor
        # Qwen3.5-VL ships as Qwen3_5VLForConditionalGeneration in newer
        # transformers; fall back to Qwen3VL for older installs.
        try:
            from transformers import Qwen3_5VLForConditionalGeneration as _VLModelCls
        except ImportError:
            from transformers import Qwen3VLForConditionalGeneration as _VLModelCls
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing_transformers_or_torch:{exc}") from exc

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing_pillow:{exc}") from exc

    device_map = model_config.get("device_map", "auto")
    torch_dtype = model_config.get("torch_dtype", "auto")
    trust_remote_code = bool(model_config.get("trust_remote_code", True))

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = _VLModelCls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if hasattr(model, "eval"):
        model.eval()

    frame_dir = logging_config.get("frame_dir") if logging_config else None
    png_factor = int(logging_config.get("png_factor", 1)) if logging_config else 1
    image_paths = [
        _grid_to_temp_image(grid, idx, frame_dir=frame_dir, png_factor=png_factor)
        for idx, grid in enumerate(frames)
    ]
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    use_thinking = bool(model_config.get("use_thinking", False))
    _template_kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if use_thinking:
        _template_kwargs["enable_thinking"] = True
    try:
        inputs = processor.apply_chat_template(messages, **_template_kwargs)
    except TypeError:
        # older processor — enable_thinking not supported, drop it
        _template_kwargs.pop("enable_thinking", None)
        inputs = processor.apply_chat_template(messages, **_template_kwargs)

    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = _safe_generate(model, inputs, gen_params)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    result = decoded[0] if decoded else ""
    if use_thinking:
        from .backend_hf_server import strip_thinking_tags
        result = strip_thinking_tags(result)
    return result


def _vllm_generate(
    prompt: str,
    frames: List[List[List[int]]],
    model_config: Dict[str, Any],
    gen_params: Dict[str, Any],
    *,
    debug: bool,
    debug_log_path: Optional[str],
    logging_config: Optional[Dict[str, Any]],
) -> str:
    base_url = str(model_config.get("vllm_base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("vllm_base_url is required for vllm backend")
    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_name is required for vllm backend")

    include_grids = bool(model_config.get("include_grids_in_prompt", False))
    h, w = _infer_hw(frames)
    prompt = _build_prompt(
        frames,
        w,
        h,
        frame_diffs=None,
        max_pois=5,
        include_grids=include_grids,
        include_system=False,
    )

    image_payloads = []
    frame_dir = logging_config.get("frame_dir") if logging_config else None
    png_factor = int(logging_config.get("png_factor", 1)) if logging_config else 1
    for idx, grid in enumerate(frames):
        path = _grid_to_temp_image(grid, idx, frame_dir=frame_dir, png_factor=png_factor)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_payloads.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    cfg = _load_prompt_config()
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [*image_payloads, {"type": "text", "text": prompt}]},
    ]

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": gen_params.get("temperature"),
        "top_p": gen_params.get("top_p"),
        "max_tokens": gen_params.get("max_new_tokens"),
    }
    response_format = model_config.get("response_format")
    if response_format is None:
        response_format = {"type": "json_object"}
    if response_format:
        payload["response_format"] = response_format
    extra_body: Dict[str, Any] = {}
    if "top_k" in gen_params:
        extra_body["top_k"] = gen_params.get("top_k")
    if "repetition_penalty" in gen_params:
        extra_body["repetition_penalty"] = gen_params.get("repetition_penalty")
    if "presence_penalty" in gen_params:
        extra_body["presence_penalty"] = gen_params.get("presence_penalty")
    guided_json = model_config.get("guided_json")
    if guided_json:
        extra_body["guided_json"] = guided_json
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing_requests:{exc}") from exc

    headers = {"Content-Type": "application/json"}
    api_key = model_config.get("vllm_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if debug and debug_log_path:
        log_payload = dict(payload)
        log_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                filtered = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        filtered.append({"type": "image_url", "image_url": {"url": "<omitted>"}})
                    else:
                        filtered.append(item)
                log_messages.append({**msg, "content": filtered})
            else:
                log_messages.append(msg)
        log_payload["messages"] = log_messages
        _append_debug_log(
            debug_log_path,
            {
                "event": "vllm_request",
                "ts": time.time(),
                "url": f"{base_url}/v1/chat/completions",
                "payload": log_payload,
            },
        )
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        if debug and debug_log_path:
            _append_debug_log(
                debug_log_path,
                {
                    "event": "vllm_response_error",
                    "ts": time.time(),
                    "status": resp.status_code,
                    "body": resp.text,
                },
            )
        raise RuntimeError(f"vllm_http_{resp.status_code}:{resp.text}")
    data = resp.json()
    if debug and debug_log_path:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        content_json = _extract_json_from_text(content) if isinstance(content, str) else None
        log_payload = {
            "event": "vllm_response",
            "ts": time.time(),
            "status": resp.status_code,
        }
        if content_json is not None:
            log_payload["content_json"] = content_json
        else:
            log_payload["content"] = content
        _append_debug_log(debug_log_path, log_payload)
        if content_json is not None:
            pretty = json.dumps(content_json, indent=2, ensure_ascii=False)
        else:
            pretty = str(content) if content is not None else ""
        _append_debug_raw(debug_log_path, "vllm_response_content_pretty:\n" + pretty)
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def _hf_server_generate(
    prompt: str,
    frames: List[List[List[int]]],
    model_config: Dict[str, Any],
    gen_params: Dict[str, Any],
    *,
    debug: bool,
    debug_log_path: Optional[str],
    logging_config: Optional[Dict[str, Any]],
) -> str:
    """Call an OpenAI-compatible HuggingFace server (TGI, transformers serve, etc.)."""
    from . import backend_hf_server

    base_url = str(model_config.get("hf_server_base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("hf_server_base_url is required for hf_server backend")
    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_name is required for hf_server backend")

    include_grids = bool(model_config.get("include_grids_in_prompt", False))
    h, w = _infer_hw(frames)
    prompt_built = _build_prompt(
        frames,
        w,
        h,
        frame_diffs=None,
        max_pois=5,
        include_grids=include_grids,
        include_system=False,
    )

    frame_dir = logging_config.get("frame_dir") if logging_config else None
    _lc_factor = int(logging_config.get("png_factor", 0)) if logging_config else 0
    png_factor = _lc_factor if _lc_factor > 0 else int(model_config.get("png_factor", 1))
    image_payloads = []
    for idx, grid in enumerate(frames):
        path = _grid_to_temp_image(grid, idx, frame_dir=frame_dir, png_factor=png_factor)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_payloads.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    cfg = _load_prompt_config()
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [*image_payloads, {"type": "text", "text": prompt_built}]},
    ]

    return backend_hf_server.generate(
        messages=messages,
        model_name=model_name,
        base_url=base_url,
        gen_params=gen_params,
        model_config=model_config,
        debug=debug,
        debug_log_path=debug_log_path,
        logging_config=logging_config,
    )


def _append_debug_log(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    max_size = 2_000_000
    keep_size = 1_000_000
    if os.path.exists(path) and os.path.getsize(path) > max_size:
        with open(path, "rb") as f:
            data = f.read()
        data = data[-keep_size:]
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")


def _append_debug_raw(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _grid_to_image(grid: List[List[int]]) -> "Image.Image":
    from PIL import Image

    if not grid:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))
    h = len(grid)
    w = len(grid[0]) if h else 1
    palette = _default_palette()
    pixels = []
    for row in grid:
        for val in row:
            idx = int(val) % len(palette)
            pixels.append(palette[idx])
    img = Image.new("RGB", (w, h))
    img.putdata(pixels)
    return img


def _grid_to_temp_image(grid: List[List[int]], idx: int, *, frame_dir: Optional[str], png_factor: int) -> str:
    import os
    import tempfile
    from PIL import Image

    img = _grid_to_image(grid)
    scale = max(1, int(png_factor))
    if scale != 1:
        resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        img = img.resize((img.width * scale, img.height * scale), resample=resample)
    tmp_dir = frame_dir or os.path.join(tempfile.gettempdir(), "lsa_vl_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"frame_{idx}.png")
    img.save(path, format="PNG")
    return path


def _default_palette() -> List[Tuple[int, int, int]]:
    return [
        (0, 0, 0),
        (30, 147, 255),
        (255, 220, 0),
        (249, 60, 49),
        (0, 200, 70),
        (255, 255, 255),
        (255, 105, 180),
        (128, 0, 128),
        (0, 128, 128),
        (255, 140, 0),
        (70, 70, 70),
        (160, 160, 160),
        (100, 200, 255),
        (200, 255, 100),
        (255, 100, 200),
        (200, 100, 255),
    ]


def _load_prompt_config() -> Dict[str, Any]:
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_gpt_oss_prompt_config() -> Dict[str, Any]:
    try:
        with open(GPT_OSS_PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# GPT OSS text-grid path
# ---------------------------------------------------------------------------

# 0 → '.', 1 → 'A', 2 → 'B', …, 15 → 'O'
_CELL_CHARS = "." + "ABCDEFGHIJKLMNOP"


def _cell_char(value: int) -> str:
    idx = max(0, min(int(value), len(_CELL_CHARS) - 1))
    return _CELL_CHARS[idx]


def _grid_to_text(grid: List[List[int]]) -> str:
    """Convert a 2D int grid to a compact letter grid (one char per cell, one row per line)."""
    return "\n".join("".join(_cell_char(v) for v in row) for row in grid)


def _diffs_to_text(diffs: List[Any]) -> str:
    """Render pre-computed frame diffs using letter values, one change per line."""
    lines: List[str] = []
    for d in diffs:
        changes = d.get("changes", [])
        lines.append(f"frame{d['from_frame']}→frame{d['to_frame']}:")
        if not changes:
            lines.append("  no changes")
        else:
            for c in changes:
                lines.append(
                    f"  ({c['x']},{c['y']}) -> {_cell_char(c['old_value'])}->{_cell_char(c['new_value'])}"
                )
    return "\n".join(lines)


def _build_gpt_oss_prompt(
    frames: List[List[List[int]]],
    frame_diffs_input: List[Any],
    w: int,
    h: int,
) -> str:
    """Build the user-turn prompt for the GPT OSS text-grid path."""
    cfg = _load_gpt_oss_prompt_config()
    parts: List[str] = []
    for line in (cfg.get("task_lines") or []):
        parts.append(str(line).replace("{W}", str(w)).replace("{H}", str(h)))

    parts.append(
        "Cell encoding: '.' = 0 (background), 'A' = 1, 'B' = 2, …, 'O' = 15. "
        "Each row is one line; x=column index (left→right), y=row index (top→bottom)."
    )
    parts.append(str((cfg.get("frames_section") or {}).get("grids_header", "Frames:")))
    for idx, grid in enumerate(frames):
        parts.append(f"Frame {idx}:")
        parts.append(_grid_to_text(grid))

    if frame_diffs_input:
        parts.append("\nPre-computed frame diffs (verify and include in your response):")
        parts.append(_diffs_to_text(frame_diffs_input))

    parts.append("")
    for line in (cfg.get("schema_lines") or []):
        parts.append(str(line))
    parts.append("")
    for line in (cfg.get("rules_lines") or []):
        parts.append(str(line))
    return "\n".join(parts)


def _hf_server_text_generate(
    prompt: str,
    model_config: Dict[str, Any],
    gen_params: Dict[str, Any],
    *,
    debug: bool,
    debug_log_path: Optional[str],
    logging_config: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Send a text-only request to the HF-compatible server (no image payloads)."""
    from . import backend_hf_server

    base_url = str(model_config.get("hf_server_base_url") or "").rstrip("/")
    model_name = model_config.get("model_name")
    if not base_url or not model_name:
        return None

    cfg = _load_gpt_oss_prompt_config()
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        return backend_hf_server.generate(
            messages=messages,
            model_name=model_name,
            base_url=base_url,
            gen_params=gen_params,
            model_config=model_config,
            debug=debug,
            debug_log_path=debug_log_path,
            logging_config=logging_config,
        )
    except Exception as exc:  # noqa: BLE001
        if debug and debug_log_path:
            _append_debug_log(debug_log_path, {"event": "gpt_oss_error", "error": str(exc)})
        return None


def _validate_gpt_oss_output(payload: Dict[str, Any], w: int, h: int) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    for key in ("Description", "Background colour letter", "sprite_positions", "poi_list", "exit_hypotheses"):
        if key not in payload:
            errs.append(f"gpt_oss_missing_{key}")

    for item in payload.get("poi_list", []) or []:
        if not isinstance(item, dict):
            errs.append("gpt_oss_poi_invalid")
            continue
        for k in ("id", "x", "y", "rationale", "confidence"):
            if k not in item:
                errs.append("gpt_oss_poi_missing_fields")
                break
        if not _in_bounds(item.get("x"), item.get("y"), w, h):
            errs.append("gpt_oss_poi_out_of_bounds")
        if not _confidence_valid(item.get("confidence")):
            errs.append("gpt_oss_poi_bad_confidence")

    for item in payload.get("sprite_positions", []) or []:
        if not isinstance(item, dict):
            errs.append("gpt_oss_sprite_invalid")
            continue
        for k in ("frame", "x", "y", "confidence"):
            if k not in item:
                errs.append("gpt_oss_sprite_missing_fields")
                break
        if not _in_bounds(item.get("x"), item.get("y"), w, h):
            errs.append("gpt_oss_sprite_out_of_bounds")

    for item in payload.get("exit_hypotheses", []) or []:
        if not isinstance(item, dict):
            errs.append("gpt_oss_exit_invalid")
            continue
        for k in ("x", "y", "rationale", "confidence"):
            if k not in item:
                errs.append("gpt_oss_exit_missing_fields")
                break
        if not _in_bounds(item.get("x"), item.get("y"), w, h):
            errs.append("gpt_oss_exit_out_of_bounds")

    return len(errs) == 0, errs


def _build_gpt_oss_report(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    model_name: str,
    gen_params: Dict[str, Any],
    payload: Dict[str, Any],
    trace_id: Optional[str],
    timestamp_step: Optional[int],
) -> Dict[str, Any]:
    # Normalise poi_list to add fields expected by controller routing.
    poi_list = []
    for i, item in enumerate(payload.get("poi_list", []) or []):
        poi = dict(item)
        poi.setdefault("intent", poi.get("rationale", ""))
        poi.setdefault("priority", i + 1)
        poi_list.append(poi)
    return {
        "schema_version": "GptOssEnvReportV1",
        "agent_name": "lsa_visual_describer",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
        "model_name": model_name,
        "generation_params": gen_params,
        "description": payload.get("Description", []),
        "background_colour_letter": payload.get("Background colour letter", []),
        "frame_diffs": payload.get("frame_diffs", []),
        "sprite_positions": payload.get("sprite_positions", []),
        "poi_list": poi_list,
        "exit_hypotheses": payload.get("exit_hypotheses", []),
        "errors": [],
        "trace_id": trace_id,
        "timestamp_step": timestamp_step,
    }


def _gpt_oss_error_report(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    model_name: str,
    gen_params: Dict[str, Any],
    errors: List[str],
    trace_id: Optional[str],
    timestamp_step: Optional[int],
) -> Dict[str, Any]:
    return {
        "schema_version": "GptOssEnvReportV1",
        "agent_name": "lsa_visual_describer",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
        "model_name": model_name,
        "generation_params": gen_params,
        "frame_diffs": [],
        "sprite_positions": [],
        "poi_list": [],
        "exit_hypotheses": [],
        "errors": errors,
        "trace_id": trace_id,
        "timestamp_step": timestamp_step,
    }


def _run_gpt_oss_path(
    *,
    frames: List[List[List[int]]],
    h: int,
    w: int,
    frame_diffs_input: List[Any],
    model_config: Dict[str, Any],
    gen_params: Dict[str, Any],
    model_name: str,
    errors: List[str],
    episode_id: str,
    game_id: str,
    seed: int,
    logging_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    debug = bool(logging_config.get("debug")) if logging_config else False
    debug_log_path = logging_config.get("debug_log_path") if logging_config else None
    trace_id = logging_config.get("trace_id") if logging_config else None
    timestamp_step = logging_config.get("timestamp_step") if logging_config else None

    _err_kw = dict(
        episode_id=episode_id, game_id=game_id, seed=seed,
        model_name=model_name, gen_params=gen_params,
        trace_id=trace_id, timestamp_step=timestamp_step,
    )

    if errors:
        return _gpt_oss_error_report(errors=errors, **_err_kw)

    prompt = _build_gpt_oss_prompt(frames, frame_diffs_input, w, h)

    if debug and debug_log_path:
        _append_debug_log(debug_log_path, {
            "event": "gpt_oss_request",
            "ts": time.time(),
            "model_name": model_name,
            "num_frames": len(frames),
            "num_diffs": len(frame_diffs_input),
            "prompt_len": len(prompt),
        })

    text_output = _hf_server_text_generate(
        prompt, model_config, gen_params,
        debug=debug, debug_log_path=debug_log_path, logging_config=logging_config,
    )
    if text_output is None:
        return _gpt_oss_error_report(errors=["no_model_output"], **_err_kw)

    if debug and debug_log_path:
        _append_debug_log(debug_log_path, {
            "event": "gpt_oss_raw_response",
            "ts": time.time(),
            "content": text_output[:4000],
        })

    parsed = _parse_json(text_output, errors)
    if parsed is None:
        return _gpt_oss_error_report(errors=errors or ["parse_failed"], **_err_kw)

    valid, val_errors = _validate_gpt_oss_output(parsed, w, h)
    if not valid:
        errors.extend(val_errors)
        if debug and debug_log_path:
            _append_debug_log(debug_log_path, {
                "event": "gpt_oss_validation_failed",
                "ts": time.time(),
                "val_errors": val_errors,
                "parsed": parsed,
            })
        retry_text = _hf_server_text_generate(
            prompt, model_config, gen_params,
            debug=debug, debug_log_path=debug_log_path, logging_config=logging_config,
        )
        if retry_text:
            parsed_retry = _parse_json(retry_text, [])
            if parsed_retry is not None:
                valid_retry, _ = _validate_gpt_oss_output(parsed_retry, w, h)
                if valid_retry:
                    parsed = parsed_retry
                    errors = [e for e in errors if e not in val_errors]

    if errors:
        return _gpt_oss_error_report(errors=errors, **_err_kw)

    if debug and debug_log_path:
        _append_debug_log(debug_log_path, {
            "event": "gpt_oss_parsed",
            "ts": time.time(),
            "poi_count": len(parsed.get("poi_list", [])),
            "sprite_count": len(parsed.get("sprite_positions", [])),
            "exit_count": len(parsed.get("exit_hypotheses", [])),
            "description": parsed.get("Description", []),
            "background_colour_letter": parsed.get("Background colour letter", []),
            "frame_diffs": parsed.get("frame_diffs", []),
            "sprite_positions": parsed.get("sprite_positions", []),
            "poi_list": parsed.get("poi_list", []),
            "exit_hypotheses": parsed.get("exit_hypotheses", []),
        })

    return _build_gpt_oss_report(payload=parsed, **_err_kw)
