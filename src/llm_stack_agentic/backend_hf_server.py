"""
HuggingFace-compatible server backend (OpenAI-compatible API).

Uses the ``openai`` Python SDK so that ``extra_body`` is merged into the
top-level request body automatically (avoids 422 with raw requests).

Works with any OpenAI-compatible endpoint:
  - huggingface/text-generation-inference (TGI)
  - ``transformers serve`` (built-in lightweight server)
  - vllm

Adds Qwen3.5 thinking-mode support via ``enable_thinking`` in extra_body.
The <think>...</think> block is stripped before returning so downstream
JSON parsing is not affected.

Config keys (in model_config):
  hf_server_base_url        URL of the server, e.g. "http://0.0.0.0:8000"
  hf_server_api_key         Optional Bearer token (or use "api_key")
  use_thinking              bool – strip <think> tags from output; set
                            server_extended_sampling:true to also send
                            enable_thinking to the server
  server_extended_sampling  bool (default false) – send non-standard params
                            (top_k, min_p, repetition_penalty, enable_thinking)
                            only when the server supports them (TGI / vllm).
                            Leave false for basic ``transformers serve``.
  request_timeout           int seconds (default 180)
  response_format           dict or null – defaults to {"type":"json_object"}
                            when thinking is off; disabled automatically when on
  guided_json               optional JSON schema for constrained decoding
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return _THINK_RE.sub("", text).strip()


def generate(
    *,
    messages: List[Dict[str, Any]],
    model_name: str,
    base_url: str,
    gen_params: Dict[str, Any],
    model_config: Dict[str, Any],
    debug: bool = False,
    debug_log_path: Optional[str] = None,
    logging_config: Optional[Dict[str, Any]] = None,
) -> str:
    """POST to an OpenAI-compatible chat/completions endpoint via the openai SDK.

    Images are passed as ``image_url`` content blocks (base64 data URIs or
    plain URLs); the SDK serialises them correctly.

    Qwen3.5 thinking params (when use_thinking=true in model_config):
      temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
      presence_penalty=0.0, repetition_penalty=1.0
    """
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing_openai_package:{exc}") from exc

    base_url = base_url.rstrip("/")
    use_thinking = bool(model_config.get("use_thinking", False))
    timeout = int(model_config.get("request_timeout", 180))

    api_key = model_config.get("hf_server_api_key") or model_config.get("api_key") or "EMPTY"
    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)

    # --- standard params -----------------------------------------------------
    create_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": float(gen_params.get("temperature", 0.6)),
        "top_p": float(gen_params.get("top_p", 0.95)),
        "max_tokens": int(gen_params.get("max_new_tokens", 8192)),
    }
    if "presence_penalty" in gen_params:
        create_kwargs["presence_penalty"] = float(gen_params["presence_penalty"])

    # structured output — disabled when thinking is on (model needs free-form)
    response_format = model_config.get("response_format")
    if response_format is None and not use_thinking:
        response_format = {"type": "json_object"}
    if response_format:
        create_kwargs["response_format"] = response_format

    # Extended sampling params (top_k, min_p, repetition_penalty, enable_thinking)
    # are non-standard OpenAI fields supported by TGI / vllm but NOT by the
    # basic `transformers serve` server.  Set server_extended_sampling:true to
    # enable them.  The openai SDK merges extra_body into the top-level JSON
    # body automatically (no 422 from "Unexpected keys").
    extra_body: Dict[str, Any] = {}
    if bool(model_config.get("server_extended_sampling", False)):
        for key in ("top_k", "repetition_penalty", "min_p"):
            if key in gen_params:
                extra_body[key] = gen_params[key]
        if use_thinking:
            extra_body["enable_thinking"] = True
        guided_json = model_config.get("guided_json")
        if guided_json:
            extra_body["guided_json"] = guided_json

    if extra_body:
        create_kwargs["extra_body"] = extra_body

    # --- debug log request ---------------------------------------------------
    if debug and debug_log_path:
        _append_debug(debug_log_path, {
            "event": "hf_server_request",
            "ts": time.time(),
            "base_url": base_url,
            "create_kwargs": _sanitize_create_kwargs(create_kwargs),
        })

    # --- SDK call ------------------------------------------------------------
    response = client.chat.completions.create(timeout=timeout, **create_kwargs)
    content = str(response.choices[0].message.content or "")

    # strip thinking tokens so downstream JSON parsing sees clean output
    if use_thinking:
        content = strip_thinking_tags(content)

    if debug and debug_log_path:
        _append_debug(debug_log_path, {
            "event": "hf_server_response",
            "ts": time.time(),
            "content": content[:2000],
        })

    return content


def _sanitize_create_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove base64 image blobs from debug logs."""
    out = dict(kwargs)
    msgs = out.get("messages")
    if not isinstance(msgs, list):
        return out
    sanitized = []
    for msg in msgs:
        content = msg.get("content")
        if isinstance(content, list):
            items = [
                {"type": "image_url", "image_url": {"url": "<omitted>"}}
                if isinstance(i, dict) and i.get("type") == "image_url"
                else i
                for i in content
            ]
            sanitized.append({**msg, "content": items})
        else:
            sanitized.append(msg)
    out["messages"] = sanitized
    return out


def _append_debug(path: str, payload: Dict[str, Any]) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")
