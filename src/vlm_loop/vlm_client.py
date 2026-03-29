from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import requests


def analyze_episode(
    *,
    backend: str,
    ollama_url: str,
    ollama_model: str,
    ollama_num_ctx: int,
    vllm_url: str,
    vllm_model: str,
    disable_thinking: bool,
    greedy: bool,
    top_p: float,
    top_k: int,
    temperature: float,
    repetition_penalty: float,
    presence_penalty: float,
    out_seq_length: int,
    frame_dir: str,
    video_path: str,
    system_prompt: str,
    prompt: str,
    metadata: dict[str, Any],
    timeout_sec: float,
    retry_count: int,
    output_path: str,
    max_prompt_frames: int,
) -> dict[str, Any]:
    request_spec = _build_request_spec(
        backend=backend,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        ollama_num_ctx=ollama_num_ctx,
        vllm_url=vllm_url,
        vllm_model=vllm_model,
        disable_thinking=disable_thinking,
        greedy=greedy,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        out_seq_length=out_seq_length,
        frame_dir=frame_dir,
        video_path=video_path,
        system_prompt=system_prompt,
        prompt=prompt,
        metadata=metadata,
        max_prompt_frames=max_prompt_frames,
    )
    last_error: str | None = None
    for attempt in range(retry_count + 1):
        try:
            response = requests.post(
                request_spec["url"],
                json=request_spec["payload"],
                headers=request_spec["headers"],
                timeout=timeout_sec,
            )
            raw_body = response.text
            try:
                data = response.json() if raw_body else {}
            except Exception:
                data = {"raw_text": raw_body}
            record = {
                "backend": backend,
                "request": _redacted_payload(request_spec["payload"]),
                "metadata": metadata,
                "status_code": response.status_code,
                "response": data if isinstance(data, dict) else raw_body,
                "response_text": raw_body,
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            if response.status_code != 200:
                last_error = f"http_{response.status_code}"
                time.sleep(0.5 * (attempt + 1))
                continue
            return record
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"vllm request failed: {last_error}")


def extract_response_text(record: dict[str, Any]) -> str:
    payload = record.get("response", {})
    if not isinstance(payload, dict):
        return _strip_reasoning(str(payload))
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return _strip_reasoning(content)
        if isinstance(content, list):
            text_chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
            return _strip_reasoning("\n".join(chunk for chunk in text_chunks if chunk))
    message = payload.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return _strip_reasoning(content)
    if isinstance(content, list):
        text_chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
        return _strip_reasoning("\n".join(chunk for chunk in text_chunks if chunk))
    return _strip_reasoning(str(payload.get("raw_text") or ""))


def _build_request_spec(
    *,
    backend: str,
    ollama_url: str,
    ollama_model: str,
    ollama_num_ctx: int,
    vllm_url: str,
    vllm_model: str,
    disable_thinking: bool,
    greedy: bool,
    top_p: float,
    top_k: int,
    temperature: float,
    repetition_penalty: float,
    presence_penalty: float,
    out_seq_length: int,
    frame_dir: str,
    video_path: str,
    system_prompt: str,
    prompt: str,
    metadata: dict[str, Any],
    max_prompt_frames: int,
) -> dict[str, Any]:
    if backend == "vllm":
        payload = _build_vllm_payload(
            model=vllm_model,
            frame_dir=frame_dir,
            video_path=video_path,
            system_prompt=system_prompt,
            prompt=prompt,
            metadata=metadata,
            max_prompt_frames=max_prompt_frames,
            disable_thinking=disable_thinking,
            greedy=greedy,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            out_seq_length=out_seq_length,
        )
        return {
            "url": f"{vllm_url.rstrip('/')}/v1/chat/completions",
            "headers": {"Content-Type": "application/json"},
            "payload": payload,
        }
    payload = _build_ollama_payload(
        model=ollama_model,
        num_ctx=ollama_num_ctx,
        frame_dir=frame_dir,
        system_prompt=system_prompt,
        prompt=prompt,
        max_prompt_frames=max_prompt_frames,
        greedy=greedy,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        out_seq_length=out_seq_length,
    )
    return {
        "url": f"{ollama_url.rstrip('/')}/api/chat",
        "headers": {"Content-Type": "application/json"},
        "payload": payload,
    }


def _build_ollama_payload(
    *,
    model: str,
    num_ctx: int,
    frame_dir: str,
    system_prompt: str,
    prompt: str,
    max_prompt_frames: int,
    greedy: bool,
    top_p: float,
    top_k: int,
    temperature: float,
    repetition_penalty: float,
    presence_penalty: float,
    out_seq_length: int,
) -> dict[str, Any]:
    images = [_base64_file(path) for path in _sample_frame_paths(frame_dir, max_prompt_frames=max_prompt_frames)]
    return {
        "model": model or "default",
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": int(num_ctx),
            "num_predict": int(out_seq_length),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repeat_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt, "images": images},
        ],
    }


def _build_vllm_payload(
    *,
    model: str,
    frame_dir: str,
    video_path: str,
    system_prompt: str,
    prompt: str,
    metadata: dict[str, Any],
    max_prompt_frames: int,
    disable_thinking: bool,
    greedy: bool,
    top_p: float,
    top_k: int,
    temperature: float,
    repetition_penalty: float,
    presence_penalty: float,
    out_seq_length: int,
) -> dict[str, Any]:
    del metadata, video_path
    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_base64_file(path)}"}}
        for path in _sample_frame_paths(frame_dir, max_prompt_frames=max_prompt_frames)
    ]
    return {
        "model": model or "default",
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(out_seq_length),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}, *image_parts]},
        ],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": not bool(disable_thinking)},
            "greedy": bool(greedy),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
        },
    }


def _sample_frame_paths(frame_dir: str, *, max_prompt_frames: int) -> list[Path]:
    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if len(frames) <= max_prompt_frames:
        return frames
    if max_prompt_frames <= 1:
        return [frames[0]]
    last_index = len(frames) - 1
    indices = {
        round(position * last_index / float(max_prompt_frames - 1))
        for position in range(max_prompt_frames)
    }
    return [frames[index] for index in sorted(indices)]


def _base64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _strip_reasoning(text: str) -> str:
    cleaned = text.strip()
    for open_tag, close_tag in (("<think>", "</think>"), ("<reasoning>", "</reasoning>")):
        while cleaned.startswith(open_tag) and close_tag in cleaned:
            cleaned = cleaned.split(close_tag, 1)[1].lstrip()
    return cleaned


def _redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(payload))
    for message in cleaned.get("messages", []):
        if isinstance(message, dict) and isinstance(message.get("images"), list):
            message["images"] = ["<omitted>" for _ in message["images"]]
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    item["image_url"] = {"url": "<omitted>"}
    return cleaned
