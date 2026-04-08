from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import requests


def call_ollama(
    *,
    ollama_url: str,
    ollama_model: str,
    ollama_num_ctx: int,
    system_prompt: str,
    prompt: str,
    frame_dir: str,
    max_prompt_frames: int,
    timeout_sec: float,
    retry_count: int,
    output_path: str,
    conversation_scope: str | None = None,
    reset_context: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload(
        model=ollama_model,
        num_ctx=ollama_num_ctx,
        system_prompt=system_prompt,
        prompt=prompt,
        frame_dir=frame_dir,
        max_prompt_frames=max_prompt_frames,
    )
    last_error: str | None = None
    record: dict[str, Any] = {}
    for attempt in range(retry_count + 1):
        try:
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout_sec,
            )
            raw_text = response.text
            try:
                response_json = response.json() if raw_text else {}
            except Exception:
                response_json = {"raw_text": raw_text}
            record = {
                "request": _redacted_payload(payload),
                "metadata": {
                    **dict(metadata or {}),
                    "conversation_scope": conversation_scope,
                    "reset_context": bool(reset_context),
                },
                "status_code": int(response.status_code),
                "response": response_json if isinstance(response_json, dict) else {"raw_text": raw_text},
                "response_text": raw_text,
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            if response.status_code == 200:
                return record
            last_error = f"http_{response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"ollama request failed: {last_error}")


def extract_response_text(record: dict[str, Any]) -> str:
    payload = record.get("response", {})
    if not isinstance(payload, dict):
        return str(payload)
    message = payload.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return str(payload.get("raw_text") or "").strip()


def _build_payload(
    *,
    model: str,
    num_ctx: int,
    system_prompt: str,
    prompt: str,
    frame_dir: str,
    max_prompt_frames: int,
) -> dict[str, Any]:
    images = [_base64_file(path) for path in _sample_frame_paths(frame_dir, max_prompt_frames=max_prompt_frames)]
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt, "images": images},
        ],
        "options": {
            "num_ctx": int(num_ctx),
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


def _redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))
    try:
        user_message = clone["messages"][1]
        if isinstance(user_message.get("images"), list):
            user_message["images"] = [f"<base64:{idx}>" for idx, _ in enumerate(user_message["images"])]
    except Exception:
        pass
    return clone
