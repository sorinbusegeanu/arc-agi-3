"""Small OpenAI-compatible multimodal client for Strata and vLLM endpoints."""

from __future__ import annotations

import base64
import copy
import json
import os
import time
import uuid
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from .runlog import append_jsonl


DEFAULT_STRATA_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_STRATA_MODEL = "qwen3.8-flash-next-iq3_s"
DEFAULT_VLLM_BASE_URL = "http://192.168.0.51:8000"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-4B-Thinking-FP8"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 6144
V11_VERSION = "11.1.4"


def user_message(
    text: str,
    image_paths: Sequence[Path] = (),
    image_labels: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one immutable multimodal user turn in OpenAI-compatible format."""
    content: list[dict[str, Any]] = [{"type": "text", "text": str(text)}]
    for index, path in enumerate(image_paths):
        label = image_labels[index] if index < len(image_labels) else f"Image {index + 1}"
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        content.append({"type": "text", "text": str(label)})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    return {"role": "user", "content": content}


class StrataRequestError(RuntimeError):
    """The configured inference endpoint did not return a usable response."""

    def __init__(
        self,
        message: str,
        *,
        request_attempts: int = 1,
        latency_seconds: float = 0.0,
        structured_output_failed: bool = False,
    ) -> None:
        super().__init__(message)
        self.request_attempts = int(request_attempts)
        self.latency_seconds = float(latency_seconds)
        self.structured_output_failed = bool(structured_output_failed)


class StrataClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        reasoning_budget_tokens: int | None = None,
        log_request: bool = False,
        log_response: bool = False,
        log_dir: str | Path | None = None,
        context_window_tokens: int | None = None,
        backend: str = "strata",
    ) -> None:
        backend = str(backend).strip().lower()
        if backend not in {"strata", "vllm"}:
            raise ValueError("backend must be strata or vllm")
        self.backend = backend
        self.backend_label = "Strata" if backend == "strata" else "vLLM"
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model).strip()
        if not self.base_url or not self.model:
            raise ValueError("base_url and model are required")
        if api_key is None:
            env_key = "VLLM_API_KEY" if backend == "vllm" else "STRATA_API_KEY"
            api_key = os.environ.get(env_key)
            if backend == "vllm" and api_key is None:
                api_key = os.environ.get("STRATA_API_KEY")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = int(max_output_tokens)
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("reasoning_effort must be none, low, medium, or high")
        if reasoning_budget_tokens is not None and int(reasoning_budget_tokens) < 1:
            raise ValueError("reasoning_budget_tokens must be positive when specified")
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget_tokens = (
            int(reasoning_budget_tokens) if reasoning_budget_tokens is not None else None
        )
        if context_window_tokens is not None and int(context_window_tokens) < 1:
            raise ValueError("context_window_tokens must be positive when specified")
        self.context_window_tokens = int(context_window_tokens) if context_window_tokens is not None else None
        self.log_request = bool(log_request)
        self.log_response = bool(log_response)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if (self.log_request or self.log_response) and self.log_dir is None:
            raise ValueError("log_dir is required when request or response logging is enabled")
        self._request_sequence = 0
        self._request_ids: list[str] = []

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[Path] = (),
        image_labels: Sequence[str] = (),
        response_format: str | None = "json_object",
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        conversation_reset_count: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._request_ids = []
        if messages is None:
            request_messages = [
                {"role": "system", "content": system_prompt},
                user_message(user_prompt, image_paths, image_labels),
            ]
        else:
            request_messages = copy.deepcopy(list(messages))
            if not request_messages:
                raise ValueError("messages must not be empty")
            if any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant", "tool"}
                or not (isinstance(message.get("content"), (str, list)) or (message.get("role") == "assistant" and message.get("content") is None))
                for message in request_messages
            ):
                raise ValueError("messages must contain system, user, or assistant messages with text or multimodal content")
        payload = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": self.max_output_tokens,
            "temperature": 0,
        }
        if self.backend == "strata":
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = copy.deepcopy(list(tools))
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        elif response_format is not None:
            payload["response_format"] = {"type": str(response_format)}
        if self.backend == "strata" and self.reasoning_budget_tokens is not None:
            payload["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        conversation_message_count = len(request_messages) if messages is not None else None
        request_attempts = 1
        total_latency = 0.0
        submitted_messages = request_messages
        plain_json_fallback_used = False
        try:
            response_body, total_latency = self._post_json(
                payload, conversation_message_count=conversation_message_count,
                conversation_reset_count=conversation_reset_count,
            )
        except StrataRequestError as exc:
            if not exc.structured_output_failed:
                raise
            request_attempts = 2
            total_latency = exc.latency_seconds
            fallback_payload = self._plain_json_fallback(payload)
            submitted_messages = fallback_payload["messages"]
            plain_json_fallback_used = True
            try:
                response_body, fallback_latency = self._post_json(
                    fallback_payload, conversation_message_count=conversation_message_count,
                    conversation_reset_count=conversation_reset_count,
                )
                total_latency += fallback_latency
            except StrataRequestError as fallback_error:
                total_latency += fallback_error.latency_seconds
                raise StrataRequestError(
                    f"{exc}; plain-text JSON recovery failed: {fallback_error}",
                    request_attempts=request_attempts,
                    latency_seconds=total_latency,
                ) from fallback_error
        try:
            result = json.loads(response_body)
            if not isinstance(result, dict):
                raise ValueError("response envelope must be an object")
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("response choices must contain an object")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ValueError("response message must be an object")
            content_text = message.get("content")
            tool_calls = message.get("tool_calls", [])
            if isinstance(content_text, list):
                content_text = "".join(
                    part.get("text", "") for part in content_text if isinstance(part, dict)
                )
            if content_text is None and isinstance(tool_calls, list) and tool_calls:
                content_text = ""
            if not isinstance(content_text, str):
                raise ValueError("response content is not text")
            if not isinstance(tool_calls, list):
                raise ValueError("response tool_calls is not an array")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise StrataRequestError(
                f"invalid {self.backend_label} response: {exc}", request_attempts=request_attempts,
                latency_seconds=total_latency,
            ) from exc
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        prompt_details = usage.get("prompt_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
        cached_prompt_tokens = prompt_details.get("cached_tokens")
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
        return {
            "backend": self.backend,
            "content": content_text,
            "tool_calls": copy.deepcopy(tool_calls),
            "model": result.get("model", self.model),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cached_prompt_tokens": cached_prompt_tokens,
                "newly_processed_prompt_tokens": (
                    max(0, int(prompt_tokens) - int(cached_prompt_tokens))
                    if prompt_tokens is not None and cached_prompt_tokens is not None else None
                ),
                "prompt_processing_latency_ms": timings.get("prompt_ms"),
            },
            "latency_seconds": round(total_latency, 6),
            "request_attempts": request_attempts,
            "request_ids": list(self._request_ids),
            **({"submitted_messages": copy.deepcopy(submitted_messages)}
               if messages is not None and plain_json_fallback_used else {}),
        }

    def discover_context_window_tokens(self) -> int | None:
        """Read the loaded model's effective context from the endpoint model list."""
        if self.context_window_tokens is not None:
            return self.context_window_tokens
        models_url = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        request = urllib.request.Request(models_url + "/models", method="GET")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 5.0)) as response:
                data = json.loads(response.read().decode("utf-8"))
            models = data.get("data", []) if isinstance(data, dict) else []
            if isinstance(models, list):
                for model in models:
                    if not isinstance(model, dict):
                        continue
                    meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
                    limit = meta.get("n_ctx") or model.get("max_model_len")
                    if limit is not None and int(limit) > 0:
                        self.context_window_tokens = int(limit)
                        return self.context_window_tokens
        except (OSError, ValueError, TypeError, urllib.error.URLError):
            pass
        return None

    def _post_json(
        self,
        payload: dict[str, Any],
        *,
        conversation_message_count: int | None = None,
        conversation_reset_count: int | None = None,
    ) -> tuple[str, float]:
        self._request_sequence += 1
        request_id = uuid.uuid4().hex
        self._request_ids.append(request_id)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        if self.log_request:
            messages = payload.get("messages", [])
            def text_length(message: dict[str, Any]) -> int:
                content = message.get("content", "")
                if isinstance(content, str):
                    return len(content)
                return sum(
                    len(block.get("text", ""))
                    for block in content if isinstance(block, dict) and block.get("type") == "text"
                ) if isinstance(content, list) else 0

            text_lengths = [text_length(message) for message in messages]
            prompt_characters = sum(text_lengths)
            latest_user = next((message for message in reversed(messages) if message.get("role") == "user"), {})
            user_content = latest_user.get("content", "")
            if isinstance(user_content, list):
                user_prompt_characters = next((
                    len(block.get("text", "")) for block in user_content
                    if isinstance(block, dict) and block.get("type") == "text"
                ), 0)
            else:
                user_prompt_characters = len(user_content) if isinstance(user_content, str) else 0
            append_jsonl(self.log_dir / "model_requests.jsonl", {
                "request_id": request_id,
                "sequence": self._request_sequence,
                "backend": self.backend,
                "endpoint": self.endpoint,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "payload_bytes": len(request.data or b""),
                "prompt_characters": prompt_characters,
                "system_prompt_characters": sum(
                    text_length(message) for message in messages if message.get("role") == "system"
                ),
                "user_prompt_characters": user_prompt_characters,
                "user_text_characters": sum(
                    text_length(message) for message in messages if message.get("role") == "user"
                ),
                "image_count": sum(
                    block.get("type") == "image_url"
                    for message in payload.get("messages", [])
                    if isinstance(message.get("content"), list)
                    for block in message["content"] if isinstance(block, dict)
                ),
                "conversation_message_count": conversation_message_count,
                "conversation_reset_count": conversation_reset_count,
                "payload": payload,
            })
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                status = getattr(response, "status", None)
            self._log_response(
                request_id, status, response_body, latency_seconds=time.monotonic() - started,
                conversation_message_count=conversation_message_count,
                conversation_reset_count=conversation_reset_count,
            )
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - started
            raw_detail = exc.read().decode("utf-8", errors="replace")
            self._log_response(
                request_id, exc.code, raw_detail, latency_seconds=elapsed,
                conversation_message_count=conversation_message_count,
                conversation_reset_count=conversation_reset_count,
            )
            detail = raw_detail[:2000]
            try:
                error_data = json.loads(detail).get("error", {})
            except (json.JSONDecodeError, AttributeError):
                error_data = {}
            structured_output_failed = (
                exc.code == 502
                and isinstance(error_data, dict)
                and error_data.get("code") == "structured_output_failed"
            )
            raise StrataRequestError(
                f"{self.backend_label} HTTP {exc.code}: {detail}", latency_seconds=elapsed,
                structured_output_failed=structured_output_failed,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed = time.monotonic() - started
            self._log_response(
                request_id, None, None, error=str(exc), latency_seconds=elapsed,
                conversation_message_count=conversation_message_count,
                conversation_reset_count=conversation_reset_count,
            )
            raise StrataRequestError(
                f"{self.backend_label} request failed: {exc}", latency_seconds=elapsed,
            ) from exc
        return response_body, time.monotonic() - started

    def _log_response(
        self,
        request_id: str,
        status: int | None,
        body: str | None,
        *,
        error: str | None = None,
        latency_seconds: float | None = None,
        conversation_message_count: int | None = None,
        conversation_reset_count: int | None = None,
    ) -> None:
        if not self.log_response:
            return
        diagnostics: dict[str, Any] = {
            "backend": self.backend,
            "conversation_message_count": conversation_message_count,
            "conversation_reset_count": conversation_reset_count,
        }
        try:
            parsed = json.loads(body) if body else {}
            usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
            usage = usage if isinstance(usage, dict) else {}
            prompt_tokens = usage.get("prompt_tokens")
            details = usage.get("prompt_tokens_details", {})
            details = details if isinstance(details, dict) else {}
            cached = details.get("cached_tokens")
            timings = parsed.get("timings", {}) if isinstance(parsed, dict) else {}
            timings = timings if isinstance(timings, dict) else {}
            diagnostics.update({
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": cached,
                "newly_processed_prompt_tokens": (
                    max(0, int(prompt_tokens) - int(cached))
                    if prompt_tokens is not None and cached is not None else None
                ),
                "prompt_processing_latency_ms": timings.get("prompt_ms"),
            })
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        append_jsonl(self.log_dir / "model_responses.jsonl", {
            "request_id": request_id,
            "sequence": self._request_sequence,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": latency_seconds,
            "body": body,
            "error": error,
            **diagnostics,
        })

    @staticmethod
    def _plain_json_fallback(payload: dict[str, Any]) -> dict[str, Any]:
        """Retry a rejected JSON-mode answer as text for local validation/repair."""
        fallback = dict(payload)
        fallback.pop("response_format", None)
        messages = [dict(message) for message in payload["messages"]]
        fallback["messages"] = messages
        user_content = messages[-1].get("content")
        repair_note = (
            "\n\nThe previous generation did not produce valid JSON. "
            "Return exactly one compact, valid JSON object, with double-quoted keys and strings. "
            "Do not use markdown fences or add commentary."
        )
        if isinstance(user_content, list):
            copied_content = [dict(block) if isinstance(block, dict) else block for block in user_content]
            messages[-1]["content"] = copied_content
            text_block = next((block for block in copied_content if block.get("type") == "text"), None)
            if text_block is not None:
                text_block["text"] = str(text_block.get("text", "")) + repair_note
        elif isinstance(user_content, str):
            messages[-1]["content"] = user_content + repair_note
        return fallback


__all__ = [
    "DEFAULT_REASONING_EFFORT", "DEFAULT_STRATA_BASE_URL", "DEFAULT_STRATA_MODEL",
    "DEFAULT_VLLM_BASE_URL", "DEFAULT_VLLM_MODEL", "DEFAULT_MAX_OUTPUT_TOKENS",
    "StrataClient", "StrataRequestError", "V11_VERSION", "user_message",
]
