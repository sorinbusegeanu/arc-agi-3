"""Small OpenAI-compatible multimodal client for a configured Strata endpoint."""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from .runlog import append_jsonl


DEFAULT_STRATA_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_STRATA_MODEL = "qwen3.8-flash-next-iq3_s"
DEFAULT_REASONING_EFFORT = "medium"


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
        max_output_tokens: int = 1024,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        reasoning_budget_tokens: int | None = None,
        log_request: bool = False,
        log_response: bool = False,
        log_dir: str | Path | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model).strip()
        if not self.base_url or not self.model:
            raise ValueError("base_url and model are required")
        self.api_key = api_key if api_key is not None else os.environ.get("STRATA_API_KEY")
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
        self.log_request = bool(log_request)
        self.log_response = bool(log_response)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if (self.log_request or self.log_response) and self.log_dir is None:
            raise ValueError("log_dir is required when request or response logging is enabled")
        self._request_sequence = 0

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
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for index, path in enumerate(image_paths):
            label = image_labels[index] if index < len(image_labels) else f"Image {index + 1}"
            payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": 0,
            "reasoning_effort": self.reasoning_effort,
        }
        if response_format is not None:
            payload["response_format"] = {"type": str(response_format)}
        if self.reasoning_budget_tokens is not None:
            payload["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        request_attempts = 1
        total_latency = 0.0
        try:
            response_body, total_latency = self._post_json(payload)
        except StrataRequestError as exc:
            if not exc.structured_output_failed:
                raise
            request_attempts = 2
            total_latency = exc.latency_seconds
            fallback_payload = self._plain_json_fallback(payload)
            try:
                response_body, fallback_latency = self._post_json(fallback_payload)
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
            if isinstance(content_text, list):
                content_text = "".join(
                    part.get("text", "") for part in content_text if isinstance(part, dict)
                )
            if not isinstance(content_text, str):
                raise ValueError("response content is not text")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise StrataRequestError(
                f"invalid Strata response: {exc}", request_attempts=request_attempts,
                latency_seconds=total_latency,
            ) from exc
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return {
            "content": content_text,
            "model": result.get("model", self.model),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "latency_seconds": round(total_latency, 6),
            "request_attempts": request_attempts,
        }

    def _post_json(self, payload: dict[str, Any]) -> tuple[str, float]:
        self._request_sequence += 1
        request_id = uuid.uuid4().hex
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
            append_jsonl(self.log_dir / "model_requests.jsonl", {
                "request_id": request_id,
                "sequence": self._request_sequence,
                "endpoint": self.endpoint,
                "payload": payload,
            })
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                status = getattr(response, "status", None)
            self._log_response(request_id, status, response_body)
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - started
            raw_detail = exc.read().decode("utf-8", errors="replace")
            self._log_response(request_id, exc.code, raw_detail)
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
                f"Strata HTTP {exc.code}: {detail}", latency_seconds=elapsed,
                structured_output_failed=structured_output_failed,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed = time.monotonic() - started
            self._log_response(request_id, None, None, error=str(exc))
            raise StrataRequestError(
                f"Strata request failed: {exc}", latency_seconds=elapsed,
            ) from exc
        return response_body, time.monotonic() - started

    def _log_response(
        self,
        request_id: str,
        status: int | None,
        body: str | None,
        *,
        error: str | None = None,
    ) -> None:
        if not self.log_response:
            return
        append_jsonl(self.log_dir / "model_responses.jsonl", {
            "request_id": request_id,
            "sequence": self._request_sequence,
            "status": status,
            "body": body,
            "error": error,
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
    "StrataClient", "StrataRequestError",
]
