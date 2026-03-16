from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from v3_1.llm.local_adapter_base import LocalLLMAdapter, LocalLLMRequest, LocalLLMResponse


class OpenAICompatLocalLLMAdapter(LocalLLMAdapter):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str | None,
        timeout_sec: float,
        connect_timeout_sec: float | None = None,
        read_timeout_sec: float | None = None,
        retry_limit: int,
        emit_raw_debug: bool = False,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.model_name = str(model_name or "")
        self.api_key_env = str(api_key_env or "")
        self.timeout_sec = float(timeout_sec or 0.0)
        inferred_local_read_timeout = 45.0 if self.base_url.startswith("http://127.0.0.1") or self.base_url.startswith("http://localhost") else self.timeout_sec
        self.connect_timeout_sec = float(connect_timeout_sec if connect_timeout_sec is not None else max(0.1, min(self.timeout_sec or 5.0, 5.0)))
        self.read_timeout_sec = float(read_timeout_sec if read_timeout_sec is not None else max(self.timeout_sec or 0.0, inferred_local_read_timeout))
        self.retry_limit = max(0, int(retry_limit or 0))
        self.emit_raw_debug = bool(emit_raw_debug)

    def generate_structured_json(self, request: LocalLLMRequest) -> LocalLLMResponse:
        started = time.perf_counter()
        if not self.base_url or not self.model_name:
            return LocalLLMResponse(
                ok=False,
                raw_text="",
                parsed_json=None,
                error_code="invalid_adapter_config",
                error_message="missing base_url or model_name",
                latency_ms=int((time.perf_counter() - started) * 1000.0),
                model_name=self.model_name or "openai_compat",
            )
        endpoint = f"{self.base_url}/chat/completions"
        auth_token = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        system_instruction = str(request.payload_json.get("system_instruction") or "")
        user_payload = {
            key: value
            for key, value in dict(request.payload_json).items()
            if key != "system_instruction"
        }
        serialized_user_payload = json.dumps(user_payload, sort_keys=True)
        prompt_diagnostics = {
            "prompt_char_count": len(serialized_user_payload),
            "prompt_approx_token_count": max(1, int((len(serialized_user_payload) / 4.0) + 0.999)),
            "payload_section_counts": dict(request.payload_json.get("payload_section_counts", {}) or {}),
            "prompt_trim_applied": bool(request.payload_json.get("prompt_trim_applied", False)),
        }
        payload = {
            "model": self.model_name,
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "max_tokens": int(request.max_output_tokens),
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction
                    or (
                        "Return only valid JSON matching the requested schema. "
                        "Do not include markdown, explanation, or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": serialized_user_payload,
                },
            ],
            "extra_body": {
                "top_k": int(request.top_k),
                "presence_penalty": float(request.presence_penalty),
                "repetition_penalty": float(request.repetition_penalty),
                "chat_template_kwargs": {"enable_thinking": bool(request.enable_thinking)},
            },
        }
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        raw_text = ""
        last_error_code = None
        last_error_message = None
        for _attempt in range(self.retry_limit + 1):
            try:
                body = json.dumps(payload).encode("utf-8")
                http_request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(http_request, timeout=max(0.1, self.connect_timeout_sec, self.read_timeout_sec, self.timeout_sec or 30.0)) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                raw_text = str((((response_payload.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
                if not raw_text.strip():
                    return LocalLLMResponse(
                        ok=False,
                        raw_text=raw_text if self.emit_raw_debug else "",
                        parsed_json=None,
                        error_code="empty_content",
                        error_message="provider returned empty content",
                        latency_ms=int((time.perf_counter() - started) * 1000.0),
                        model_name=self.model_name,
                    )
                if "<think>" in raw_text.lower():
                    return LocalLLMResponse(
                        ok=False,
                        raw_text=raw_text if self.emit_raw_debug else "",
                        parsed_json=None,
                        error_code="think_content",
                        error_message="provider returned thinking content",
                        latency_ms=int((time.perf_counter() - started) * 1000.0),
                        model_name=self.model_name,
                    )
                parsed = json.loads(raw_text) if raw_text else None
                if isinstance(parsed, dict):
                    metadata = dict(parsed.get("metadata", {}) or {})
                    metadata.update(prompt_diagnostics)
                    metadata["adapter_transport_timeout_sec"] = max(0.1, self.connect_timeout_sec, self.read_timeout_sec, self.timeout_sec or 30.0)
                    metadata["adapter_connect_timeout_sec"] = float(self.connect_timeout_sec)
                    metadata["adapter_read_timeout_sec"] = float(self.read_timeout_sec)
                    parsed["metadata"] = metadata
                return LocalLLMResponse(
                    ok=isinstance(parsed, dict),
                    raw_text=raw_text if self.emit_raw_debug else "",
                    parsed_json=parsed if isinstance(parsed, dict) else None,
                    error_code=None if isinstance(parsed, dict) else "non_json_output",
                    error_message=None if isinstance(parsed, dict) else "provider did not return a valid JSON object",
                    latency_ms=int((time.perf_counter() - started) * 1000.0),
                    model_name=self.model_name,
                )
            except urllib.error.HTTPError as exc:
                last_error_code = f"http_{exc.code}"
                last_error_message = str(exc)
            except urllib.error.URLError as exc:
                last_error_code = "connection_error"
                last_error_message = str(exc)
            except TimeoutError as exc:
                last_error_code = "timeout"
                last_error_message = str(exc)
            except json.JSONDecodeError as exc:
                last_error_code = "json_decode_error"
                last_error_message = str(exc)
            except Exception as exc:  # pragma: no cover
                last_error_code = "adapter_exception"
                last_error_message = str(exc)
        return LocalLLMResponse(
            ok=False,
            raw_text=raw_text if self.emit_raw_debug else "",
            parsed_json=None,
            error_code=last_error_code or "unknown_error",
            error_message=last_error_message or "unknown llm adapter error",
            latency_ms=int((time.perf_counter() - started) * 1000.0),
            model_name=self.model_name,
        )
