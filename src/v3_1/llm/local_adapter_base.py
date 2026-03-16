from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LocalLLMRequest:
    payload_json: dict[str, Any]
    schema_name: str
    schema_version: str
    max_output_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    enable_thinking: bool
    stream: bool
    round_id: int
    session_id: str


@dataclass(frozen=True)
class LocalLLMResponse:
    ok: bool
    raw_text: str
    parsed_json: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    latency_ms: int
    model_name: str


class LocalLLMAdapter:
    def generate_structured_json(self, request: LocalLLMRequest) -> LocalLLMResponse:
        raise NotImplementedError
