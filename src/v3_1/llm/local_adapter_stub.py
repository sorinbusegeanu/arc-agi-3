from __future__ import annotations

from v3_1.llm.local_adapter_base import LocalLLMAdapter, LocalLLMRequest, LocalLLMResponse


class StubLocalLLMAdapter(LocalLLMAdapter):
    def generate_structured_json(self, request: LocalLLMRequest) -> LocalLLMResponse:
        del request
        return LocalLLMResponse(
            ok=False,
            raw_text="",
            parsed_json=None,
            error_code="adapter_not_configured",
            error_message="local llm adapter is not configured",
            latency_ms=0,
            model_name="stub",
        )
