from __future__ import annotations

from v4_5.advisory.base import AdvisoryBackend
from v4_5.contracts import AdvisoryRequest, AdvisoryResponse, SCHEMA_VERSION


class NullAdvisor(AdvisoryBackend):
    def advise(self, request: AdvisoryRequest) -> AdvisoryResponse:
        return AdvisoryResponse(
            schema_version=SCHEMA_VERSION,
            agent_name="NullAdvisor",
            round_id=request.round_id,
            advisory_only=True,
            suggestions=(),
            rationale_codes=("NO_ADVISORY",),
        )
