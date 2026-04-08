from __future__ import annotations

from abc import ABC, abstractmethod

from v4_5.contracts import AdvisoryRequest, AdvisoryResponse


class AdvisoryBackend(ABC):
    @abstractmethod
    def advise(self, request: AdvisoryRequest) -> AdvisoryResponse:
        raise NotImplementedError
