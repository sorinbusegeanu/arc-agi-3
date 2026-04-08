from __future__ import annotations

from abc import ABC, abstractmethod


class BaseToonClient(ABC):
    @abstractmethod
    def call_text(self, *, prompt: str, bootstrap_context: str, endpoint_name: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def call_video(self, *, prompt: str, video_path: str, endpoint_name: str | None = None) -> str:
        raise NotImplementedError
