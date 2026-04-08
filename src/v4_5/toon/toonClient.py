from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from v4_5.config.bootstrapConfig import BootstrapConfig
from v4_5.contracts.errors import ToonTextCallError, ToonVideoCallError
from v4_5.toon.baseClient import BaseToonClient


class ToonClient(BaseToonClient):
    def __init__(self, config: BootstrapConfig) -> None:
        self.config = config

    def call_text(self, *, prompt: str, bootstrap_context: str, endpoint_name: str | None = None) -> str:
        try:
            return self._generate(
                model=(endpoint_name or self.config.toon_text_endpoint_name or self.config.ollama_model),
                prompt=f"{prompt}\n\n{bootstrap_context}".strip(),
            )
        except Exception as exc:
            raise ToonTextCallError(str(exc)) from exc

    def call_video(self, *, prompt: str, video_path: str, endpoint_name: str | None = None) -> str:
        try:
            prompt_text = f"{prompt}\n\nVideo artifact path: {video_path}"
            attachments = ()
            path = Path(video_path)
            if path.exists():
                attachments = (base64.b64encode(path.read_bytes()).decode("ascii"),)
            return self._generate(
                model=(endpoint_name or self.config.toon_video_endpoint_name or self.config.ollama_model),
                prompt=prompt_text,
                images=attachments,
            )
        except Exception as exc:
            raise ToonVideoCallError(str(exc)) from exc

    def _generate(self, *, model: str, prompt: str, images: tuple[str, ...] = ()) -> str:
        payload = {
            "model": model or self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": int(self.config.ollama_num_ctx)},
        }
        if images:
            payload["images"] = list(images)
        last_error = None
        for _ in range(max(1, int(self.config.retry_count) + 1)):
            try:
                return self._request_generate(payload)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error) if last_error is not None else "ollama request failed")

    def _request_generate(self, payload: dict) -> str:
        url = self.config.ollama_url.rstrip("/") + "/api/generate"
        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=float(self.config.timeout_sec)) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(str(exc)) from exc
        data = json.loads(raw)
        if isinstance(data, dict) and "response" in data:
            return str(data["response"])
        raise RuntimeError(f"unexpected ollama response: {raw}")
