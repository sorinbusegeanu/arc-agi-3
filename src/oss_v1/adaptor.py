from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Any
import numpy as np

# Import the generic LSA environment adapter factory
try:
    from llm_stack_agentic.lsa_env_adapter import default_env_adapter
except Exception:  # pragma: no cover - defensive for missing optional dependency
    default_env_adapter = None

class ArcEngineAdaptor:
    """Thin wrapper around the ArcAgiDefaultAdapter that exposes a simple
    interface used by the OSS v1 agent.
    ``game_id`` is the id of the game under ``other_repos/arc-interactive/environment_files``.
    """

    def __init__(self, game_id: str, seed: int = 0):
        if default_env_adapter is None:
            raise ImportError("llm_stack_agentic package not available")
        # Resolve the path to the ARC-Interactive environment files.
        repo_root = Path(__file__).resolve().parents[3]
        env_dir = str(repo_root / "other_repos" / "arc-interactive" / "environment_files")

        self._env = default_env_adapter(op_mode="offline", env_dir=env_dir)
        # ``ArcAgiDefaultAdapter`` expects an episode id string.
        episode_id = f"oss:{game_id}"
        raw_obs = self._env.reset(episode_id, game_id, seed)
        if raw_obs is None:
            raise RuntimeError("Failed to reset environment")
        self._last_raw = raw_obs
        self.game_id = game_id
        self.seed = seed

    # ------------------------------------------------------------------
    # Basic adaptor interface expected by the agent.
    def reset(self) -> np.ndarray:
        """Reset the underlying environment and return a 2‑D numpy array."""
        episode_id = f"oss:{self.game_id}"
        raw_obs = self._env.reset(episode_id, self.game_id, self.seed)
        if raw_obs is None:
            raise RuntimeError("Failed to reset environment")
        self._last_raw = raw_obs
        return np.asarray(self._last_raw.frame[0])

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> np.ndarray:
        """Perform a single environment step.

        Parameters
        ----------
        action : int
            The integer id of the action (1–6).
        x, y : Optional[int]
            Coordinates for click actions (action 6). If omitted the agent will not use coordinates.
        """
        payload: dict[str, Any] = {
            "type": "coord" if action == 6 else "discrete",
            "id": action,
        }
        if action == 6:
            if x is None or y is None:
                raise ValueError("CLICK actions require explicit x and y coordinates")
            payload.update({"x": int(x), "y": int(y)})

        raw_obs, _reward, _done, _info = self._env.step(payload)
        if raw_obs is None:
            raise RuntimeError("Environment step returned None")
        self._last_raw = raw_obs
        return np.asarray(self._last_raw.frame[0])

    def observe(self) -> np.ndarray:
        """Return the most recent observation grid as a 2‑D numpy array."""
        if not hasattr(self, "_last_raw") or self._last_raw is None:
            raise RuntimeError("Environment has not been reset yet")
        return np.asarray(self._last_raw.frame[0])

    def available_actions(self) -> List[int]:
        """Return a list of valid action ids for the current state."""
        if not hasattr(self, "_last_raw"):
            raise RuntimeError("Environment has not been reset yet")
        info = self._env.get_available_actions(self._last_raw)
        # ``discrete_mask`` is a list of bools indexed by action id.
        actions: List[int] = [i for i, flag in enumerate(info["discrete_mask"]) if flag]
        # Include the click action id if it is enabled.
        if info.get("coord_enabled"):
            actions.append(info.get("coord_action_id", 6))
        return sorted(actions)

    @property
    def state(self) -> str | None:
        """Return the current game state string (e.g., WIN, GAME_OVER)."""
        if not hasattr(self, "_last_raw"):
            return None
        return getattr(self._last_raw, "state", None)

    @property
    def last_grid(self) -> np.ndarray:
        return self.observe()
