"""random_explorer.py — Module 1.

Pure random-action agent. Runs N episodes, logs frames/actions/positions.
No reward shaping; no model updates.
"""
from __future__ import annotations

import logging
import random
from typing import Any, List, Optional, Tuple

import numpy as np

from .structs import EpisodeRecord
from .poi_detector import SpriteDetector   # for extract_centroid (optional, None on first pass)
from .config import MAX_STEPS_PER_EP
from .utils import to_action_key as _to_action_key

try:
    from arc_agi_agent.normalize import normalize_observation as _normalize
    _HAS_NORMALIZE = True
except ImportError:
    _HAS_NORMALIZE = False

logger = logging.getLogger(__name__)


def _norm_state(state_val) -> str:
    """Convert raw state enum/string to 'won', 'lost', or ''."""
    if state_val is None:
        return ""
    s = getattr(state_val, "name", str(state_val)).lower()
    if "win" in s or s == "won":
        return "won"
    if any(x in s for x in ("loss", "lose", "over", "fail", "lost", "dead")):
        return "lost"
    return ""


def _extract_grid(obs: Any) -> np.ndarray:
    """Pull primary grid from raw obs (FrameDataRaw or dict fallback)."""
    # Try FrameDataRaw
    frame = getattr(obs, "frame", None)
    if frame is not None and len(frame) > 0:
        arr = np.asarray(frame[0])
        if arr.ndim >= 2:
            return arr[:, :] if arr.ndim == 2 else arr[:, :, 0]
    # Dict fallback
    if isinstance(obs, dict):
        for key in ("frame", "grid", "observation"):
            val = obs.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim >= 2:
                    return arr if arr.ndim == 2 else arr[0]
    return np.zeros((64, 64), dtype=np.uint8)


def _available_actions(obs: Any) -> List[Any]:
    """Return list of available GameAction objects from obs."""
    avail = getattr(obs, "available_actions", None)
    if avail:
        return list(avail)
    if isinstance(obs, dict):
        avail = obs.get("available_actions")
        if avail:
            return list(avail)
    return []


def _step_terminal(obs: Any):
    """Return (done: bool, state_str: str) from a step obs using normalize_observation."""
    if _HAS_NORMALIZE:
        norm = _normalize(obs, schema_warnings=[])
        done = bool(norm.meta.get("terminal", False))
        state_str = _norm_state(norm.meta.get("state"))
        if not done and state_str in ("won", "lost"):
            done = True
        return done, state_str
    # Fallback: direct attribute access
    state_val = getattr(obs, "state", None)
    state_str = _norm_state(state_val)
    done = bool(getattr(obs, "terminal", False)) or state_str in ("won", "lost")
    return done, state_str


class RandomExplorer:
    """Run N episodes with random actions. Return EpisodeRecord per episode."""

    def __init__(self, env_factory, cfg: dict, seed: int, workers: int = 1):
        self._factory = env_factory
        self._cfg = cfg
        self._seed = seed
        self._max_steps = int(cfg.get("max_steps_per_ep", MAX_STEPS_PER_EP))
        self._workers = max(1, int(workers))
        self._sprite_det: Optional[SpriteDetector] = None

    def _centroid(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._sprite_det is not None:
            return self._sprite_det.extract_centroid(frame)
        return None

    def run(self, n_episodes: int) -> List[EpisodeRecord]:
        """Run n_episodes random episodes. Uses multiple processes when workers > 1."""
        if self._workers > 1 and n_episodes >= self._workers:
            return self._run_parallel(n_episodes)
        return self._run_sequential(n_episodes, ep_offset=0)

    def _run_sequential(self, n: int, ep_offset: int = 0) -> List[EpisodeRecord]:
        """Run n episodes sequentially starting at factory index ep_offset."""
        records: List[EpisodeRecord] = []
        rng = random.Random(self._seed + ep_offset)

        for local_idx in range(n):
            ep_idx = ep_offset + local_idx
            env, game_id, env_seed = self._factory(ep_idx)
            obs = env.reset()

            frames: List[np.ndarray] = []
            actions_taken: List[str] = []
            positions: List[Optional[Tuple[int, int]]] = []

            grid = _extract_grid(obs)
            frames.append(grid.copy())
            positions.append(self._centroid(grid))

            episode_terminal = False
            episode_exit_state = ""
            done = False

            for step in range(self._max_steps):
                if done:
                    break

                avail = _available_actions(obs)
                if not avail:
                    break

                action = rng.choice(avail)
                action_key = _to_action_key(action)

                obs = env.step(action)
                grid = _extract_grid(obs)
                done, state_str = _step_terminal(obs)

                frames.append(grid.copy())
                actions_taken.append(action_key)
                positions.append(self._centroid(grid))

                if done:
                    episode_terminal = True
                    episode_exit_state = state_str or "terminal"
                    logger.info(
                        "episode_done ep=%d step=%d exit_state=%s",
                        ep_idx, step, episode_exit_state,
                    )
                    break

            while len(positions) < len(frames):
                positions.append(None)

            record = EpisodeRecord(
                episode_id=ep_idx,
                frames=frames,
                actions=actions_taken,
                positions=positions,
                terminal=episode_terminal,
                exit_state=episode_exit_state,
            )
            records.append(record)

            logger.info(
                "random_explorer ep=%d game=%s seed=%d steps=%d terminal=%s exit=%s",
                ep_idx, game_id, env_seed, len(actions_taken), episode_terminal, episode_exit_state,
            )

        return records

    def _run_parallel(self, n_episodes: int) -> List[EpisodeRecord]:
        """Split n_episodes across self._workers processes."""
        from concurrent.futures import ProcessPoolExecutor

        chunk_size = max(1, n_episodes // self._workers)
        chunks = []
        for start in range(0, n_episodes, chunk_size):
            count = min(chunk_size, n_episodes - start)
            chunks.append((self._factory, start, count, self._cfg, self._seed))

        logger.info("random_explorer parallel workers=%d chunks=%d", self._workers, len(chunks))
        with ProcessPoolExecutor(max_workers=self._workers) as ex:
            results = list(ex.map(_random_worker_fn, chunks))

        return [ep for chunk_records in results for ep in chunk_records]


# ── Top-level worker function (must be at module level to be picklable) ───────

def _random_worker_fn(args):
    """Worker entry point for RandomExplorer parallel mode."""
    factory, ep_start, ep_count, cfg, seed = args
    explorer = RandomExplorer(factory, cfg, seed=seed, workers=1)
    return explorer._run_sequential(ep_count, ep_offset=ep_start)
