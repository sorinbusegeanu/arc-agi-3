"""focused_explorer.py — Module 5.

Reward-shaped exploration toward unvisited POIs.
Uses a FrontierQueue to ensure the agent visits POIs in order of confidence.
Calls ConsequenceAnalyser on arrival to update HypothesisStore.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, List, Optional, Tuple

import numpy as np

from .structs import EpisodeRecord, POIRecord, ConsequenceResult
from .hypothesis_store import HypothesisStore
from .consequence_analyser import ConsequenceAnalyser
from .config import (
    MAX_STEPS_PER_EP, K_PROXIMITY_PX, ALPHA_REWARD, STUCK_STEPS, MAX_SPRITE_AREA,
    MIN_MOVEMENT_DIST,
)
from .utils import to_action_key as _to_action_key

# Import RewardShaper for base reward — do not replicate its logic
try:
    from arc_agi_agent.rl.reward_shaper import RewardShaper as _RewardShaper
    _REWARD_SHAPER = _RewardShaper()
except Exception:
    _REWARD_SHAPER = None  # type: ignore

# Components for fallback centroid extraction (Bug 1)
try:
    from arc_agi_agent.components import extract_components as _extract_components
    from arc_agi_agent.components import bbox_iou as _bbox_iou
    _HAS_COMPONENTS = True
except ImportError:
    _HAS_COMPONENTS = False

# normalize_observation for done signal (Bug 4)
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


# ── Grid/obs helpers (same as in random_explorer) ────────────────────────────

def _extract_grid(obs: Any) -> np.ndarray:
    frame = getattr(obs, "frame", None)
    if frame is not None and len(frame) > 0:
        arr = np.asarray(frame[0])
        if arr.ndim >= 2:
            return arr if arr.ndim == 2 else arr[:, :, 0]
    if isinstance(obs, dict):
        for key in ("frame", "grid", "observation"):
            val = obs.get(key)
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim >= 2:
                    return arr if arr.ndim == 2 else arr[0]
    return np.zeros((64, 64), dtype=np.uint8)


def _available_actions(obs: Any) -> List[Any]:
    avail = getattr(obs, "available_actions", None)
    if avail:
        return list(avail)
    if isinstance(obs, dict):
        avail = obs.get("available_actions")
        if avail:
            return list(avail)
    return []


def _step_terminal(obs: Any):
    """Return (done: bool, state_str: str) from step obs. Bug 4 fix."""
    if _HAS_NORMALIZE:
        norm = _normalize(obs, schema_warnings=[])
        done = bool(norm.meta.get("terminal", False))
        state_str = _norm_state(norm.meta.get("state"))
        if not done and state_str in ("won", "lost"):
            done = True
        return done, state_str
    state_val = getattr(obs, "state", None)
    state_str = _norm_state(state_val)
    done = bool(getattr(obs, "terminal", False)) or state_str in ("won", "lost")
    return done, state_str


# ── Fallback centroid extractor (Bug 1 / Bug E) ──────────────────────────────

_ALL_COLORS = list(range(16))


def _small_components(comps, percentile: int = 25, hard_max: int = 200):
    """Keep the bottom `percentile`% of components by area, up to hard_max.

    Self-adjusts to the game's actual component size distribution rather than
    using a fixed MAX_SPRITE_AREA cap that may exclude all components.
    Always includes components up to at least 4 cells.
    """
    if not comps:
        return []
    areas = sorted(c.area for c in comps)
    idx = max(0, len(areas) * percentile // 100)
    cutoff = areas[idx]
    cutoff = min(cutoff, hard_max)
    cutoff = max(cutoff, 4)
    return [c for c in comps if c.area <= cutoff]


def _extract_position_fallback(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    self_hint: Optional[POIRecord] = None,
    min_dist: float = MIN_MOVEMENT_DIST,
) -> Optional[Tuple[int, int]]:
    """Find the small, isolated component that moved most between frames.

    self_hint: if provided, filter to matching color first to reduce false matches.
    Uses percentile-based area filter (_small_components) instead of hard MAX_SPRITE_AREA
    so the pool is never empty due to an overly tight cap.
    Returns (x, y) = (col, row) or None.
    """
    if not _HAS_COMPONENTS:
        return None
    try:
        raw_prev = _extract_components(
            frame_prev, colors=_ALL_COLORS, connectivity=8, min_area=4, max_objects=64
        )
        raw_curr = _extract_components(
            frame_curr, colors=_ALL_COLORS, connectivity=8, min_area=4, max_objects=64
        )
    except Exception:
        return None

    # Bug E1 fix: percentile-based filter instead of hard MAX_SPRITE_AREA cap
    all_small_prev = _small_components(raw_prev)
    all_small_curr = _small_components(raw_curr)

    logger.debug(
        "fallback_pool_before_hint curr=%d prev=%d",
        len(all_small_curr), len(all_small_prev),
    )

    # If self_hint available: prefer components matching SELF color
    comps_prev, comps_curr = all_small_prev, all_small_curr
    if self_hint and self_hint.color_signature:
        self_color = self_hint.color_signature[0]
        hint_prev = [c for c in all_small_prev if c.color == self_color]
        hint_curr = [c for c in all_small_curr if c.color == self_color]
        if hint_prev and hint_curr:
            comps_prev, comps_curr = hint_prev, hint_curr
            logger.debug(
                "fallback_pool_after_hint curr=%d prev=%d hint_color=%d",
                len(comps_curr), len(comps_prev), self_color,
            )
        else:
            # Bug E2 fix: warn and use full pool instead of silently failing
            logger.warning(
                "hint_filter_skipped: color=%d not found in small components "
                "(hint_curr=%d hint_prev=%d) — using full pool",
                self_color, len(hint_curr), len(hint_prev),
            )
            logger.debug(
                "fallback_pool_after_hint curr=%d prev=%d hint_color=%d",
                len(comps_curr), len(comps_prev), self_color,
            )

    if not comps_prev or not comps_curr:
        return None

    best_dist = 0.0
    best_centroid = None

    for c_curr in comps_curr:
        best_iou = 0.0
        best_prev = None
        for c_prev in comps_prev:
            if c_curr.color != c_prev.color:
                continue
            iou = _bbox_iou(c_prev.bbox, c_curr.bbox)
            if iou > best_iou:
                best_iou = iou
                best_prev = c_prev
        if best_prev is None:
            continue
        dy = c_curr.centroid[0] - best_prev.centroid[0]
        dx = c_curr.centroid[1] - best_prev.centroid[1]
        dist = (dy ** 2 + dx ** 2) ** 0.5
        logger.debug(
            "fallback_candidate color=%d area=%d iou=%.2f dist=%.2f centroid=%s",
            c_curr.color, c_curr.area, best_iou, dist, c_curr.centroid,
        )
        if dist > best_dist:
            best_dist = dist
            best_centroid = c_curr.centroid  # (row, col)

    # Bug E3 fix: threshold from config
    if best_centroid is None or best_dist < min_dist:
        logger.debug("fallback_no_movement best_dist=%.3f threshold=%.3f", best_dist, min_dist)
        return None
    # Return as (x, y) = (col, row)
    return (int(round(best_centroid[1])), int(round(best_centroid[0])))


# ── Base reward via RewardShaper ──────────────────────────────────────────────

def _base_reward(
    grid_prev: np.ndarray,
    grid_curr: np.ndarray,
    win: bool,
    done: bool,
    step_idx: int,
    game_id: str,
) -> float:
    if _REWARD_SHAPER is None:
        return 1.0 if win else 0.0
    try:
        result = _REWARD_SHAPER.compute(
            event={},
            done=done,
            win=win,
            t=step_idx,
            ctx={"grid_prev": grid_prev, "grid_curr": grid_curr, "game_id": game_id},
        )
        return float(result.get("r_total", 0.0))
    except Exception:
        return 1.0 if win else 0.0


# ── POI proximity reward shaping ──────────────────────────────────────────────

def _shaped_reward(
    base: float,
    position: Optional[Tuple[int, int]],
    target: Optional[POIRecord],
    alpha: float = ALPHA_REWARD,
) -> float:
    if target is None or position is None:
        return base
    y0, x0, y1, x1 = target.bbox
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    px, py = position
    dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return base + alpha * (1.0 / (dist + 1.0))


# ── FrontierQueue ─────────────────────────────────────────────────────────────

class FrontierQueue:
    """Ordered queue of unvisited POIs. Rebuilds from live store after each visit/episode.

    Bug C fix: queue refreshes every episode and after every visit so visited POIs
    are never re-targeted and newly reachable POIs are immediately available.
    """

    def __init__(self, store: HypothesisStore, cfg: dict):
        self.store = store
        self._stuck_steps_limit = int(cfg.get("stuck_steps", STUCK_STEPS))
        self._queue: List[POIRecord] = []
        self._stuck_steps: int = 0
        # diagnostics
        self.refresh_count: int = 0
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild from live store state. Reads .visited and .depriority live."""
        self._queue = self.store.get_targets()
        self.refresh_count += 1
        logger.debug("frontier_refresh n=%d queue_len=%d", self.refresh_count, len(self._queue))

    def current_target(self) -> Optional[POIRecord]:
        if not self._queue:
            self._refresh()
        return self._queue[0] if self._queue else None

    def mark_visited(self, poi_id: str, result: ConsequenceResult) -> None:
        """Record consequence, remove from queue, refresh for next episode."""
        self.store.record_consequence(poi_id, result)
        self._queue = [p for p in self._queue if p.poi_id != poi_id]
        self._stuck_steps = 0
        self._refresh()
        logger.info(
            "poi_visited poi_id=%s consequence=%s remaining_targets=%d",
            poi_id[:8], result.label, len(self._queue),
        )

    def skip_current(self) -> None:
        if self._queue:
            skipped = self._queue.pop(0)
            logger.info("poi_skipped poi_id=%s", skipped.poi_id[:8])
        self._stuck_steps = 0
        if not self._queue:
            self._refresh()

    def tick(self, moved: bool) -> None:
        """Call every step. Triggers skip if stuck too long."""
        if not moved:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if self._stuck_steps >= self._stuck_steps_limit:
            self.skip_current()


# ── FocusedExplorer ───────────────────────────────────────────────────────────

class FocusedExplorer:
    """Run M episodes with reward shaped toward unvisited POIs in the frontier queue."""

    def __init__(
        self,
        env_factory,
        store: HypothesisStore,
        cfg: dict,
        seed: int,
        sprite_detector=None,   # SpriteDetector instance (optional, for centroid extraction)
    ):
        self._factory = env_factory
        self._store = store
        self._cfg = cfg
        self._seed = seed
        self._max_steps = int(cfg.get("max_steps_per_ep", MAX_STEPS_PER_EP))
        self._k_prox = int(cfg.get("k_proximity_px", K_PROXIMITY_PX))
        self._alpha = float(cfg.get("alpha_reward", ALPHA_REWARD))
        self._stuck_steps = int(cfg.get("stuck_steps", STUCK_STEPS))
        self._sprite_det = sprite_detector
        self._ca = ConsequenceAnalyser(cfg)
        # diagnostics set during run()
        self.last_frontier_refreshes: int = 0
        self.last_queue_exhausted_episodes: int = 0

    def _get_self_record(self) -> Optional[POIRecord]:
        """Fetch the current SELF POIRecord from store (live, not cached)."""
        matches = [p for p in self._store.get_all() if p.tag == "SELF"]
        return matches[0] if matches else None

    def _get_position(
        self,
        frame_prev: Optional[np.ndarray],
        frame_curr: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Always derives position from current frame. Never returns a stale stored bbox.

        Uses _extract_position_fallback() with SELF color hint when available (Bug B fix).
        Falls back to stored bbox centroid only on the very first step (no prev frame).
        """
        self_record = self._get_self_record()

        if frame_prev is not None:
            min_dist = float(self._cfg.get("min_movement_dist", MIN_MOVEMENT_DIST))
            pos = _extract_position_fallback(frame_prev, frame_curr, self_hint=self_record, min_dist=min_dist)
            if pos is not None:
                return pos

        # First step only — no movement to detect yet; use stored bbox as seed position
        if self_record is not None and frame_prev is None:
            cy = (self_record.bbox[0] + self_record.bbox[2]) / 2.0
            cx = (self_record.bbox[1] + self_record.bbox[3]) / 2.0
            return (int(round(cx)), int(round(cy)))

        return None

    def run(self, m_episodes: int) -> List[EpisodeRecord]:
        """Run m_episodes with POI-shaped reward. Returns list of EpisodeRecord."""
        records: List[EpisodeRecord] = []
        rng = random.Random(self._seed)

        frontier = FrontierQueue(self._store, self._cfg)
        queue_exhausted_eps = 0

        for ep_idx in range(m_episodes):
            # Bug C fix: refresh at start of every episode from live store state
            frontier._refresh()
            if frontier.current_target() is None:
                queue_exhausted_eps += 1

            env, game_id, env_seed = self._factory(ep_idx)
            obs = env.reset()

            frames: List[np.ndarray] = []
            actions_taken: List[str] = []
            positions: List[Optional[Tuple[int, int]]] = []

            grid = _extract_grid(obs)
            frames.append(grid.copy())
            prev_grid = grid.copy()
            pos = self._get_position(None, grid)
            positions.append(pos)

            episode_terminal = False
            episode_exit_state = ""
            done = False
            win = False
            prev_pos = pos

            for step in range(self._max_steps):
                if done:
                    break

                avail = _available_actions(obs)
                if not avail:
                    break

                target = frontier.current_target()

                action = rng.choice(avail)
                action_key = _to_action_key(action)  # Bug D fix: always canonical string

                obs = env.step(action)
                curr_grid = _extract_grid(obs)
                done, state_str = _step_terminal(obs)
                win = state_str == "won"

                curr_pos = self._get_position(prev_grid, curr_grid)

                # Shaped reward (logged but not used to update any model in this module)
                base = _base_reward(prev_grid, curr_grid, win, done, step, game_id)
                _ = _shaped_reward(base, curr_pos, target, self._alpha)

                # Arrival detection
                near = False
                if target is not None and curr_pos is not None:
                    near = self._ca.is_near_poi(curr_pos, target)
                    if near:
                        ca_result = self._ca.analyse(prev_grid, curr_grid)
                        frontier.mark_visited(target.poi_id, ca_result)
                        if ca_result.label == "GAME_WON":
                            actions_taken.append(action_key + ":GAME_WON")
                            frames.append(curr_grid.copy())
                            positions.append(curr_pos)
                            episode_terminal = True
                            episode_exit_state = "won"
                            done = True
                            break

                logger.debug(
                    "step=%d pos=%s target=%s near=%s done=%s",
                    step, curr_pos,
                    target.poi_id[:8] if target else None,
                    near, done,
                )

                # Stuck detection via FrontierQueue.tick()
                moved = (
                    prev_pos is not None and curr_pos is not None
                    and math.sqrt(
                        (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
                    ) >= 1.0
                )
                frontier.tick(moved)

                frames.append(curr_grid.copy())
                actions_taken.append(action_key)
                positions.append(curr_pos)

                prev_grid = curr_grid.copy()
                prev_pos = curr_pos

                if done:
                    episode_terminal = True
                    episode_exit_state = state_str or "terminal"
                    logger.info(
                        "episode_done ep=%d step=%d exit_state=%s",
                        ep_idx, step, episode_exit_state,
                    )
                    break

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
                "focused_explorer ep=%d game=%s seed=%d steps=%d terminal=%s exit=%s",
                ep_idx, game_id, env_seed, len(actions_taken), episode_terminal, episode_exit_state,
            )

        # Store diagnostics for analysis_loop
        self.last_frontier_refreshes = frontier.refresh_count
        self.last_queue_exhausted_episodes = queue_exhausted_eps
        return records
