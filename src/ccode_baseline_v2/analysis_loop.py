"""analysis_loop.py — Module 6 (Orchestrator).

Phase 1: RandomExplorer  → N episodes
Phase 2: POIDetector + ConsequenceAnalyser → HypothesisStore
Phase 3: FocusedExplorer → M episodes
Repeat Phase 2–3 until GAME_WON or BUDGET_EXHAUSTED.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .structs import EpisodeRecord
from .config import N_RANDOM_EPISODES, M_FOCUSED_EPISODES, MAX_VERSIONS, default_cfg
from .random_explorer import RandomExplorer
from .poi_detector import POIDetector
from .hypothesis_store import HypothesisStore
from .focused_explorer import FocusedExplorer
from .match_detector import MatchDetector

logger = logging.getLogger(__name__)


def _had_game_won(ep: EpisodeRecord) -> bool:
    """True if episode contains a GAME_WON sentinel or exit_state=='won'."""
    return any("GAME_WON" in a for a in ep.actions) or ep.exit_state == "won"


class AnalysisLoop:
    """Orchestrates the full perception + hypothesis-driven exploration loop."""

    def __init__(
        self,
        env_factory,
        cfg: Optional[dict] = None,
        seed: int = 0,
        store_path: Optional[str] = None,
        out_dir: str = ".",
        workers: int = 1,
    ):
        """
        Parameters
        ----------
        env_factory : callable(ep_idx) → (env, game_id, env_seed)
        cfg         : config dict; defaults from config.default_cfg() if None
        seed        : base random seed
        store_path  : optional path to pre-existing HypothesisStore JSON to resume from
        out_dir     : directory for saving hypothesis store snapshots
        """
        self._factory = env_factory
        self._cfg = cfg if cfg is not None else default_cfg()
        self._seed = seed
        self._out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self._store = HypothesisStore()
        if store_path and os.path.isfile(store_path):
            self._store.load(store_path)
            logger.info("analysis_loop loaded store from %s (version=%d)", store_path, self._store.version)

        self._n_random    = int(self._cfg.get("n_random_episodes", N_RANDOM_EPISODES))
        self._m_focused   = int(self._cfg.get("m_focused_episodes", M_FOCUSED_EPISODES))
        self._max_versions = int(self._cfg.get("max_versions", MAX_VERSIONS))
        self._workers = max(1, int(workers))
        self._history: List[Dict[str, Any]] = []   # per-version metrics
        self._match_detector = MatchDetector()
        self._pattern_match_history: List[Dict[str, Any]] = []
        self._match_detected_version: Optional[int] = None

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def _store_metrics(self, version: int) -> Dict[str, Any]:
        """Snapshot key metrics from the current store state."""
        pois = list(self._store._pois.values())
        tags: Dict[str, int] = {}
        for p in pois:
            tags[p.tag] = tags.get(p.tag, 0) + 1
        consequences: Dict[str, int] = {}
        for p in pois:
            if p.consequence:
                consequences[p.consequence] = consequences.get(p.consequence, 0) + 1
        return {
            "version":          version,
            "total_pois":       len(pois),
            "tags":             tags,
            "reachable":        sum(1 for p in pois if p.reachable),
            "visited":          sum(1 for p in pois if p.visited),
            "deprioritised":    sum(1 for p in pois if p.depriority),
            "consequences":     consequences,
            "targets_available": len(self._store.get_targets()),
            "conf_mean":        round(sum(p.confidence for p in pois) / max(1, len(pois)), 3),
            "conf_max":         round(max((p.confidence for p in pois), default=0.0), 3),
        }

    @staticmethod
    def _episode_metrics(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
        steps = [len(ep.actions) for ep in episodes]
        none_pos = sum(1 for ep in episodes for p in ep.positions if p is None)
        total_pos = sum(len(ep.positions) for ep in episodes)
        return {
            "n_episodes":          len(episodes),
            "steps_mean":          round(sum(steps) / max(1, len(steps)), 1),
            "steps_total":         sum(steps),
            "game_won":            any(_had_game_won(ep) for ep in episodes),
            "terminal_episodes":   sum(1 for ep in episodes if ep.terminal),
            "positions_none_rate": round(none_pos / max(1, total_pos), 3),
        }

    def _save_summary(self, exit_reason: str, elapsed: float) -> None:
        pois = list(self._store._pois.values())
        summary = {
            "exit_reason":    exit_reason,
            "elapsed_sec":    round(elapsed, 1),
            "versions_run":   len(self._history),
            "history":        self._history,
            "final": self._store_metrics(self._store.version) if self._history else {},
            "pattern_match_history": self._pattern_match_history,
            "trigger_poi_visit_counts": {
                p.poi_id: p.visit_count for p in pois if p.visit_count > 1
            },
            "match_detected_version": self._match_detected_version,
            "exit_unlocked": self._match_detected_version is not None,
        }
        path = os.path.join(self._out_dir, "run_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("summary written to %s", path)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> str:
        """
        Run the full analysis loop.

        Returns
        -------
        "GAME_WON" | "BUDGET_EXHAUSTED"
        """
        t0 = time.time()
        store = self._store
        version = store.version

        # Phase 1 — Random exploration
        logger.info("analysis_loop phase=1 n_random=%d", self._n_random)
        explorer = RandomExplorer(self._factory, self._cfg, seed=self._seed, workers=self._workers)
        episodes: List[EpisodeRecord] = explorer.run(self._n_random)

        if any(_had_game_won(ep) for ep in episodes):
            store.save(os.path.join(self._out_dir, "hypothesis_store_final.json"))
            logger.info("analysis_loop exit=GAME_WON (random phase)")
            self._save_summary("GAME_WON", time.time() - t0)
            return "GAME_WON"

        while version < self._max_versions:
            # Phase 2 — Analyse
            version += 1
            logger.info("analysis_loop phase=2 version=%d episodes=%d", version, len(episodes))

            detector = POIDetector(self._cfg)
            new_pois = detector.detect(episodes, store=store)
            merge_result = store.update(new_pois, version)

            sd = detector.sprite_detector
            self_tagged = bool(sd and sd.self_tagged_by)
            corr_scores = sd.last_corr_scores if sd else []
            logger.info(
                "analysis_loop self_tagged=%s tagged_by=%s corr_scores=%s merge=%s",
                self_tagged,
                sd.self_tagged_by if sd else "",
                corr_scores[:3],
                merge_result,
            )

            store_snap = self._store_metrics(version)
            logger.info(
                "analysis_loop pois_total=%d targets=%d visited=%d reachable=%d",
                store_snap["total_pois"], store_snap["targets_available"],
                store_snap["visited"], store_snap["reachable"],
            )

            if not store.get_targets():
                store.save(os.path.join(self._out_dir, f"hypothesis_store_v{version}.json"))
                logger.info("analysis_loop exit=BUDGET_EXHAUSTED (no reachable targets)")
                diag = {
                    **store_snap,
                    "phase3": {},
                    "poi_merges": merge_result.get("merged", 0),
                    "poi_new": merge_result.get("new", 0),
                    "self_tagged": self_tagged,
                    "self_correlation_scores": corr_scores,
                    "self_bbox_updated": bool(sd and sd.self_bbox_updated),
                    "frontier_refreshes": 0,
                    "queue_exhausted_episodes": 0,
                    "action_key_sample": [],
                }
                self._history.append(diag)
                self._save_summary("BUDGET_EXHAUSTED", time.time() - t0)
                return "BUDGET_EXHAUSTED"

            # Phase 3 — Focused exploration
            logger.info("analysis_loop phase=3 version=%d m_focused=%d", version, self._m_focused)
            focused = FocusedExplorer(
                env_factory=self._factory,
                store=store,
                cfg=self._cfg,
                seed=self._seed + version,
                sprite_detector=sd,
                match_detector=self._match_detector,
                workers=self._workers,
            )
            episodes = focused.run(self._m_focused)
            ep_snap = self._episode_metrics(episodes)
            frontier_refreshes = focused.last_frontier_refreshes
            queue_exhausted_eps = focused.last_queue_exhausted_episodes
            action_key_sample = episodes[0].actions[:5] if episodes and episodes[0].actions else []
            self_bbox_updated = bool(sd and sd.self_bbox_updated)

            # Pattern-match bookkeeping
            for entry in focused.pattern_match_history:
                self._pattern_match_history.append({"version": version, **entry})
            last_match = focused.last_match_result
            if last_match is not None and last_match.matched and self._match_detected_version is None:
                self._match_detected_version = version
                logger.info("PATTERN_MATCH version=%d score=%.3f confidence=%s",
                            version, last_match.match_score, last_match.confidence)

            match_snap = {
                "match_score": last_match.match_score if last_match else 0.0,
                "confidence": last_match.confidence if last_match else "none",
                "matched": bool(last_match and last_match.matched),
                "poi_a": last_match.poi_id_a if last_match else None,
                "poi_b": last_match.poi_id_b if last_match else None,
            }

            logger.info(
                "analysis_loop phase3 terminal_eps=%d positions_none_rate=%.3f "
                "frontier_refreshes=%d queue_exhausted_eps=%d self_bbox_updated=%s "
                "match_score=%.3f match_confidence=%s",
                ep_snap["terminal_episodes"], ep_snap["positions_none_rate"],
                frontier_refreshes, queue_exhausted_eps, self_bbox_updated,
                match_snap["match_score"], match_snap["confidence"],
            )

            self._history.append({
                **store_snap,
                "phase3": ep_snap,
                "poi_merges": merge_result.get("merged", 0),
                "poi_new": merge_result.get("new", 0),
                "self_tagged": self_tagged,
                "self_correlation_scores": corr_scores,
                "self_bbox_updated": self_bbox_updated,
                "frontier_refreshes": frontier_refreshes,
                "queue_exhausted_episodes": queue_exhausted_eps,
                "action_key_sample": action_key_sample,
                "match_result": match_snap,
            })

            if ep_snap["game_won"]:
                store.save(os.path.join(self._out_dir, "hypothesis_store_final.json"))
                logger.info("analysis_loop exit=GAME_WON version=%d", version)
                self._save_summary("GAME_WON", time.time() - t0)
                return "GAME_WON"

            store.save(os.path.join(self._out_dir, f"hypothesis_store_v{version}.json"))
            logger.info("analysis_loop saved store v%d", version)

        logger.info("analysis_loop exit=BUDGET_EXHAUSTED (max_versions=%d reached)", self._max_versions)
        store.save(os.path.join(self._out_dir, f"hypothesis_store_v{version}.json"))
        self._save_summary("BUDGET_EXHAUSTED", time.time() - t0)
        return "BUDGET_EXHAUSTED"
