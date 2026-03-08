from __future__ import annotations

from typing import Any, Dict, Optional

from codex_baseline_v2.adapters.trajectory_import import import_legacy_trajectories


class RolloutCollectionAdapterV2:
    """Thin adapter to collect legacy rollouts via arc_agi_agent RolloutCollector.

    Caller must provide env_factory and module map consistent with the legacy stack.
    """

    def __init__(self, collector_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.collector_cfg = collector_cfg or {}

    def collect_legacy_batch(
        self,
        env_factory: Any,
        modules: Dict[str, Any],
        rollout_cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            from arc_agi_agent.rl.rollout_collector import RolloutCollector
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Legacy RolloutCollector not available") from exc

        collector = RolloutCollector(cfg=self.collector_cfg)
        return collector.collect(env_factory=env_factory, modules=modules, cfg=rollout_cfg, ctx=ctx)

    def collect_v2_episodes(
        self,
        env_factory: Any,
        modules: Dict[str, Any],
        rollout_cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
        game_id_override: Optional[str] = None,
    ):
        batch = self.collect_legacy_batch(env_factory, modules, rollout_cfg=rollout_cfg, ctx=ctx)
        return import_legacy_trajectories(batch, game_id_override=game_id_override)
