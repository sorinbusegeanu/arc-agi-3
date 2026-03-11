from __future__ import annotations

import argparse
import json

from codex_baseline_v2.inference.dependency_updater import update_dependency_graph
from codex_baseline_v2.inference.latent_state_inducer import induce_latent_states
from codex_baseline_v2.inference.mechanic_graph_builder import build_mechanic_graph
from codex_baseline_v2.memory.graph_store import save_graph_state
from codex_baseline_v2.memory.store import load_blackboard, save_blackboard
from codex_baseline_v2.shared.config import load_config
from codex_baseline_v2.shared.schemas import BlackboardStateV2
from codex_baseline_v2.shared.storage import StoragePathsV2


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze causal world structures from latest V2 blackboard")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = load_config(json.load(handle))
    storage = StoragePathsV2(cfg.memory.storage_dir)
    blackboard = load_blackboard(storage, cfg.game_id)
    if blackboard is None:
        raise SystemExit("No blackboard state found.")

    latent_states = induce_latent_states(
        blackboard.event_table,
        blackboard.intervention_table,
        blackboard.trigger_zone_table,
        blackboard.event_edge_table,
        blackboard.area_table,
        existing=blackboard.latent_states,
    )
    mechanic_graph = build_mechanic_graph(
        blackboard.cause_effect_table,
        blackboard.event_table,
        latent_states,
        blackboard.topology_delta_table,
        blackboard.causal_chain_hypotheses,
        existing=blackboard.mechanic_graph,
        round_id=blackboard.round_id,
        step_id=max((event.end_step_idx for event in blackboard.event_table), default=0),
    )
    dependency_graph = update_dependency_graph(
        mechanic_graph,
        blackboard.reachability_table,
        existing=blackboard.dependency_graph,
        round_id=blackboard.round_id,
        step_id=max((event.end_step_idx for event in blackboard.event_table), default=0),
    )
    updated = BlackboardStateV2.from_dict(
        {
            **blackboard.to_dict(),
            "latent_states": [row.to_dict() for row in latent_states],
            "mechanic_graph": mechanic_graph.to_dict(),
            "dependency_graph": dependency_graph.to_dict(),
        }
    )
    save_blackboard(cfg.memory, storage, updated)
    save_graph_state(storage, updated)


if __name__ == "__main__":
    main()
