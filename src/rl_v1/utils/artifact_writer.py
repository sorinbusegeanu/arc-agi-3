from __future__ import annotations

from pathlib import Path

from rl_v1.utils.io import append_jsonl, write_json


class ArtifactWriter:
    def __init__(self, root) -> None:
        self.root = Path(root)

    def write_eval_summary(self, name: str, payload: dict) -> None:
        write_json(self.root / f"{name}_summary.json", payload)

    def write_episode_summaries(self, name: str, rows: list[dict]) -> None:
        append_jsonl(self.root / f"{name}_episodes.jsonl", rows)

    def write_planner_trace(self, name: str, rows: list[dict]) -> None:
        # Preserve per-step top-k action probability fields when present
        # (top1_action_id/top1_action_prob/top2_action_id/top2_action_prob).
        write_json(self.root / f"{name}_planner_trace.json", rows)

    def write_effective_config(self, payload: dict) -> None:
        write_json(self.root / "effective_config.json", payload)

    def write_preflight_report(self, payload: dict) -> None:
        write_json(self.root / "preflight_report.json", payload)

    def write_smoke_test_report(self, payload: dict) -> None:
        write_json(self.root / "smoke_test_report.json", payload)

    def write_gradient_diagnostics_row(self, row: dict) -> None:
        append_jsonl(self.root / "gradient_diagnostics.jsonl", [row])

    def write_world_pretrain_summary(self, payload: dict) -> None:
        write_json(self.root / "world_pretrain_summary.json", payload)

    def write_world_eval_summary(self, payload: dict) -> None:
        write_json(self.root / "world_eval_summary.json", payload)

    def write_world_eval_per_game(self, payload: dict) -> None:
        write_json(self.root / "world_eval_per_game.json", payload)

    def write_world_pretrain_update_rows(self, rows: list[dict]) -> None:
        append_jsonl(self.root / "world_pretrain_updates.jsonl", rows)
