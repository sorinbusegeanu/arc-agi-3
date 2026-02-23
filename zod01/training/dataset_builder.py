from __future__ import annotations

import argparse
import json
from pathlib import Path

from zod01.training.logio import episode_outcomes, grouped_steps


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def build_ranker_dataset(log_dir: str, out_path: str) -> int:
    outcomes = episode_outcomes(log_dir)
    steps = grouped_steps(log_dir)
    rows: list[dict[str, object]] = []

    for eid, evs in steps.items():
        episode_won = bool(outcomes.get(eid, False))
        for ev in evs:
            chosen = str(ev.get("chosen_action", ""))
            cands = ev.get("candidate_debug", [])
            if not isinstance(cands, list):
                continue
            for c in cands:
                if not isinstance(c, dict):
                    continue
                action = str(c.get("action", ""))
                row = {
                    "episode_id": eid,
                    "game_id": str(ev.get("game_id", "")),
                    "seed": int(ev.get("seed", 0)),
                    "variant_id": str(ev.get("variant_id", "")),
                    "step_idx": int(ev.get("step_idx", 0)),
                    "state_hash": str(ev.get("state_hash", "")),
                    "action": action,
                    "features": c.get("features", [0.0] * 6),
                    "unsafe": int("cycle-risk" in c.get("tags", []) or "complex-risk" in c.get("tags", [])),
                    "label": int(episode_won and action == chosen),
                }
                rows.append(row)

    _write_jsonl(Path(out_path), rows)
    return len(rows)


def build_critic_dataset(log_dir: str, out_path: str) -> int:
    outcomes = episode_outcomes(log_dir)
    steps = grouped_steps(log_dir)
    rows: list[dict[str, object]] = []

    for eid, evs in steps.items():
        episode_won = bool(outcomes.get(eid, False))
        for ev in evs:
            tags = set(ev.get("tags", [])) if isinstance(ev.get("tags", []), list) else set()
            rows.append(
                {
                    "episode_id": eid,
                    "game_id": str(ev.get("game_id", "")),
                    "seed": int(ev.get("seed", 0)),
                    "step_idx": int(ev.get("step_idx", 0)),
                    "action": str(ev.get("chosen_action", "")),
                    "loop_risk": int("cycle-risk" in tags),
                    "irreversible_risk": int("complex-risk" in tags),
                    "dead_end": int((not episode_won) and bool(ev.get("terminal", False))),
                    "episode_won": int(episode_won),
                }
            )

    _write_jsonl(Path(out_path), rows)
    return len(rows)


def build_mechanic_dataset(log_dir: str, out_path: str, k: int = 5) -> int:
    steps = grouped_steps(log_dir)
    rows: list[dict[str, object]] = []

    for eid, evs in steps.items():
        window: list[str] = []
        for ev in evs:
            toks = ev.get("delta_tokens", [])
            if isinstance(toks, list):
                window.extend(str(t) for t in toks)
            window = window[-k:]
            label = "sig:noop" if str(ev.get("delta_kind", "")) == "no_op" else "sig:change"
            rows.append(
                {
                    "episode_id": eid,
                    "game_id": str(ev.get("game_id", "")),
                    "step_idx": int(ev.get("step_idx", 0)),
                    "window_tokens": window[:],
                    "label": label,
                    "action": str(ev.get("chosen_action", "")),
                }
            )

    _write_jsonl(Path(out_path), rows)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Build supervised datasets from zod01 logs")
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="zod01/datasets")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranker_rows = build_ranker_dataset(args.log_dir, str(out_dir / "ranker.jsonl"))
    critic_rows = build_critic_dataset(args.log_dir, str(out_dir / "critic.jsonl"))
    mech_rows = build_mechanic_dataset(args.log_dir, str(out_dir / "mechanic.jsonl"))

    print(
        json.dumps(
            {
                "ranker_rows": ranker_rows,
                "critic_rows": critic_rows,
                "mechanic_rows": mech_rows,
                "out_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
