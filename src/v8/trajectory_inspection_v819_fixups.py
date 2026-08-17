from __future__ import annotations

import json
from pathlib import Path


_INSTALLED = False


def _best_visible_solution(root: str | Path, game_id: str):
    from v8 import trajectory_inspection_v819 as inspection

    game = str(game_id)
    optimizer_root = Path(root) / "trajectory_optimizer"
    best = inspection._load_best_successful(
        optimizer_root / "best_successful.json"
    ).get(game)

    inbox = optimizer_root / "solutions_inbox"
    try:
        paths = sorted(inbox.glob("*.json"))
    except OSError:
        paths = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        record = inspection._validated_solution_record(raw)
        if record is None or str(record.get("game_id", "")) != game:
            continue
        if inspection._is_better_solution(record, best):
            best = record
    return best


def _show_best_trajectory_v819(root: str | Path, game_id: str) -> int:
    from v8 import trajectory_inspection_v819 as inspection
    from v8.action_targeting_v810 import native_action_id

    game = str(game_id)
    record = _best_visible_solution(root, game)
    if record is None:
        print(f"game={game} no successful trajectory found", flush=True)
        return 1

    reliability = float(record.get("reliability", 0.0))
    print(
        f"game={game} cost={int(record['total_cost'])} "
        f"source={record['source']} reliability={reliability:.3f}",
        flush=True,
    )
    levels = inspection._normalize_levels(record.get("levels")) or ()
    for index, actions in enumerate(levels):
        formatted = ",".join(f"A{int(native_action_id(action))}" for action in actions)
        print(f"L{index}: {formatted}", flush=True)
    return 0


def _ingest_inbox_v819_prioritized(service) -> None:
    from v8 import trajectory_inspection_v819 as inspection

    # Complete WIN persistence is tiny and must not wait behind trajectory
    # optimization inbox work.
    inspection._ingest_solution_inbox(service)
    inspection._BASE_INGEST_INBOX_V818(service)


def install_trajectory_inspection_v819_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    inspection.show_best_trajectory = _show_best_trajectory_v819
    inspection._ingest_inbox_v819 = _ingest_inbox_v819_prioritized
    optimizer.TrajectoryOptimizationService._ingest_inbox = _ingest_inbox_v819_prioritized
    v818._ingest_inbox_v818 = _ingest_inbox_v819_prioritized
    _INSTALLED = True
