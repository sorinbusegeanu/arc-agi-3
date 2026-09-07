from __future__ import annotations

from v8.model import MemoryLevel


_INSTALLED = False


def _trial_counts(transfer) -> dict[object, int]:
    trials = getattr(transfer, "_trials", {})
    if not isinstance(trials, dict):
        return {}
    return {uid: len(rows) for uid, rows in trials.items()}


def _persist_new_passed_trials(runtime, before_counts: dict[object, int]) -> int:
    transfer = runtime.peers.transfer
    trials = getattr(transfer, "_trials", {})
    if not isinstance(trials, dict):
        return 0

    read_view = runtime.read_view
    refresh = getattr(read_view, "_refresh_strategy_cache", None)
    if callable(refresh):
        refresh()
    by_uid = dict(getattr(read_view, "_node_by_uid", {}) or {})
    if not by_uid:
        node_records = getattr(read_view, "node_records", None)
        if callable(node_records):
            by_uid = {row.uid: row for row in node_records()}

    written = 0
    for uid, rows in trials.items():
        start = max(0, int(before_counts.get(uid, 0)))
        row = by_uid.get(uid)
        if row is None:
            continue
        for trial in tuple(rows)[start:]:
            if not bool(getattr(trial, "passed", False)):
                continue
            effect = float(getattr(trial, "effect", 0.0))
            if effect <= float(getattr(transfer, "effect_threshold", 0.0)):
                continue
            common = {
                "unique": True,
                "target_game_hash": int(trial.target_game_hash),
                "provenance_games": tuple(int(v) for v in trial.formation_games),
                "causal_intervention": str(trial.intervention),
                "effect_direction": 1,
            }
            runtime.peers._append_evidence(
                "transfer_trial_pass",
                row,
                effect,
                **common,
            )
            if int(getattr(row, "level", -1)) == int(MemoryLevel.M4):
                runtime.peers._append_evidence(
                    "concept_transfer_pass",
                    row,
                    effect,
                    **common,
                )
            written += 1
    return written


def _install_transfer_pass_evidence() -> None:
    from v8 import experiments as experiments_module
    from v8 import learning_fixes_v088 as learning

    current_run = learning._run_automatic_transfer_experiments_v088

    def run_automatic_transfer_experiments(runtime, **kwargs):
        before = _trial_counts(runtime.peers.transfer)
        summary = current_run(runtime, **kwargs)
        _persist_new_passed_trials(runtime, before)
        return summary

    learning._run_automatic_transfer_experiments_v088 = run_automatic_transfer_experiments
    experiments_module.run_automatic_transfer_experiments = run_automatic_transfer_experiments


def install_transfer_pass_evidence_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_transfer_pass_evidence()
    _INSTALLED = True
