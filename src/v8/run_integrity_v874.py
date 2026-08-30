from __future__ import annotations

"""v8.74 research-run integrity repairs.

This late layer repairs three production-observability/control gaps without changing
learning thresholds or cognitive mechanisms:

* publish M2 formation telemetry from the estimator that actually runs discover();
* enforce the paper traceability dependency graph after developmental-order gates;
* propagate verified current-run WINs from worker processes into adaptive allocation.
"""

import hashlib
from dataclasses import replace
from pathlib import Path


_INSTALLED = False
_BASE_RUNTIME_METRICS = None
_BASE_V82_EVALUATE = None
_BASE_RECORD_VERIFIED_SUCCESS = None
_BASE_GAME_STATE = None
_BASE_STATE_DICT = None


def _formation_metrics_v874(self):
    payload = dict(_BASE_RUNTIME_METRICS(self))
    existing = payload.get("formation_telemetry")
    merged = dict(existing) if isinstance(existing, dict) else {}

    peers = getattr(self, "peers", None)
    if peers is not None:
        promotion = getattr(peers, "promotion", None)
        production_m2 = getattr(promotion, "generative_compression", None)
        fallback_m2 = getattr(peers, "compression", None)
        for estimator in (production_m2, fallback_m2):
            telemetry = getattr(estimator, "_v870_formation_telemetry", None)
            if isinstance(telemetry, dict) and telemetry:
                merged.update(telemetry)
                break

        roles = getattr(peers, "roles", None)
        telemetry = getattr(roles, "_v870_formation_telemetry", None)
        if isinstance(telemetry, dict) and telemetry:
            merged.update(telemetry)

    payload["formation_telemetry"] = merged
    return payload


def _dependency_requirement(hypothesis_id: str, dependency: str) -> str:
    from v8.evaluation import CONTRACTS

    for contract in CONTRACTS:
        if contract.hypothesis_id != str(hypothesis_id):
            continue
        if str(dependency) in set(contract.dependencies):
            return str(contract.dependency_min_status)
        break
    return "VALID"


def enforce_traceability_dependencies_v874(decisions):
    """Apply TRACEABILITY dependencies independent of hypothesis declaration order."""

    from v8.evaluation import ScientificHypothesisEvaluator
    from v8.scientific_traceability import TRACEABILITY

    rows = list(decisions)
    trace = {row.hypothesis_id: row for row in TRACEABILITY}

    # Statuses only stay fixed or become less conclusive, so this converges in at
    # most the number of hypotheses even for forward dependencies such as H12->H13.
    for _ in range(max(1, len(rows))):
        status = {row.hypothesis_id: row.final_decision for row in rows}
        changed = False
        updated = []
        for row in rows:
            record = trace.get(row.hypothesis_id)
            dependencies = () if record is None else record.ordering_dependencies
            blocked = []
            for dependency in dependencies:
                required = _dependency_requirement(row.hypothesis_id, dependency)
                actual = status.get(dependency)
                if not ScientificHypothesisEvaluator._dependency_satisfied(
                    actual, required
                ):
                    blocked.append((str(dependency), str(actual or "MISSING")))

            if not blocked:
                updated.append(row)
                continue

            final = row.final_decision
            if final == "VALID":
                final = (
                    "PARTIALLY_VALID"
                    if int(row.evidence_count) > 0
                    else "INSUFFICIENT_EVIDENCE"
                )

            message = "blocked traceability dependencies: " + ",".join(
                f"{dependency}={actual}" for dependency, actual in blocked
            )
            blocker_parts = [
                part
                for part in str(row.blocker).split("; ")
                if part and not part.startswith("blocked traceability dependencies:")
            ]
            blocker_parts.append(message)
            replacement = replace(
                row,
                dependency_gate="BLOCKED",
                final_decision=final,
                blocker="; ".join(blocker_parts),
            )
            changed = changed or replacement != row
            updated.append(replacement)

        rows = updated
        if not changed:
            break
    return tuple(rows)


def _evaluate_v874(self, evidence):
    return enforce_traceability_dependencies_v874(_BASE_V82_EVALUATE(self, evidence))


def _win_marker_path(game_id: str, root: str | Path | None = None) -> Path | None:
    from v8 import verified_success_metrics_v866 as verified

    success_root = verified._root_path(root) or verified._configured_success_root()
    if success_root is None:
        return None
    digest = hashlib.blake2b(
        str(game_id).encode("utf-8"),
        digest_size=16,
        person=b"v8.74-win",
    ).hexdigest()
    return success_root / "wins" / f"{digest}.json"


def _write_win_marker_v874(game_id: str, root: str | Path | None = None) -> None:
    from v8 import verified_success_metrics_v866 as verified

    marker = _win_marker_path(str(game_id), root)
    if marker is None:
        return
    verified._atomic_json(
        marker,
        {
            "schema_version": 1,
            "game_id": str(game_id),
            "terminal_state": "WIN",
        },
    )


def _record_verified_success_v874(
    *,
    game_id: str,
    seed: int,
    terminal_state: str,
    levels_completed: int,
    actions,
    capture_step: int | None,
    trajectory_id: str | None = None,
    root: str | Path | None = None,
) -> bool:
    persisted = bool(
        _BASE_RECORD_VERIFIED_SUCCESS(
            game_id=str(game_id),
            seed=int(seed),
            terminal_state=str(terminal_state),
            levels_completed=int(levels_completed),
            actions=actions,
            capture_step=capture_step,
            trajectory_id=trajectory_id,
            root=root,
        )
    )
    if persisted and str(terminal_state).upper() == "WIN":
        _write_win_marker_v874(str(game_id), root)
    return persisted


def _has_verified_win_v874(game_id: str) -> bool:
    marker = _win_marker_path(str(game_id))
    return bool(marker is not None and marker.is_file())


def _sync_verified_win_v874(self, game_id: str) -> bool:
    game = str(game_id)
    with self._lock:
        if bool(self._game_won.get(game, False)):
            return True

    if not _has_verified_win_v874(game):
        return False

    with self._lock:
        self.register_games((game,))
        self._game_won[game] = True
    return True


def _game_state_v874(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819

    won = _sync_verified_win_v874(self, str(game_id))
    state = _BASE_GAME_STATE(self, str(game_id))
    if won and state == v819.GameLearningState.UNSOLVED:
        # A worker-verified WIN is game-level competence even before an optimizer
        # candidate/frontier row exists.  Keep optimizing rather than rediscovering.
        return v819.GameLearningState.SOLVED_OPTIMIZING
    return state


def _state_dict_v874(self):
    with self._lock:
        games = tuple(self._games)
    for game in games:
        _sync_verified_win_v874(self, game)

    payload = dict(_BASE_STATE_DICT(self))
    won = payload.get("game_won")
    weights = payload.get("sampling_weight")
    states = payload.get("game_level_states")
    if not isinstance(won, dict) or not isinstance(weights, dict):
        return payload

    games_with_records = {
        str(row.get("game_id", ""))
        for row in states
        if isinstance(row, dict)
    } if isinstance(states, list) else set()

    with self._lock:
        for game, value in won.items():
            game = str(game)
            if not bool(value) or game in games_with_records:
                continue
            signals = self._signals.get(game)
            multiplier = 1.0 if signals is None else float(signals.multiplier)
            weights[game] = max(
                1e-9,
                float(self.config.optimizing_weight) * multiplier,
            )
    return payload


def install_run_integrity_v874() -> None:
    global _INSTALLED
    global _BASE_RUNTIME_METRICS, _BASE_V82_EVALUATE
    global _BASE_RECORD_VERIFIED_SUCCESS, _BASE_GAME_STATE, _BASE_STATE_DICT
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import evaluation_v82
    from v8 import runtime_v82
    from v8 import verified_success_metrics_v866 as verified

    _BASE_RUNTIME_METRICS = runtime_v82.V82ContinuousMemoryRuntime.metrics
    _BASE_V82_EVALUATE = evaluation_v82.V82ScientificHypothesisEvaluator.evaluate
    _BASE_RECORD_VERIFIED_SUCCESS = verified.record_verified_success_v866
    _BASE_GAME_STATE = v819.AdaptiveLearningCoordinator.game_state
    _BASE_STATE_DICT = v819.AdaptiveLearningCoordinator.state_dict

    runtime_v82.V82ContinuousMemoryRuntime.metrics = _formation_metrics_v874
    evaluation_v82.V82ScientificHypothesisEvaluator.evaluate = _evaluate_v874
    verified.record_verified_success_v866 = _record_verified_success_v874
    v819.AdaptiveLearningCoordinator.game_state = _game_state_v874
    v819.AdaptiveLearningCoordinator.state_dict = _state_dict_v874
    _INSTALLED = True
