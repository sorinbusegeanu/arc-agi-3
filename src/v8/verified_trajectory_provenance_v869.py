from __future__ import annotations

"""v8.69 preserve generic episode provenance and research-role integrity."""

from copy import deepcopy


_INSTALLED = False
_BASE_VERIFIED_PROXY_STEP = None
_BASE_VERIFIED_EXPORT_RECORD = None
_BASE_FORMAT_BEST_TRAJECTORY_LINES = None
_BASE_RESEARCH_BUILD_PACKET = None
_BASE_RESEARCH_DERIVE_CHAIN = None


def _gym_episode_seed(self) -> int:
    return int(self.seed) + max(0, int(self._episode) - 1)


def _chess_episode_seed(self) -> int:
    return int(self.seed) + max(0, int(self._episode) - 1)


def _sudoku_episode_seed(self) -> int:
    return int(self.seed) + max(0, int(self._episode) - 1) * 104729


def _active_episode_seed(inner, fallback: int) -> int:
    getter = getattr(inner, "cognitive_episode_seed", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError):
            pass
    return int(fallback)


def _verified_proxy_step_v869(self, action):
    from v8 import verified_success_metrics_v866 as verified

    self._steps += 1
    self._actions.append(int(action))
    result = self._inner.step(action)
    boundary = self._inner.cognitive_boundary_event()
    if not bool(boundary.continuation) and int(boundary.primary_valence) > 0:
        verified.record_verified_success_v866(
            game_id=self._game_id,
            seed=_active_episode_seed(self._inner, self._seed),
            terminal_state="WIN",
            levels_completed=1,
            actions=tuple(self._actions),
            capture_step=self._steps,
        )
    return result


def _verified_export_record_v869(row: dict[str, object]) -> dict[str, object]:
    record = dict(_BASE_VERIFIED_EXPORT_RECORD(row))
    try:
        record["seed"] = int(row["seed"])
    except (KeyError, TypeError, ValueError):
        pass
    return record


def _format_best_trajectory_lines_v869(
    game_id: str,
    record: dict[str, object],
) -> tuple[str, ...]:
    lines = list(_BASE_FORMAT_BEST_TRAJECTORY_LINES(game_id, record))
    if lines and "seed" in record:
        try:
            lines[0] = f"{lines[0]} seed={int(record['seed'])}"
        except (TypeError, ValueError):
            pass
    return tuple(lines)


def _research_derive_chain_v869(summary):
    from v8.research.contracts import ChainStatus
    from v8.research.models import ChainEdgeEvidence

    evidence = dict(_BASE_RESEARCH_DERIVE_CHAIN(summary))
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    if not isinstance(metrics, dict) or "m3_role_candidate_count" not in metrics:
        return evidence
    try:
        role_candidates = max(0, int(metrics.get("m3_role_candidate_count", 0) or 0))
        m3_total = max(0, int((metrics.get("level_counts", {}) or {}).get("3", 0) or 0))
    except (TypeError, ValueError):
        return evidence
    if role_candidates > 0:
        evidence["M3_ROLE_FORMATION"] = ChainEdgeEvidence(
            "M3_ROLE_FORMATION",
            ChainStatus.PASS,
            role_candidates,
            ("evidence.evidence_kind_counts.role_candidate",),
        )
    elif m3_total > 0:
        evidence["M3_ROLE_FORMATION"] = ChainEdgeEvidence(
            "M3_ROLE_FORMATION",
            ChainStatus.INSUFFICIENT_EVIDENCE,
            0,
            ("metrics.level_counts.M3", "evidence.evidence_kind_counts.role_candidate"),
            "M3 memories exist, but no role_candidate evidence exists; M3 level count includes carriers and cannot establish role formation",
        )
    return evidence


def _research_build_packet_v869(
    summary,
    *,
    revision,
    argv,
    h_report,
    reporting_cut,
    evidence_digest,
    log_tail,
):
    enriched = deepcopy(dict(summary))
    metrics = dict(enriched.get("metrics", {}) or {})
    kinds = evidence_digest.get("evidence_kind_counts", {}) if isinstance(evidence_digest, dict) else {}
    if isinstance(kinds, dict):
        metrics["m3_role_candidate_count"] = max(0, int(kinds.get("role_candidate", 0) or 0))
        enriched["metrics"] = metrics
    return _BASE_RESEARCH_BUILD_PACKET(
        enriched,
        revision=revision,
        argv=argv,
        h_report=h_report,
        reporting_cut=reporting_cut,
        evidence_digest=evidence_digest,
        log_tail=log_tail,
    )


def install_verified_trajectory_provenance_v869() -> None:
    global _INSTALLED
    global _BASE_VERIFIED_PROXY_STEP, _BASE_VERIFIED_EXPORT_RECORD
    global _BASE_FORMAT_BEST_TRAJECTORY_LINES
    global _BASE_RESEARCH_BUILD_PACKET, _BASE_RESEARCH_DERIVE_CHAIN
    if _INSTALLED:
        return

    from v8.environments.chess_env import ChessAdapter
    from v8.environments.gym_adapter import GymDiscreteAdapter
    from v8.environments.sudoku_env import SudokuAdapter
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import verified_success_metrics_v866 as verified
    from v8 import verified_trajectory_export_v868 as export
    from v8.research import researcher_packet

    GymDiscreteAdapter.cognitive_episode_seed = _gym_episode_seed
    ChessAdapter.cognitive_episode_seed = _chess_episode_seed
    SudokuAdapter.cognitive_episode_seed = _sudoku_episode_seed

    _BASE_VERIFIED_PROXY_STEP = verified._VerifiedAdapterProxy.step
    verified._VerifiedAdapterProxy.step = _verified_proxy_step_v869

    _BASE_VERIFIED_EXPORT_RECORD = export._verified_export_record
    export._verified_export_record = _verified_export_record_v869

    _BASE_FORMAT_BEST_TRAJECTORY_LINES = inspection._format_best_trajectory_lines
    inspection._format_best_trajectory_lines = _format_best_trajectory_lines_v869

    _BASE_RESEARCH_DERIVE_CHAIN = researcher_packet.derive_chain_evidence
    researcher_packet.derive_chain_evidence = _research_derive_chain_v869
    _BASE_RESEARCH_BUILD_PACKET = researcher_packet.build_packet
    researcher_packet.build_packet = _research_build_packet_v869
    _INSTALLED = True
