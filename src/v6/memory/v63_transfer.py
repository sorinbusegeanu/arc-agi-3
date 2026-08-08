from __future__ import annotations

import sys
import time
from typing import Any

V63_ABSTRACTION_VERSION = "v63_relational_abstraction_v1"
QUALIFIED_EVIDENCE_PREFIX = "v63_unseen_transfer:"
UNQUALIFIED_EVIDENCE_PREFIX = "v63_unqualified_transfer:"

_PATCHED = False
_ORIGINAL_RECORD_PREDICTION_OUTCOME: Any = None
_ORIGINAL_CONCEPT_TRANSFER_COUNTS: Any = None


def _table_columns(connection: Any, table: str) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if exists is None:
        return set()
    return {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }


def install_v63_transfer_policy() -> None:
    global _PATCHED
    global _ORIGINAL_RECORD_PREDICTION_OUTCOME
    global _ORIGINAL_CONCEPT_TRANSFER_COUNTS
    if _PATCHED:
        _patch_v6_system_if_loaded()
        return

    from v6.memory.v621_runtime import (
        V621AbstractionEngine,
        V621MemoryController,
        V621PromotionEngine,
    )

    _ORIGINAL_RECORD_PREDICTION_OUTCOME = (
        V621MemoryController.record_prediction_outcome
    )
    _ORIGINAL_CONCEPT_TRANSFER_COUNTS = (
        V621PromotionEngine._concept_transfer_counts
    )
    V621MemoryController.record_prediction_outcome = (
        _record_prediction_outcome_v63
    )
    V621AbstractionEngine._direct_concept_transfer_for_roles = (
        _direct_concept_transfer_for_roles_v63
    )
    V621PromotionEngine._concept_transfer_counts = (
        _concept_transfer_counts_v63
    )
    _PATCHED = True
    _patch_v6_system_if_loaded()


def _patch_v6_system_if_loaded() -> None:
    module = sys.modules.get("v6.main")
    system_type = None if module is None else getattr(module, "V6System", None)
    if system_type is None:
        return
    system_type._initial_retention_score = _initial_retention_score_v63
    system_type._apply_trajectory_efficiency_bonus = (
        _apply_trajectory_efficiency_bonus_v63
    )


def _initial_retention_score_v63(
    self: Any,
    *,
    isf_total: float,
    replay_priority: float,
    explanatory_reach: float,
    transfer_potential: float,
    recurrence: float,
    future_option_impact: float,
    efficiency_bonus: float = 0.0,
) -> float:
    from v6.memory.v63_policy import unified_memory_fitness

    # Replay is an output of fitness, not an input back into retention.
    # Future-option impact is already represented through ISF and is not
    # counted a second time here.
    del replay_priority, future_option_impact
    score, _components = unified_memory_fitness(
        isf_score=float(isf_total),
        explanatory_reach=(
            float(explanatory_reach)
            if float(explanatory_reach) > 0.0
            else None
        ),
        transfer_prior=(
            float(transfer_potential)
            if float(transfer_potential) > 0.0
            else None
        ),
        transfer_empirical=None,
        recurrence_score=(
            float(recurrence) if float(recurrence) > 0.0 else None
        ),
        efficiency_score=(
            float(efficiency_bonus)
            if float(efficiency_bonus) > 0.0
            else None
        ),
    )
    return score


def _apply_trajectory_efficiency_bonus_v63(
    self: Any,
    *,
    trajectory_record: Any,
    interaction_ids: list[int],
) -> None:
    from v6.memory.substrate import MemoryScore, trajectory_node_id
    from v6.memory.v63_policy import SCORE_POLICY_VERSION, unified_memory_fitness
    from v6.memory_lifecycle import ReplayCandidate

    efficiency_active = bool(trajectory_record.efficiency_active)
    memory_bonus = float(
        trajectory_record.efficiency_memory_bonus
        if efficiency_active
        else 0.0
    )
    replay_bonus = float(
        trajectory_record.efficiency_replay_bonus
        if efficiency_active
        else 0.0
    )
    retention_bonus = float(
        trajectory_record.efficiency_retention_bonus
        if efficiency_active
        else 0.0
    )
    promotion_bonus = float(
        trajectory_record.efficiency_promotion_bonus
        if efficiency_active
        else 0.0
    )
    useful_outcome = bool(
        str(trajectory_record.outcome_class) in {"WIN", "LEVEL_COMPLETE"}
        or float(trajectory_record.future_option_gain or 0.0) > 0.0
    )
    efficiency_score = (
        None
        if not efficiency_active
        or not useful_outcome
        or trajectory_record.efficiency_score is None
        else float(trajectory_record.efficiency_score)
    )

    for interaction_id in interaction_ids:
        row = self.connection.execute(
            """
            SELECT memory_fitness_base,
                   memory_replay_priority_base,
                   memory_replay_priority,
                   memory_status
            FROM interactions
            WHERE id = ?
            """,
            (int(interaction_id),),
        ).fetchone()
        base_fitness = float(
            (row[0] if row and row[0] is not None else 0.0) or 0.0
        )
        base_replay = float(
            (row[1] if row and row[1] is not None else 0.0) or 0.0
        )
        current_replay = float(
            (row[2] if row and row[2] is not None else base_replay)
            or base_replay
        )

        memory_fitness, components = unified_memory_fitness(
            isf_score=base_fitness,
            explanatory_reach=None,
            transfer_prior=None,
            transfer_empirical=None,
            recurrence_score=None,
            efficiency_score=efficiency_score,
        )
        if efficiency_score is None:
            memory_fitness = base_fitness
        replay_priority = max(current_replay, base_replay, memory_fitness)
        retention_score_base = base_fitness
        retention_score = memory_fitness

        self.connection.execute(
            """
            UPDATE interactions
            SET
                trajectory_efficiency_active = ?,
                trajectory_outcome_class = ?,
                comparable_outcome_group_id = ?,
                trajectory_efficiency_score = ?,
                efficiency_memory_bonus = ?,
                efficiency_replay_bonus = ?,
                efficiency_retention_bonus = ?,
                efficiency_promotion_bonus = ?,
                memory_fitness = ?,
                memory_replay_priority = MAX(COALESCE(memory_replay_priority, 0.0), ?),
                retention_score_base = ?,
                retention_score = ?
            WHERE id = ?
            """,
            (
                int(efficiency_active),
                str(trajectory_record.outcome_class),
                str(trajectory_record.comparable_outcome_group_id),
                None
                if trajectory_record.efficiency_score is None
                else float(trajectory_record.efficiency_score),
                memory_bonus,
                replay_bonus,
                retention_bonus,
                promotion_bonus,
                float(memory_fitness),
                float(replay_priority),
                float(retention_score_base),
                float(retention_score),
                int(interaction_id),
            ),
        )
        self.connection.execute(
            """
            UPDATE prediction_results
            SET
                trajectory_efficiency_active = ?,
                trajectory_outcome_class = ?,
                comparable_outcome_group_id = ?,
                trajectory_efficiency_score = ?,
                efficiency_memory_bonus = ?,
                efficiency_replay_bonus = ?,
                efficiency_retention_bonus = ?,
                efficiency_promotion_bonus = ?,
                memory_fitness = ?,
                memory_replay_priority = MAX(COALESCE(memory_replay_priority, 0.0), ?),
                retention_score_base = ?,
                retention_score = ?
            WHERE interaction_id = ?
            """,
            (
                int(efficiency_active),
                str(trajectory_record.outcome_class),
                str(trajectory_record.comparable_outcome_group_id),
                None
                if trajectory_record.efficiency_score is None
                else float(trajectory_record.efficiency_score),
                memory_bonus,
                replay_bonus,
                retention_bonus,
                promotion_bonus,
                float(memory_fitness),
                float(replay_priority),
                float(retention_score_base),
                float(retention_score),
                int(interaction_id),
            ),
        )

        candidate = self.memory_controller.replay_candidates.get(
            str(interaction_id)
        )
        if candidate is not None:
            self.memory_controller.replay_candidates[str(interaction_id)] = (
                ReplayCandidate(
                    interaction_id=candidate.interaction_id,
                    replay_priority=float(replay_priority),
                    reason=str(candidate.reason),
                    family_id=candidate.family_id,
                    context_signature=candidate.context_signature,
                    status=candidate.status,
                )
            )
            self._sync_post_factum_replay_fields(int(interaction_id))

        node_id = self._interaction_memory_node_id(interaction_id)
        self.memory.upsert_score(
            MemoryScore(
                node_id=node_id,
                replay_priority=float(replay_priority),
                memory_state="active" if efficiency_active else None,
                retention_score=float(retention_score),
                forgetting_score=float(max(0.0, 1.0 - retention_score)),
            ),
            step=int(interaction_id),
        )
        self.memory.connection.execute(
            """
            UPDATE memory_scores
            SET memory_fitness=?,
                efficiency_score=?,
                score_components_json=?,
                score_policy_version=?
            WHERE node_id=?
            """,
            (
                float(memory_fitness),
                efficiency_score,
                __import__("json").dumps(components, sort_keys=True),
                SCORE_POLICY_VERSION,
                node_id,
            ),
        )

    trajectory_node = trajectory_node_id(self.episode_id)
    attrs = {
        "trajectory_efficiency_active": bool(efficiency_active),
        "trajectory_efficiency_score": trajectory_record.efficiency_score,
        "efficiency_memory_bonus": memory_bonus,
        "efficiency_replay_bonus": replay_bonus,
        "efficiency_retention_bonus": retention_bonus,
        "efficiency_promotion_bonus": promotion_bonus,
        "memory_fitness_policy": "v63_unified_memory_fitness_v1",
    }
    self.memory.update_node_support_and_attrs(
        trajectory_node,
        attrs,
        support_increment=0,
    )
    self.memory.connection.commit()


def _record_prediction_outcome_v63(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    started = time.time() - 0.001
    _ORIGINAL_RECORD_PREDICTION_OUTCOME(self, *args, **kwargs)

    game = kwargs.get("game")
    global_step = kwargs.get("global_step")
    if game in (None, ""):
        target_game = None
    else:
        target_game = str(game)

    rows = self.memory.connection.execute(
        """
        SELECT attempt_id, concept_id, evidence_source
        FROM concept_transfer_attempts_v621
        WHERE created_at >= ?
          AND global_step IS ?
          AND game IS ?
        """,
        (started, global_step, target_game),
    ).fetchall()
    if not rows:
        return

    for row in rows:
        attempt_id = str(row[0])
        concept_id = str(row[1])
        original_source = str(row[2] or "runtime_concept_test")
        source_games = _concept_source_games(self.memory, concept_id)
        qualifies = bool(
            target_game
            and source_games
            and target_game not in source_games
        )
        prefix = (
            QUALIFIED_EVIDENCE_PREFIX
            if qualifies
            else UNQUALIFIED_EVIDENCE_PREFIX
        )
        if original_source.startswith(
            (QUALIFIED_EVIDENCE_PREFIX, UNQUALIFIED_EVIDENCE_PREFIX)
        ):
            evidence_source = original_source
        else:
            evidence_source = prefix + original_source
        self.memory.connection.execute(
            """
            UPDATE concept_transfer_attempts_v621
            SET evidence_source=?
            WHERE attempt_id=?
            """,
            (evidence_source, attempt_id),
        )
    self.memory.connection.commit()


def _concept_source_games(memory: Any, concept_id: str) -> set[str]:
    node = memory.get_node(str(concept_id))
    if node is None:
        return set()
    attrs = dict(node.get("attrs") or {})
    explicit = {
        str(value)
        for value in attrs.get("source_games", []) or []
        if value not in (None, "")
    }
    if explicit:
        return explicit

    columns = _table_columns(memory.connection, "role_transfer_attempts")
    if "role_signature" not in columns:
        return set()
    source_game_column = (
        "source_game_key"
        if "source_game_key" in columns
        else "source_game"
        if "source_game" in columns
        else None
    )
    if source_game_column is None:
        return set()

    output: set[str] = set()
    for role_id in attrs.get("source_roles", []) or []:
        role = memory.get_node(str(role_id))
        if role is None:
            continue
        role_attrs = dict(role.get("attrs") or {})
        signature = str(
            role_attrs.get(
                "role_signature",
                role.get("canonical_key", ""),
            )
            or ""
        )
        if not signature:
            continue
        for row in memory.connection.execute(
            f"""
            SELECT DISTINCT {source_game_column}
            FROM role_transfer_attempts
            WHERE role_signature=?
              AND {source_game_column} IS NOT NULL
              AND {source_game_column} != ''
            """,
            (signature,),
        ).fetchall():
            output.add(str(row[0]))
    return output


def _direct_concept_transfer_for_roles_v63(
    self: Any,
    role_ids: set[str],
) -> dict[str, int]:
    if not role_ids:
        return {"tests": 0, "successes": 0}
    concepts = [
        node
        for node in self.memory.query_nodes(
            memory_level="M4",
            node_type="ConceptMemory",
        )
        if set(
            str(value)
            for value in node.get("attrs", {}).get(
                "source_roles",
                [],
            )
        )
        == role_ids
    ]
    if not concepts:
        return {"tests": 0, "successes": 0}
    concept_ids = [str(node["node_id"]) for node in concepts]
    placeholders = ",".join("?" for _ in concept_ids)
    row = self.memory.connection.execute(
        f"""
        SELECT COUNT(*),
               COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0)
        FROM concept_transfer_attempts_v621
        WHERE concept_id IN ({placeholders})
          AND evidence_source LIKE ?
        """,
        (*concept_ids, QUALIFIED_EVIDENCE_PREFIX + "%"),
    ).fetchone()
    return {
        "tests": int(row[0] or 0),
        "successes": int(row[1] or 0),
    }


def _concept_transfer_counts_v63(
    self: Any,
    concept_id: str,
) -> tuple[int, int]:
    node = self.memory.get_node(str(concept_id))
    attrs = {} if node is None else dict(node.get("attrs") or {})
    if attrs.get("concept_version") != V63_ABSTRACTION_VERSION:
        return _ORIGINAL_CONCEPT_TRANSFER_COUNTS(self, concept_id)
    row = self.memory.connection.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0)
        FROM concept_transfer_attempts_v621
        WHERE concept_id=?
          AND evidence_source LIKE ?
        """,
        (str(concept_id), QUALIFIED_EVIDENCE_PREFIX + "%"),
    ).fetchone()
    return int(row[0] or 0), int(row[1] or 0)
