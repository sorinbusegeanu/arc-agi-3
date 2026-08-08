from __future__ import annotations

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
