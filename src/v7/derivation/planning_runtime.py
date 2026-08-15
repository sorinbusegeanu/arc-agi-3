from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Iterable

from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.evidence_store import EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleIndexMutation
from v7.memory.models import EdgeMutation, NodeMutation, ScoreMutation
from v7.memory.planning import (
    MAX_STRATEGY_STEPS,
    REL_PLAN_FAILURE,
    REL_PLAN_OPTION_LOSS,
    REL_PLAN_STALL,
    REL_PLAN_SUCCESS,
    REL_PLAN_TRANSITION,
    REL_STRATEGY_STEP_BASE,
    TYPE_EXECUTABLE_PROCEDURE,
    planning_context,
)
from v7.memory.status import memory_is_active
from v7.memory.writer import CanonicalMemoryWriter

_MASK63 = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class PlanningDerivationStats:
    transition_edges: int = 0
    success_markers: int = 0
    failure_markers: int = 0
    stall_markers: int = 0
    option_loss_markers: int = 0
    procedures: int = 0
    strategy_step_edges: int = 0


class Phase1PlanningBuilder:
    """Derive persistent planning edges and executable M6 procedures."""

    def __init__(
        self,
        writer: CanonicalMemoryWriter,
        evidence_store: EvidenceStore,
    ) -> None:
        self.writer = writer
        self.evidence_store = evidence_store

    def derive(self) -> PlanningDerivationStats:
        episodes = self._load(EvidenceType.EPISODE)
        trajectories = self._load(EvidenceType.TRAJECTORY)
        edge_counts = self._planning_edge_counts(episodes)
        applied = self._sync_edge_counts(edge_counts)
        procedure_count, strategy_edges = self._derive_procedures(
            episodes=episodes,
            trajectories=trajectories,
        )
        return PlanningDerivationStats(
            transition_edges=sum(
                1
                for (_source, relation, _target) in applied
                if relation == REL_PLAN_TRANSITION
            ),
            success_markers=sum(
                1
                for (_source, relation, _target) in applied
                if relation == REL_PLAN_SUCCESS
            ),
            failure_markers=sum(
                1
                for (_source, relation, _target) in applied
                if relation == REL_PLAN_FAILURE
            ),
            stall_markers=sum(
                1
                for (_source, relation, _target) in applied
                if relation == REL_PLAN_STALL
            ),
            option_loss_markers=sum(
                1
                for (_source, relation, _target) in applied
                if relation == REL_PLAN_OPTION_LOSS
            ),
            procedures=procedure_count,
            strategy_step_edges=strategy_edges,
        )

    def _planning_edge_counts(
        self,
        episodes: list[dict[str, Any]],
    ) -> Counter[tuple[MemoryId, int, MemoryId]]:
        counts: Counter[tuple[MemoryId, int, MemoryId]] = Counter()
        by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            game = row.get("source_game")
            memory_id = row.get("memory_id")
            if game is None or memory_id is None:
                continue
            by_game[str(game)].append(row)

        for game in sorted(by_game):
            rows = sorted(
                by_game[game],
                key=lambda item: int(item.get("source_global_step") or -1),
            )
            for index, row in enumerate(rows):
                memory_id = MemoryId(int(row["memory_id"]))
                if not memory_is_active(
                    getattr(self.writer, "_nodes").get(memory_id)
                ):
                    continue
                polarity = int(row.get("terminal_polarity") or 0)
                if polarity > 0:
                    counts[(memory_id, REL_PLAN_SUCCESS, memory_id)] += 1
                elif polarity < 0:
                    counts[(memory_id, REL_PLAN_FAILURE, memory_id)] += 1
                if int(row.get("changed_cells") or 0) <= 0:
                    counts[(memory_id, REL_PLAN_STALL, memory_id)] += 1
                if float(row.get("future_option_delta") or 0.0) < 0.0:
                    counts[(memory_id, REL_PLAN_OPTION_LOSS, memory_id)] += 1

                if polarity != 0 or index + 1 >= len(rows):
                    continue
                following = rows[index + 1]
                following_id = following.get("memory_id")
                if following_id is None:
                    continue
                target_id = MemoryId(int(following_id))
                if not memory_is_active(
                    getattr(self.writer, "_nodes").get(target_id)
                ):
                    continue
                if not self._continuous_transition(row, following):
                    continue
                counts[(memory_id, REL_PLAN_TRANSITION, target_id)] += 1
        return counts

    @staticmethod
    def _continuous_transition(
        current: dict[str, Any],
        following: dict[str, Any],
    ) -> bool:
        current_segment = str(current.get("trajectory_segment_id") or "")
        following_segment = str(following.get("trajectory_segment_id") or "")
        if current_segment or following_segment:
            if not current_segment or current_segment != following_segment:
                return False
        if bool(following.get("reset_boundary_before_step")):
            return False
        after_context = planning_context(
            current.get("next_context_signatures", ()) or (),
            fallback=-1,
        )
        next_context = planning_context(
            following.get("context_signatures", ()) or (),
            fallback=int(following.get("context_signature") or -2),
        )
        if after_context < 0 or next_context < 0 or after_context != next_context:
            return False
        current_step = current.get("source_global_step")
        next_step = following.get("source_global_step")
        if current_step is not None and next_step is not None:
            if int(next_step) != int(current_step) + 1:
                return False
        return True

    def _derive_procedures(
        self,
        *,
        episodes: list[dict[str, Any]],
        trajectories: list[dict[str, Any]],
    ) -> tuple[int, int]:
        episodes_by_segment: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for row in episodes:
            game = str(row.get("source_game") or "")
            if not game or row.get("memory_id") is None:
                continue
            segment = str(row.get("trajectory_segment_id") or "")
            episodes_by_segment[(game, segment)].append(row)
        for key in episodes_by_segment:
            episodes_by_segment[key].sort(
                key=lambda item: int(item.get("source_global_step") or -1)
            )

        grouped: dict[
            int,
            list[
                tuple[
                    dict[str, Any],
                    tuple[int, ...],
                    tuple[int, ...],
                    tuple[MemoryId, ...],
                ]
            ],
        ] = defaultdict(list)
        for row in trajectories:
            if not bool(row.get("success")):
                continue
            actions = tuple(
                int(value)
                for value in (row.get("action_sequence", ()) or ())[
                    :MAX_STRATEGY_STEPS
                ]
            )
            contexts = tuple(
                int(value)
                for value in (row.get("context_sequence", ()) or ())[
                    :MAX_STRATEGY_STEPS
                ]
            )
            pair_count = min(len(actions), len(contexts), MAX_STRATEGY_STEPS)
            if pair_count <= 0:
                continue
            actions = actions[:pair_count]
            contexts = contexts[:pair_count]
            segment = self._episode_segment(
                row,
                episodes_by_segment,
                pair_count,
            )
            if len(segment) != pair_count:
                continue
            signature = _procedure_signature(actions, contexts)
            grouped[signature].append((row, actions, contexts, segment))

        created = 0
        edge_counts: Counter[tuple[MemoryId, int, MemoryId]] = Counter()
        nodes = getattr(self.writer, "_nodes")
        for signature, observations in sorted(grouped.items()):
            key = CanonicalMemoryKey(
                MemoryLevel.M6,
                TYPE_EXECUTABLE_PROCEDURE,
                (int(signature),),
            )
            strategy_id = self.writer.canonical_memory_id(key)
            support = len(observations)
            mean_efficiency = sum(
                1.0
                / max(
                    1.0,
                    float(
                        item[0].get("steps_to_success") or len(item[1])
                    ),
                )
                for item in observations
            ) / support
            mean_future = sum(
                float(item[0].get("future_option_per_action") or 0.0)
                for item in observations
            ) / support

            if strategy_id is None:
                candidate = CanonicalCandidateMutation(
                    key=key,
                    support_delta=support,
                    significance=1.0,
                    learning_value=min(1.0, mean_efficiency),
                    explanatory_potential=min(1.0, mean_efficiency),
                    future_option_delta=mean_future,
                )
                strategy_id = self.writer.apply_canonical_candidate_batch(
                    (candidate,)
                )[key]
                created += 1
            else:
                current_support = int(nodes[strategy_id].support_count)
                if support > current_support:
                    self.writer.apply_mutation_batch(
                        (
                            NodeMutation(
                                strategy_id,
                                MemoryLevel.M6,
                                TYPE_EXECUTABLE_PROCEDURE,
                                support_delta=support - current_support,
                            ),
                        )
                    )
                # Do not overwrite significance/learning/future-option here.
                # Those fields are updated from actual post-creation strategy
                # successes and failures by OnlineHierarchyBuilder.
                self.writer.apply_score_batch(
                    (
                        ScoreMutation(
                            memory_id=strategy_id,
                            explanatory_potential=min(
                                1.0,
                                mean_efficiency,
                            ),
                        ),
                    )
                )

            representative = observations[0]
            actions = representative[1]
            contexts = representative[2]
            for context, action in zip(contexts, actions, strict=True):
                self.writer.apply_role_index_batch(
                    (
                        RoleIndexMutation(
                            int(context),
                            int(action),
                            strategy_id,
                            None,
                        ),
                    )
                )

            for _row, _actions, _contexts, segment in observations:
                for position, memory_id in enumerate(segment):
                    edge_counts[
                        (
                            strategy_id,
                            REL_STRATEGY_STEP_BASE + position,
                            memory_id,
                        )
                    ] += 1

        applied = self._sync_edge_counts(edge_counts)
        strategy_edges = sum(
            1
            for (_source, relation, _target) in applied
            if REL_STRATEGY_STEP_BASE
            <= relation
            < REL_STRATEGY_STEP_BASE + MAX_STRATEGY_STEPS
        )
        return created, strategy_edges

    def _episode_segment(
        self,
        trajectory: dict[str, Any],
        episodes_by_segment: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ],
        expected_length: int,
    ) -> tuple[MemoryId, ...]:
        game = str(trajectory.get("source_game") or "")
        segment_id = str(trajectory.get("trajectory_segment_id") or "")
        end_step = trajectory.get("source_global_step")
        if not game or end_step is None:
            return ()
        rows = episodes_by_segment.get((game, segment_id), ())
        # Legacy data has no segment identity; strict step/context continuity
        # below remains the compatibility fallback.
        if not rows and segment_id:
            return ()
        if not rows:
            rows = episodes_by_segment.get((game, ""), ())
        end_index = -1
        for index, row in enumerate(rows):
            if int(row.get("source_global_step") or -1) == int(end_step):
                end_index = index
                break
        if end_index < 0:
            return ()
        start = end_index - int(expected_length) + 1
        if start < 0:
            return ()
        segment = rows[start : end_index + 1]
        if len(segment) != expected_length:
            return ()
        if any(
            int(row.get("terminal_polarity") or 0) != 0
            for row in segment[:-1]
        ):
            return ()
        if any(
            not self._continuous_transition(left, right)
            for left, right in zip(segment, segment[1:])
        ):
            return ()
        memory_ids = tuple(
            MemoryId(int(row["memory_id"])) for row in segment
        )
        nodes = getattr(self.writer, "_nodes")
        if any(not memory_is_active(nodes.get(memory_id)) for memory_id in memory_ids):
            return ()
        return memory_ids

    def _sync_edge_counts(
        self,
        desired: Counter[tuple[MemoryId, int, MemoryId]],
    ) -> set[tuple[MemoryId, int, MemoryId]]:
        edge_support = getattr(self.writer, "_edge_support")
        mutations: list[EdgeMutation] = []
        changed: set[tuple[MemoryId, int, MemoryId]] = set()
        for key, wanted in sorted(
            desired.items(),
            key=lambda item: (
                int(item[0][0]),
                item[0][1],
                int(item[0][2]),
            ),
        ):
            current = int(edge_support.get(key, 0))
            delta = int(wanted) - current
            if delta <= 0:
                continue
            mutations.append(
                EdgeMutation(
                    source_id=key[0],
                    relation_type=key[1],
                    target_id=key[2],
                    support_delta=delta,
                )
            )
            changed.add(key)
        if mutations:
            self.writer.apply_edge_batch(mutations)
        return changed

    def _load(self, evidence_type: EvidenceType) -> list[dict[str, Any]]:
        return self.evidence_store.load_evidence(int(evidence_type))


def _procedure_signature(
    actions: Iterable[int],
    contexts: Iterable[int],
) -> int:
    digest = blake2b(digest_size=8)
    digest.update(b"v7-phase1-procedure-v1")
    digest.update(
        str(tuple(int(value) for value in actions)).encode("ascii")
    )
    digest.update(
        str(tuple(int(value) for value in contexts)).encode("ascii")
    )
    return int.from_bytes(digest.digest(), "little") & _MASK63
