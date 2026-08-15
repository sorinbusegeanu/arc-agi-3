from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.read_view import MemoryReadView
from v7.memory.status import memory_is_active

# Typed relations stored in the canonical adjacency graph. They deliberately
# live outside the M1-M6 type-id range and are stable logical semantics.
REL_PLAN_TRANSITION = 70_001
REL_PLAN_SUCCESS = 70_002
REL_PLAN_FAILURE = 70_003
REL_PLAN_STALL = 70_004
REL_PLAN_OPTION_LOSS = 70_005
REL_STRATEGY_STEP_BASE = 80_000
MAX_STRATEGY_STEPS = 256
TYPE_EXECUTABLE_PROCEDURE = 601


def planning_context(signatures: Iterable[int], fallback: int = 0) -> int:
    """Return the reusable combined planning context with legacy compatibility."""
    values = tuple(int(value) for value in signatures)
    if len(values) >= 5:
        return int(values[3])
    if len(values) == 4:
        return int(values[3])
    if len(values) >= 3:
        return int(values[2])
    if values:
        return int(values[-1])
    return int(fallback)


@dataclass(frozen=True, slots=True)
class PlanningSignal:
    reachability: float = 0.0
    success_reachability: float = 0.0
    failure_risk: float = 0.0
    stall_risk: float = 0.0
    option_loss_risk: float = 0.0
    reachable_nodes: int = 0


@dataclass(frozen=True, slots=True)
class StrategyStep:
    position: int
    memory_id: MemoryId
    context_signature: int
    action_id: int


@dataclass(frozen=True, slots=True)
class StrategyProcedure:
    strategy_id: MemoryId
    steps: tuple[StrategyStep, ...]


class PersistentPlanningGraph:
    """Immutable read-side planning graph reconstructed from active memories."""

    def __init__(
        self,
        *,
        transitions: dict[MemoryId, tuple[MemoryId, ...]],
        success_nodes: frozenset[MemoryId],
        failure_nodes: frozenset[MemoryId],
        stall_nodes: frozenset[MemoryId],
        option_loss_nodes: frozenset[MemoryId],
        m1_location: dict[MemoryId, tuple[int, int]],
        strategies: dict[MemoryId, StrategyProcedure],
    ) -> None:
        self.transitions = transitions
        self.success_nodes = success_nodes
        self.failure_nodes = failure_nodes
        self.stall_nodes = stall_nodes
        self.option_loss_nodes = option_loss_nodes
        self.m1_location = m1_location
        self.strategies = strategies

    @classmethod
    def from_view(cls, view: MemoryReadView) -> "PersistentPlanningGraph":
        m1_location = cls._reverse_contingency_index(view)
        transition_sets: dict[MemoryId, set[MemoryId]] = defaultdict(set)
        success: set[MemoryId] = set()
        failure: set[MemoryId] = set()
        stall: set[MemoryId] = set()
        option_loss: set[MemoryId] = set()
        raw_strategy_steps: dict[MemoryId, dict[int, set[MemoryId]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for source, relation in view.adjacency:
            source_node = view.nodes.get(source)
            if not memory_is_active(source_node):
                continue
            targets = tuple(
                target
                for target in view.adjacency[(source, relation)]
                if memory_is_active(view.nodes.get(target))
            )
            if relation == REL_PLAN_TRANSITION:
                transition_sets[source].update(targets)
            elif relation == REL_PLAN_SUCCESS:
                success.add(source)
            elif relation == REL_PLAN_FAILURE:
                failure.add(source)
            elif relation == REL_PLAN_STALL:
                stall.add(source)
            elif relation == REL_PLAN_OPTION_LOSS:
                option_loss.add(source)
            elif (
                REL_STRATEGY_STEP_BASE
                <= relation
                < REL_STRATEGY_STEP_BASE + MAX_STRATEGY_STEPS
            ):
                if source_node is None or source_node.level != MemoryLevel.M6:
                    continue
                position = int(relation - REL_STRATEGY_STEP_BASE)
                raw_strategy_steps[source][position].update(targets)

        strategies: dict[MemoryId, StrategyProcedure] = {}
        for strategy_id, by_position in raw_strategy_steps.items():
            steps: list[StrategyStep] = []
            for position in sorted(by_position):
                pairs = sorted(
                    (
                        (m1_location[target][0], m1_location[target][1], target)
                        for target in by_position[position]
                        if target in m1_location
                    ),
                    key=lambda item: (item[0], item[1], int(item[2])),
                )
                if not pairs:
                    continue
                context, action, memory_id = pairs[0]
                steps.append(
                    StrategyStep(
                        position=position,
                        memory_id=memory_id,
                        context_signature=context,
                        action_id=action,
                    )
                )
            if steps and [step.position for step in steps] == list(range(len(steps))):
                strategies[strategy_id] = StrategyProcedure(
                    strategy_id,
                    tuple(steps),
                )

        return cls(
            transitions={
                source: tuple(sorted(targets, key=int))
                for source, targets in transition_sets.items()
            },
            success_nodes=frozenset(success),
            failure_nodes=frozenset(failure),
            stall_nodes=frozenset(stall),
            option_loss_nodes=frozenset(option_loss),
            m1_location=m1_location,
            strategies=strategies,
        )

    @staticmethod
    def _reverse_contingency_index(
        view: MemoryReadView,
    ) -> dict[MemoryId, tuple[int, int]]:
        packed = view.packed_cognition.contingencies
        output: dict[MemoryId, tuple[int, int]] = {}
        for row in range(len(packed.key_a)):
            context = int(packed.key_a[row])
            action = int(packed.key_b[row])
            start = int(packed.offsets[row])
            stop = start + int(packed.lengths[row])
            for raw_memory_id in packed.values[start:stop]:
                memory_id = MemoryId(int(raw_memory_id))
                node = view.nodes.get(memory_id)
                if (
                    node is not None
                    and node.level == MemoryLevel.M1
                    and memory_is_active(node)
                ):
                    output.setdefault(memory_id, (context, action))
        return output

    def evaluate(
        self,
        view: MemoryReadView,
        source_ids: Iterable[MemoryId],
        *,
        depth: int = 3,
        max_nodes: int = 64,
    ) -> PlanningSignal:
        sources = tuple(
            memory_id
            for memory_id in sorted(set(source_ids), key=int)
            if memory_is_active(view.nodes.get(memory_id))
        )
        if not sources:
            return PlanningSignal()

        weights = {
            memory_id: max(1, int(view.nodes[memory_id].support_count))
            for memory_id in sources
        }
        total_weight = max(1, sum(weights.values()))

        def weighted_marker(markers: frozenset[MemoryId]) -> float:
            return sum(
                weight
                for memory_id, weight in weights.items()
                if memory_id in markers
            ) / total_weight

        direct_failure = weighted_marker(self.failure_nodes)
        direct_stall = weighted_marker(self.stall_nodes)
        direct_option_loss = weighted_marker(self.option_loss_nodes)
        direct_success = weighted_marker(self.success_nodes)

        reached: set[MemoryId] = set()
        queue: deque[tuple[MemoryId, int]] = deque(
            (memory_id, 0) for memory_id in sources
        )
        best_depth: dict[MemoryId, int] = {
            memory_id: 0 for memory_id in sources
        }
        success_reach = direct_success
        descendant_failure = 0.0
        descendant_stall = 0.0
        descendant_option_loss = 0.0

        while queue and len(reached) < max(1, int(max_nodes)):
            current, distance = queue.popleft()
            if distance >= max(1, int(depth)):
                continue
            for target in self.transitions.get(current, ()):
                if not memory_is_active(view.nodes.get(target)):
                    continue
                next_distance = distance + 1
                old = best_depth.get(target)
                if old is not None and old <= next_distance:
                    continue
                best_depth[target] = next_distance
                reached.add(target)
                discount = 1.0 / (1.0 + next_distance)
                if target in self.success_nodes:
                    success_reach = max(success_reach, discount)
                if target in self.failure_nodes:
                    descendant_failure = max(descendant_failure, discount)
                if target in self.stall_nodes:
                    descendant_stall = max(descendant_stall, discount)
                if target in self.option_loss_nodes:
                    descendant_option_loss = max(descendant_option_loss, discount)
                if len(reached) >= max_nodes:
                    break
                queue.append((target, next_distance))

        reachability = (
            min(
                1.0,
                math.log1p(len(reached))
                / math.log1p(max(2, int(max_nodes))),
            )
            if reached
            else 0.0
        )
        return PlanningSignal(
            reachability=reachability,
            success_reachability=min(1.0, success_reach),
            failure_risk=min(
                1.0,
                max(direct_failure, 0.5 * descendant_failure),
            ),
            stall_risk=min(1.0, max(direct_stall, 0.5 * descendant_stall)),
            option_loss_risk=min(
                1.0,
                max(direct_option_loss, 0.5 * descendant_option_loss),
            ),
            reachable_nodes=len(reached),
        )
