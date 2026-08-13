from __future__ import annotations

from dataclasses import replace

from v7.environment.cognition import ContextualActionScorer
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.ids import MemoryId


class ValidatedContextualActionScorer(ContextualActionScorer):
    def score_actions(self, *, view, contexts, actions, overlay):
        rows = super().score_actions(view=view, contexts=contexts, actions=actions, overlay=overlay)
        output = []
        for row in rows:
            valid = []
            rejected = []
            for raw_id in row.support.concept_ids:
                memory_id = MemoryId(int(raw_id))
                node = view.nodes.get(memory_id)
                if node is not None and int(node.status_flags) & int(ConceptValidationStatus.TRANSFER_REJECTED):
                    rejected.append(int(raw_id))
                else:
                    valid.append(int(raw_id))
            penalty = 0.08 * self._memory_strength(view, rejected)
            output.append(replace(
                row,
                score=float(row.score) - penalty,
                support=replace(row.support, concept_ids=tuple(valid)),
            ))
        return tuple(output)
