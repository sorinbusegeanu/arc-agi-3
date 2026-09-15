from __future__ import annotations

from typing import Any

from v9.cognition.grounding import GroundingMaturity

from .completed_runtime import CompletedContinuousMemoryRuntime


class FinalContinuousMemoryRuntime(CompletedContinuousMemoryRuntime):
    """Final v9.7.8 behavior authority and supervision integration."""

    def _grounded_policy_scores(self) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, dict[int, float]]]]:
        by_type: dict[str, dict[int, list[float]]] = {}
        by_context: dict[str, dict[int, dict[int, list[float]]]] = {}
        payload_by_low_uid: dict[int, dict[str, Any]] = {
            int(uid.lo): payload for uid, payload in self.graph.payloads.items()
        }
        for key, state in self.grounding.eligible_states():
            _symbol_uid, interaction_uid, environment_id, context_scope_id, _lineage_uid = key
            payload = payload_by_low_uid.get(int(interaction_uid))
            if payload is None:
                continue
            action_token = payload.get("action_id", payload.get("executable_action_token"))
            if action_token is None:
                continue
            try:
                environment_type = str(self.environments.resolve(int(environment_id)).environment_type)
            except KeyError:
                continue
            action = int(action_token)
            confidence = state.support / max(1e-9, state.support + state.contradiction)
            maturity = max(0.0, min(1.0, (int(state.maturity) - int(GroundingMaturity.G2)) / 3.0))
            score = confidence * maturity
            by_type.setdefault(environment_type, {}).setdefault(action, []).append(score)
            context = payload.get("context_signature", payload.get("grounded_context_signature"))
            if context is not None:
                by_context.setdefault(environment_type, {}).setdefault(int(context), {}).setdefault(action, []).append(score)
        return (
            {
                environment_type: {action: sum(values) / len(values) for action, values in actions.items()}
                for environment_type, actions in by_type.items()
            },
            {
                environment_type: {
                    context: {action: sum(values) / len(values) for action, values in actions.items()}
                    for context, actions in contexts.items()
                }
                for environment_type, contexts in by_context.items()
            },
        )

    def record_symbolic_validation(self, **kwargs: Any):
        state = super().record_symbolic_validation(**kwargs)
        symbol_structure_uid = int(kwargs["symbol_structure_uid"])
        interaction_structure_uid = int(kwargs["interaction_structure_uid"])
        validation_kind = str(kwargs["validation_kind"])
        trial_id = str(kwargs["trial_id"])
        symbol_uid = self._uid_by_low(symbol_structure_uid)
        symbol_identity = None
        if symbol_uid is not None:
            symbol_identity = self.graph.payloads.get(symbol_uid, {}).get("symbol_identity")
        relation_by_kind = {
            "prediction": "SYMBOL_TO_INTERACTION_PREDICTION",
            "generalization": "INTERACTION_TO_SYMBOL_GENERALIZATION",
            "heldout_transfer": "CROSS_MODAL_HELDOUT_TRANSFER",
            "composition": "CROSS_MODAL_COMPOSITION",
            "symbol_mediated_learning": "CROSS_MODAL_HELDOUT_TRANSFER",
        }
        metadata = {
            "grounding_symbol_structure_uid": symbol_structure_uid,
            "grounding_interaction_structure_uid": interaction_structure_uid,
            "symbol_relation": relation_by_kind[validation_kind],
            "cross_modal_control": "aligned" if bool(state.behavior_eligible) else "contradicted",
            "world_to_symbol_generalization": validation_kind in {"generalization", "heldout_transfer", "composition", "symbol_mediated_learning"},
            "grounding_validation_trial_id": trial_id,
        }
        if symbol_identity is not None:
            metadata["symbol_identity"] = symbol_identity
        root = self._uid_by_low(interaction_structure_uid)
        if root is not None:
            frontier = {root}
            for uid in tuple(frontier):
                if uid in self.graph.payloads:
                    self.graph.payloads[uid].update(metadata)
            for level in range(int(GroundingMaturity.G2), 8):
                try:
                    from v9.memory.model import MemoryLevel
                    memory_level = MemoryLevel(level)
                except ValueError:
                    continue
                next_frontier = set()
                for uid in self.graph.uids_at_level(memory_level):
                    payload = self.graph.payloads.get(uid, {})
                    parents = {
                        (int(raw[0]), int(raw[1]))
                        for raw in payload.get("parents", ())
                        if isinstance(raw, (list, tuple)) and len(raw) == 2
                    }
                    if any((int(parent.hi), int(parent.lo)) in parents for parent in frontier):
                        payload.update(metadata)
                        next_frontier.add(uid)
                frontier.update(next_frontier)
        return state
