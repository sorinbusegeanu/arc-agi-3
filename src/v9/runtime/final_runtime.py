from __future__ import annotations

from typing import Any, Iterable

from v9.cognition.grounding import GroundingMaturity

from .completed_runtime import CompletedContinuousMemoryRuntime


class FinalContinuousMemoryRuntime(CompletedContinuousMemoryRuntime):
    """Final v9.7.8 behavior authority and supervision integration."""

    def _payload_by_low_uid(self) -> dict[int, dict[str, Any]]:
        return {int(uid.lo): payload for uid, payload in self.graph.payloads.items()}

    def _grounded_policy_scores(self) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, dict[int, float]]]]:
        by_type: dict[str, dict[int, list[float]]] = {}
        by_context: dict[str, dict[int, dict[int, list[float]]]] = {}
        payload_by_low_uid = self._payload_by_low_uid()
        for key, state in self.grounding.eligible_states():
            _symbol_uid, interaction_uid, environment_id, _context_scope_id, _lineage_uid = key
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
            {environment_type: {action: sum(values) / len(values) for action, values in actions.items()} for environment_type, actions in by_type.items()},
            {
                environment_type: {
                    context: {action: sum(values) / len(values) for action, values in actions.items()}
                    for context, actions in contexts.items()
                }
                for environment_type, contexts in by_context.items()
            },
        )

    def grounding_action_scores(self, symbol_structure_uid: int, *, environment_instance_id: int | None = None) -> dict[int, float]:
        """Return validated symbol-conditioned native action scores."""
        payload_by_low_uid = self._payload_by_low_uid()
        by_action: dict[int, list[float]] = {}
        for key, state in self.grounding.eligible_states(environment_instance_id=environment_instance_id):
            symbol_uid, interaction_uid, _environment_id, _context_scope_id, _lineage_uid = key
            if int(symbol_uid) != int(symbol_structure_uid):
                continue
            payload = payload_by_low_uid.get(int(interaction_uid))
            if payload is None:
                continue
            action_token = payload.get("action_id", payload.get("executable_action_token"))
            if action_token is None:
                continue
            confidence = state.support / max(1e-9, state.support + state.contradiction)
            maturity = max(0.0, min(1.0, (int(state.maturity) - int(GroundingMaturity.G2)) / 3.0))
            by_action.setdefault(int(action_token), []).append(confidence * maturity)
        return {action: sum(values) / len(values) for action, values in by_action.items()}

    def _symbol_identity_for_structure(self, symbol_structure_uid: int):
        symbol_uid = self._uid_by_low(symbol_structure_uid)
        if symbol_uid is None:
            return None
        payload = self.graph.payloads.get(symbol_uid, {})
        identity = payload.get("symbol_identity")
        if isinstance(identity, (list, tuple)) and len(identity) == 4:
            return tuple(int(value) for value in identity)
        required = ("symbol_vocabulary_id", "symbol_stream_id", "symbol_id", "symbol_position")
        if all(key in payload for key in required):
            return tuple(int(payload[key]) for key in required)
        for raw_parent in payload.get("parents", ()):
            if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                continue
            from v9.memory.identity import MemoryUid
            parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
            parent_identity = self.graph.payloads.get(parent, {}).get("symbol_identity")
            if isinstance(parent_identity, (list, tuple)) and len(parent_identity) == 4:
                return tuple(int(value) for value in parent_identity)
        return None

    def record_symbolic_validation(self, **kwargs: Any):
        state = super().record_symbolic_validation(**kwargs)
        symbol_structure_uid = int(kwargs["symbol_structure_uid"])
        interaction_structure_uid = int(kwargs["interaction_structure_uid"])
        validation_kind = str(kwargs["validation_kind"])
        trial_id = str(kwargs["trial_id"])
        symbol_identity = self._symbol_identity_for_structure(symbol_structure_uid)
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
            root_payload = self.graph.payloads.get(root, {})
            action = root_payload.get("action_id", root_payload.get("executable_action_token"))
            if action is not None:
                metadata["action_id"] = int(action)
            parents = root_payload.get("parents", ())
            for raw_parent in parents:
                if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                    continue
                from v9.memory.identity import MemoryUid
                parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
                parent_payload = self.graph.payloads.get(parent, {})
                if "primary_valence" in parent_payload:
                    metadata["primary_valence"] = int(parent_payload["primary_valence"])
                    break
            frontier = {root}
            if root in self.graph.payloads:
                self.graph.payloads[root].update(metadata)
            from v9.memory.model import MemoryLevel
            for level in range(2, 8):
                memory_level = MemoryLevel(level)
                next_frontier = set()
                for uid in self.graph.uids_at_level(memory_level):
                    payload = self.graph.payloads.get(uid, {})
                    parent_keys = {
                        (int(raw[0]), int(raw[1]))
                        for raw in payload.get("parents", ())
                        if isinstance(raw, (list, tuple)) and len(raw) == 2
                    }
                    if any((int(parent.hi), int(parent.lo)) in parent_keys for parent in frontier):
                        payload.update(metadata)
                        next_frontier.add(uid)
                frontier.update(next_frontier)
        return state

    def metrics(self) -> dict[str, Any]:
        result = dict(super().metrics())
        trial_ids = sorted({trial_id for state in self.grounding.states.values() for trial_id in state.validation_trial_ids})
        result["grounding_validation_trials"] = len(trial_ids)
        result["grounding_validation_trial_ids"] = trial_ids
        result["grounding_context_scopes"] = len({key[3] for key in self.grounding.states})
        result["grounding_lineages"] = len({key[4] for key in self.grounding.states})
        result["grounding_environments"] = len({key[2] for key in self.grounding.states})
        return result

    def dashboard_metrics(self) -> dict[str, Any]:
        return self.metrics()
