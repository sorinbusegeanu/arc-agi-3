from __future__ import annotations

from typing import Any

from v9.cognition.grounding import GroundingMaturity
from v9.memory.model import MemoryLevel


def install_grounding_action_index(runtime_cls: type) -> None:
    """Replace full-graph grounding policy scans with an incremental M1 index."""
    if getattr(runtime_cls, "_grounding_action_index_installed", False):
        return

    original_init = runtime_cls.__init__
    original_restore = runtime_cls._restore
    original_defer = runtime_cls._defer_base_group
    original_publish = runtime_cls._publish
    original_publish_group = runtime_cls._publish_group

    def _is_action_payload(payload: dict[str, Any]) -> bool:
        return payload.get("action_id", payload.get("executable_action_token")) is not None

    def _index_row(self: Any, uid: Any, payload: dict[str, Any]) -> None:
        if not _is_action_payload(payload):
            return
        self.__dict__.setdefault("_grounding_action_payload_by_low", {})[int(uid.lo)] = (uid, payload)

    def _rebuild(self: Any) -> None:
        index: dict[int, tuple[Any, dict[str, Any]]] = {}
        for uid in tuple(self.graph._uids_by_level[MemoryLevel.M1]):
            payload = self.graph.payloads.get(uid)
            if payload is None or not _is_action_payload(payload):
                continue
            index[int(uid.lo)] = (uid, payload)
        for uid, (node, payload, _evidence) in getattr(self, "_deferred_base_nodes", {}).items():
            if node.level is MemoryLevel.M1 and _is_action_payload(payload):
                index[int(uid.lo)] = (uid, payload)
        self._grounding_action_payload_by_low = index
        self.set_telemetry_gauge("grounding_action_index_size", len(index))
        self.set_telemetry_gauge(
            "grounding_action_index_rebuilds",
            int(getattr(self, "_grounding_action_index_rebuilds", 0)) + 1,
        )
        self._grounding_action_index_rebuilds = int(getattr(self, "_grounding_action_index_rebuilds", 0)) + 1

    def runtime_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _rebuild(self)

    def restore(self: Any, *args: Any, **kwargs: Any) -> None:
        original_restore(self, *args, **kwargs)
        _rebuild(self)

    def defer_base_group(self: Any, rows: Any) -> None:
        original_defer(self, rows)
        for node, payload, _evidence in rows:
            if node.level is MemoryLevel.M1:
                _index_row(self, node.uid, payload)

    def publish(self: Any, node: Any, payload: dict[str, Any], evidence: Any, **kwargs: Any) -> None:
        original_publish(self, node, payload, evidence, **kwargs)
        if node.level is MemoryLevel.M1 and node.uid in self.graph.payloads:
            _index_row(self, node.uid, self.graph.payloads[node.uid])

    def publish_group(self: Any, rows: Any) -> bool:
        accepted = original_publish_group(self, rows)
        if accepted:
            for node, payload, _evidence in rows:
                if node.level is MemoryLevel.M1:
                    _index_row(self, node.uid, self.graph.payloads.get(node.uid, payload))
        return accepted

    def _lookup(self: Any, interaction_uid_low: int) -> dict[str, Any] | None:
        index = self.__dict__.setdefault("_grounding_action_payload_by_low", {})
        row = index.get(int(interaction_uid_low))
        if row is None:
            return None
        uid, cached_payload = row
        live = self.graph.payloads.get(uid)
        if live is not None:
            if live is not cached_payload:
                index[int(interaction_uid_low)] = (uid, live)
            return live
        deferred = getattr(self, "_deferred_base_nodes", {}).get(uid)
        if deferred is not None:
            payload = deferred[1]
            index[int(interaction_uid_low)] = (uid, payload)
            return payload
        index.pop(int(interaction_uid_low), None)
        return None

    def grounded_policy_scores(self: Any):
        by_type: dict[str, dict[int, list[float]]] = {}
        by_context: dict[str, dict[int, dict[int, list[float]]]] = {}
        scanned = 0
        for key, state in self.grounding.eligible_states():
            scanned += 1
            _symbol_uid, interaction_uid, environment_id, _context_scope_id, _lineage_uid = key
            payload = _lookup(self, int(interaction_uid))
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
        self.set_telemetry_gauge("grounding_policy_states_scanned", scanned)
        self.set_telemetry_gauge("grounding_policy_full_graph_scan", 0)
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

    def grounding_action_scores(self: Any, symbol_structure_uid: int, *, environment_instance_id: int | None = None) -> dict[int, float]:
        by_action: dict[int, list[float]] = {}
        for key, state in self.grounding.eligible_states(environment_instance_id=environment_instance_id):
            symbol_uid, interaction_uid, _environment_id, _context_scope_id, _lineage_uid = key
            if int(symbol_uid) != int(symbol_structure_uid):
                continue
            payload = _lookup(self, int(interaction_uid))
            if payload is None:
                continue
            action_token = payload.get("action_id", payload.get("executable_action_token"))
            if action_token is None:
                continue
            confidence = state.support / max(1e-9, state.support + state.contradiction)
            maturity = max(0.0, min(1.0, (int(state.maturity) - int(GroundingMaturity.G2)) / 3.0))
            by_action.setdefault(int(action_token), []).append(confidence * maturity)
        return {action: sum(values) / len(values) for action, values in by_action.items()}

    runtime_cls.__init__ = runtime_init
    runtime_cls._restore = restore
    runtime_cls._defer_base_group = defer_base_group
    runtime_cls._publish = publish
    runtime_cls._publish_group = publish_group
    runtime_cls._grounded_policy_scores = grounded_policy_scores
    if hasattr(runtime_cls, "grounding_action_scores"):
        runtime_cls.grounding_action_scores = grounding_action_scores
    runtime_cls._grounding_action_index_installed = True
