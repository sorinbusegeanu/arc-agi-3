from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

from v9.cognition.grounding import GroundingMaturity
from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.m0_episode import M0Episode
from v9.memory.symbolic_relations import derive_symbolic_relations, shuffled_alignment_control
from v9.modalities.contract import PassiveSymbolEvent
from v9.modalities.symbols import SymbolOccurrence, SymbolTemporalPhase

from .completed_runtime import CompletedContinuousMemoryRuntime
from .memory_pipeline import DerivationTask, IngestionTask, derive_memory, prepare_ingestion
from .multiprocess import EncodedTransition


class FinalContinuousMemoryRuntime(CompletedContinuousMemoryRuntime):
    """Authoritative v9.7.8 symbolic-grounding runtime."""

    def _payload_by_low_uid(self) -> dict[int, dict[str, Any]]:
        return {int(uid.lo): payload for uid, payload in self.graph.payloads.items()}

    def _occurrence_payloads(self) -> dict[tuple[int, int], dict[str, Any]]:
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for payload in self.graph.payloads.values():
            raw = payload.get("symbol_occurrence_id")
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                result[(int(raw[0]), int(raw[1]))] = payload
        for _uid, (_node, payload, _evidence) in self._deferred_base_nodes.items():
            raw = payload.get("symbol_occurrence_id")
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                result[(int(raw[0]), int(raw[1]))] = payload
        return result

    def symbol_occurrences(
        self,
        *,
        vocabulary_id: int | None = None,
        symbol_id: int | None = None,
        stream_id: int | None = None,
        position_start: int | None = None,
        position_end: int | None = None,
        environment_instance_id: int | None = None,
        episode_id: int | None = None,
        temporal_phase: str | None = None,
        macro_step_start: int | None = None,
        macro_step_end: int | None = None,
        micro_step_start: int | None = None,
        micro_step_end: int | None = None,
        causal_watermark_start: int | None = None,
        causal_watermark_end: int | None = None,
        context_signature: int | None = None,
        nearby_action_id: int | None = None,
        transformation_signature: int | None = None,
        progress: bool | None = None,
        outcome: int | None = None,
        provenance_id: Any | None = None,
    ):
        rows = list(
            super().symbol_occurrences(
                vocabulary_id=vocabulary_id,
                symbol_id=symbol_id,
                environment_instance_id=environment_instance_id,
                episode_id=episode_id,
                temporal_phase=temporal_phase,
                macro_step_start=macro_step_start,
                macro_step_end=macro_step_end,
            )
        )
        payloads = self._occurrence_payloads()
        if provenance_id is not None:
            if hasattr(provenance_id, "hi") and hasattr(provenance_id, "lo"):
                provenance_key = (int(provenance_id.hi), int(provenance_id.lo))
            elif isinstance(provenance_id, (list, tuple)) and len(provenance_id) == 2:
                provenance_key = (int(provenance_id[0]), int(provenance_id[1]))
            else:
                raise ValueError("provenance_id must be EventUid-like or a two-integer tuple")
        else:
            provenance_key = None

        selected = []
        for row in rows:
            if stream_id is not None and int(row.stream_id.value) != int(stream_id):
                continue
            if position_start is not None and int(row.position) < int(position_start):
                continue
            if position_end is not None and int(row.position) > int(position_end):
                continue
            if micro_step_start is not None and int(row.micro_step) < int(micro_step_start):
                continue
            if micro_step_end is not None and int(row.micro_step) > int(micro_step_end):
                continue
            if causal_watermark_start is not None and int(row.causal_watermark) < int(causal_watermark_start):
                continue
            if causal_watermark_end is not None and int(row.causal_watermark) > int(causal_watermark_end):
                continue
            if provenance_key is not None and (int(row.provenance_id.hi), int(row.provenance_id.lo)) != provenance_key:
                continue
            payload = payloads.get((int(row.occurrence_id.hi), int(row.occurrence_id.lo)), {})
            if context_signature is not None and int(payload.get("nearby_context_signature", -1)) != int(context_signature):
                continue
            if nearby_action_id is not None and int(payload.get("nearby_action_id", -1)) != int(nearby_action_id):
                continue
            if transformation_signature is not None and int(payload.get("nearby_transformation_signature", -1)) != int(transformation_signature):
                continue
            if progress is not None and bool(payload.get("nearby_progress", False)) != bool(progress):
                continue
            if outcome is not None and int(payload.get("nearby_outcome", 0)) != int(outcome):
                continue
            selected.append(row)
        return tuple(selected)

    def symbol_occurrence_records(self, **filters: Any) -> tuple[tuple[Any, dict[str, Any]], ...]:
        payloads = self._occurrence_payloads()
        return tuple(
            (row, dict(payloads.get((int(row.occurrence_id.hi), int(row.occurrence_id.lo)), {})))
            for row in self.symbol_occurrences(**filters)
        )

    def _persist_prepared_occurrences(self, prepared: Any) -> None:
        super()._persist_prepared_occurrences(prepared)
        transition = prepared.transition
        contextual = {
            "nearby_context_signature": int(getattr(transition, "before_signature", 0)),
            "nearby_action_id": int(getattr(transition, "action_id", 0)),
            "nearby_transformation_signature": int(stable_u64(int(getattr(transition, "before_signature", 0)), int(getattr(transition, "after_signature", 0)), person=b"v9-symbol-nearby")),
            "nearby_progress": bool(getattr(transition, "task_success", False) or int(getattr(transition, "levels_completed", 0)) > 0),
            "nearby_outcome": int(getattr(transition, "primary_valence", 0)),
            "nearby_boundary": str(getattr(transition, "boundary_scope", "NONE")),
            "nearby_task_success": bool(getattr(transition, "task_success", False)),
            "nearby_task_failure": bool(getattr(transition, "task_failure", False)),
        }
        durable_updates: dict[MemoryUid, dict[str, Any]] = {}
        for symbol in getattr(prepared, "symbols", ()):
            uid = symbol.m0.uid
            if uid in self.graph.payloads:
                if hasattr(self, "canonical_store"):
                    durable_updates[uid] = contextual
                else:
                    self.graph.payloads[uid].update(contextual)
            elif uid in self._deferred_base_nodes:
                node, payload, evidence = self._deferred_base_nodes[uid]
                merged = dict(payload)
                merged.update(contextual)
                self._deferred_base_nodes[uid] = (node, merged, evidence)
        if durable_updates:
            self.replace_canonical_payloads(durable_updates)

    def _configured_task(self, sequence: int, transition: EncodedTransition) -> IngestionTask:
        scientific = self.config.scientific
        return IngestionTask(
            int(sequence),
            int(self._watermark) + 1,
            transition,
            min(int(scientific.symbol_budget_per_window), int(scientific.max_symbol_facts_per_window)),
            int(scientific.symbol_payload_bytes),
            int(scientific.max_cross_modal_facts_per_macro_event),
            str(scientific.symbol_deduplication_policy),
            int(scientific.symbol_window_time_span),
            str(scientific.symbol_codec_name),
            int(scientific.symbol_codec_version),
        )

    @staticmethod
    def _semantic_rows(adapter: Any, name: str, *args: Any) -> tuple[Any, ...]:
        fn = getattr(adapter, name, None)
        return tuple(fn(*args)) if callable(fn) else ()

    def record_interaction(self, adapter: Any, *, producer_id: int, producer_sequence: int, global_step: int, native_action: int, before_observation: Any, after_observation: Any, episode_id: Any, symbol_codec: Any | None = None):
        """Direct callers use the same prepared-ingestion contract as worker actors."""
        sequence = self.reserve_producer_sequence(int(producer_id), int(producer_sequence))
        identity = adapter.identity()
        boundary = adapter.boundary_event()
        progress_fn = getattr(adapter, "task_progress", None)
        progress = progress_fn() if callable(progress_fn) else SimpleNamespace(success=False, failure=False, truncated=False, level_index=0, levels_completed=0)
        before_signature = int(adapter.encode_observation(before_observation))
        after_signature = int(adapter.encode_observation(after_observation))
        action_schema_id = int(adapter.action_schema().schema_id)
        actions_after = tuple(sorted(set(int(value) for value in adapter.available_actions())))
        symbols: tuple[object, ...] = ()
        if bool(self.config.scientific.symbolic_grounding_enabled):
            raw_symbols = tuple(adapter.optional_symbol_stream())
            phase = SymbolTemporalPhase.AFTER_OUTCOME.value if int(boundary.primary_valence) != 0 or bool(progress.success) or bool(progress.failure) else SymbolTemporalPhase.AFTER_ACTION.value
            symbols = tuple({"token": value, "phase": phase, "macro_step": int(global_step), "micro_step": index} for index, value in enumerate(raw_symbols))
        semantic_before = self._semantic_rows(adapter, "semantic_observation", before_observation)
        semantic_after = self._semantic_rows(adapter, "semantic_observation", after_observation)
        semantic_action = self._semantic_rows(adapter, "semantic_action", native_action)
        semantic_delta = self._semantic_rows(adapter, "semantic_delta", semantic_before, semantic_after)
        transition = EncodedTransition(
            actor_id=int(producer_id), producer_sequence=sequence, global_step=int(global_step),
            environment_identity=(identity.family, identity.environment_type, identity.config, identity.instance),
            episode_id=int(episode_id.value), observation_schema_id=int(adapter.observation_schema().schema_id),
            before_signature=before_signature, action_id=int(adapter.encode_action(native_action)), after_signature=after_signature,
            available_actions_after=len(actions_after), primary_valence=int(boundary.primary_valence), symbols=symbols,
            curriculum_step=None, game_scenario=str(identity.environment_type), symbols_only=False,
            action_schema_id=action_schema_id,
            available_action_set_signature=int(stable_u64(action_schema_id, *actions_after, person=b"v9-action-set")),
            boundary_scope=str(boundary.scope.value), task_success=bool(progress.success), task_failure=bool(progress.failure),
            task_truncated=bool(progress.truncated), level_index=int(progress.level_index), levels_completed=int(progress.levels_completed),
            semantic_before=semantic_before, semantic_action=semantic_action, semantic_after=semantic_after, semantic_delta=semantic_delta,
        )
        prepared = prepare_ingestion(self._configured_task(sequence, transition))
        signatures = self.apply_prepared_ingestion(prepared)
        self._develop(signatures)
        if prepared.event is None:
            raise RuntimeError("interaction preparation unexpectedly omitted world event")
        return prepared.event.experience

    def record_symbol_stream(self, adapter: Any, *, producer_id: int, producer_sequence: int, episode_id: Any, symbol_codec: Any | None = None) -> int:
        if not bool(self.config.scientific.symbolic_grounding_enabled):
            return 0
        raw_symbols = tuple(adapter.optional_symbol_stream())
        if not raw_symbols:
            return 0
        sequence = self.reserve_producer_sequence(int(producer_id), int(producer_sequence))
        identity = adapter.identity()
        symbols = tuple({"token": value, "phase": SymbolTemporalPhase.COINCIDENT.value, "macro_step": sequence, "micro_step": index} for index, value in enumerate(raw_symbols))
        transition = EncodedTransition(
            actor_id=int(producer_id), producer_sequence=sequence, global_step=sequence,
            environment_identity=(identity.family, identity.environment_type, identity.config, identity.instance), episode_id=int(episode_id.value),
            observation_schema_id=int(adapter.observation_schema().schema_id), before_signature=0, action_id=0, after_signature=0,
            available_actions_after=len(tuple(adapter.available_actions())), primary_valence=0, symbols=symbols,
            curriculum_step=None, game_scenario=str(identity.environment_type), symbols_only=True,
        )
        prepared = prepare_ingestion(self._configured_task(sequence, transition))
        signatures = self.apply_prepared_ingestion(prepared)
        self._develop(signatures)
        return len(prepared.symbol_occurrences)

    def _ingest(self, event: Any) -> tuple[int, ...]:
        signatures = super()._ingest(event)
        if not isinstance(event, PassiveSymbolEvent):
            return signatures
        occurrence = SymbolOccurrence(
            event.identity.event_id,
            event.symbol_id,
            int(event.position),
            event.stream_id,
            event.vocabulary_id,
            f"external-passive:{int(self.config.scientific.symbol_grounding_schema_version)}",
            int(event.identity.causal_watermark),
            int(event.identity.producer_sequence),
            int(event.position),
            int(event.identity.environment_instance_id),
            event.identity.episode_id,
            event.identity.event_id,
            int(event.identity.modality_id.value),
            SymbolTemporalPhase.COINCIDENT.value,
            int(event.identity.producer_sequence),
        )
        self._index_symbol_occurrences((occurrence,))
        m0 = M0Episode.from_event(event, context_signature=0, payload_digest=self._payload_digest(event))
        payload = {
            "symbol_occurrence_id": [int(occurrence.occurrence_id.hi), int(occurrence.occurrence_id.lo)],
            "symbol_id": int(occurrence.symbol_id.value),
            "symbol_position": int(occurrence.position),
            "symbol_stream_id": int(occurrence.stream_id.value),
            "symbol_vocabulary_id": int(occurrence.vocabulary_id.value),
            "symbol_codec_id": str(occurrence.codec_id),
            "symbol_causal_watermark": int(occurrence.causal_watermark),
            "symbol_macro_step": int(occurrence.macro_step),
            "symbol_micro_step": int(occurrence.micro_step),
            "symbol_environment_instance_id": int(occurrence.environment_instance_id),
            "symbol_episode_id": int(occurrence.episode_id.value),
            "symbol_provenance_id": [int(occurrence.provenance_id.hi), int(occurrence.provenance_id.lo)],
            "symbol_modality_id": int(occurrence.modality_id),
            "symbol_temporal_phase": str(occurrence.temporal_phase),
            "symbol_source_sequence": int(occurrence.source_sequence),
        }
        if m0.uid in self.graph.payloads:
            if hasattr(self, "canonical_store"):
                self.replace_canonical_payloads({m0.uid: payload})
            else:
                self.graph.payloads[m0.uid].update(payload)
        elif m0.uid in self._deferred_base_nodes:
            node, current, evidence = self._deferred_base_nodes[m0.uid]
            merged = dict(current)
            merged.update(payload)
            self._deferred_base_nodes[m0.uid] = (node, merged, evidence)
        return signatures

    def _derive_prepared_symbolic_relations(self, prepared: Any, previous_interaction: Any) -> tuple[tuple[int, ...], set[int]]:
        rows = tuple(getattr(prepared, "symbols", ()) or ())
        if not rows:
            return (), set()
        transition = prepared.transition
        current = getattr(prepared, "m1g", None)
        watermark = max(int(row.event.identity.causal_watermark) for row in rows)
        cross_modal_limit = int(self.config.scientific.max_cross_modal_facts_per_macro_event)
        derived = list(
            derive_symbolic_relations(
                rows,
                interaction_grounding=current,
                previous_interaction_grounding=previous_interaction,
                transition=transition,
                causal_watermark=watermark,
                occurrences=getattr(prepared, "symbol_occurrences", ()),
                max_cross_modal_facts=cross_modal_limit,
            )
        )
        cross_modal_used = sum(int(item.relation.channel.value == "CROSS_MODAL") for item in derived)
        if previous_interaction is not None and cross_modal_used < cross_modal_limit:
            for index, row in enumerate(rows):
                if cross_modal_used >= cross_modal_limit:
                    break
                occurrence = prepared.symbol_occurrences[index] if index < len(prepared.symbol_occurrences) else None
                derived.append(shuffled_alignment_control(row, previous_interaction, causal_watermark=watermark, occurrence=occurrence))
                cross_modal_used += 1
        signatures: list[int] = []
        families: set[int] = set()
        for item in derived:
            signatures.append(self._record_normalized(item.relation, defer_publication=True, payload_extra=item.payload()))
            self._register_family_relation(item.relation)
            families.add(int(item.relation.family_signature or item.relation.structural_signature))
        return tuple(signatures), families

    def _develop_shared_families(self, family_ids: set[int]) -> None:
        for family_id in sorted(family_ids):
            rows = tuple(
                row for row in self._m1n_family_occurrences.get(int(family_id), ())
                if float(getattr(row, "support", 1.0)) > float(getattr(row, "contradiction", 0.0))
                and float(getattr(row, "support", 1.0)) > 0.0
            )
            channels = {row.channel.value for row in rows}
            evidence = {uid for row in rows for uid in row.provenance.evidence}
            if len(rows) < 2 or len(evidence) < 2 or "WORLD" not in channels or not ({"SYMBOL", "CROSS_MODAL"} & channels):
                continue
            support = len(rows)
            if support <= int(self._shared_family_support.get(int(family_id), 0)):
                continue
            result = derive_memory(DerivationTask(0, int(family_id), rows, support, tuple(sorted(self._formation_environments)), int(self._watermark)))
            super().apply_derivation_results_batch((result,))
            self._shared_family_support[int(family_id)] = support

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

    def grounding_consequence_prediction(self, interaction_structure_uid: int) -> dict[str, float]:
        root = self._uid_by_low(int(interaction_structure_uid))
        if root is None:
            return {"confidence": 0.0, "future_option_bonus": 0.0, "deliberation_support": 0.0}
        payload = self.graph.payloads.get(root, {})
        authority = float(payload.get("grounding_authority", self.grounding_prediction_score(interaction_structure_uid)))
        return {
            "confidence": max(0.0, min(1.0, authority)),
            "future_option_bonus": 0.25 * max(0.0, authority),
            "deliberation_support": max(0.0, authority),
        }

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
        effect = float(kwargs["effect"])
        environment_id = int(kwargs["environment_instance_id"])
        target_environment_id = kwargs.get("target_environment_id")
        context_scope_id = int(kwargs.get("context_scope_id", 0))
        lineage_uid = int(kwargs.get("lineage_uid", 0))
        validation_watermark = int(self._watermark)
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
            "cross_modal_control": "aligned" if effect > 0.0 else "contradicted",
            "world_to_symbol_generalization": validation_kind in {"generalization", "heldout_transfer", "composition", "symbol_mediated_learning"},
            "heldout_transfer": validation_kind in {"heldout_transfer", "composition", "symbol_mediated_learning"},
            "novel_composition": validation_kind == "composition",
            "symbol_mediated_learning": validation_kind == "symbol_mediated_learning",
            "grounding_validation_trial_id": trial_id,
            "validation_causal_watermark": validation_watermark,
            "grounding_evaluation_watermark": validation_watermark,
            "grounding_source_environment_id": environment_id,
            "grounding_target_environment_id": None if target_environment_id is None else int(target_environment_id),
            "grounding_context_scope_id": context_scope_id,
            "grounding_lineage_uid": lineage_uid,
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
                parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
                parent_payload = self.graph.payloads.get(parent, {})
                if "primary_valence" in parent_payload:
                    metadata["primary_valence"] = int(parent_payload["primary_valence"])
                    break
            frontier = {root}
            durable_updates: dict[MemoryUid, dict[str, Any]] = {}
            if root in self.graph.payloads:
                if hasattr(self, "canonical_store"):
                    durable_updates[root] = metadata
                else:
                    self.graph.payloads[root].update(metadata)
            from v9.memory.model import MemoryLevel
            for level in range(2, 8):
                memory_level = MemoryLevel(level)
                next_frontier = set()
                for uid in self.graph.uids_at_level(memory_level):
                    payload = self.graph.payloads.get(uid, {})
                    parent_keys = {(int(raw[0]), int(raw[1])) for raw in payload.get("parents", ()) if isinstance(raw, (list, tuple)) and len(raw) == 2}
                    if any((int(parent.hi), int(parent.lo)) in parent_keys for parent in frontier):
                        if hasattr(self, "canonical_store"):
                            durable_updates[uid] = metadata
                        else:
                            payload.update(metadata)
                        next_frontier.add(uid)
                frontier.update(next_frontier)
            if durable_updates:
                self.replace_canonical_payloads(durable_updates)
        return state

    def metrics(self) -> dict[str, Any]:
        result = dict(super().metrics())
        trial_ids = sorted({trial_id for state in self.grounding.states.values() for trial_id in state.validation_trial_ids})
        result["grounding_validation_trials"] = len(trial_ids)
        result["grounding_validation_trial_ids"] = trial_ids
        result["grounding_context_scopes"] = len({key[3] for key in self.grounding.states})
        result["grounding_lineages"] = len({key[4] for key in self.grounding.states})
        result["grounding_environments"] = len({key[2] for key in self.grounding.states})
        result["symbol_grounding_provenance"] = {
            "design_version": str(self.config.scientific.design_version),
            "scientific_config_id": str(self.config.scientific.config_id.value),
            "symbol_grounding_schema_version": int(self.config.scientific.symbol_grounding_schema_version),
            "symbol_codec_name": str(self.config.scientific.symbol_codec_name),
            "symbol_codec_version": int(self.config.scientific.symbol_codec_version),
            "model_version": str(self.unified_telemetry.model_version),
            "graph_generation": int(self.graph.generation),
            "validation_trial_ids": trial_ids,
        }
        return result

    def dashboard_metrics(self) -> dict[str, Any]:
        return self.metrics()
