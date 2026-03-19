from __future__ import annotations

import ray

from v3_1.contracts.messages import PersistentMemoryFlushRequest, PersistentMemoryFlushResult, PersistentMemoryLoadRequest, PersistentMemoryLoadResult
from v3_1.storage.artifact_store import ArtifactStore
from v3_1.storage.manifests import manifest_entry, persistent_memory_database_manifest, persistent_memory_flush_manifest, session_memory_snapshot_manifest
from v3_1.storage.persistent_memory import PersistentMemoryStore
from v3_1.storage.session_store import SessionStore
from v3_1.storage.sqlite_index import SQLiteIndex


@ray.remote
class StorageAgent:
    def __init__(self, *, root_dir: str, sqlite_path: str | None = None, persistent_memory_db_path: str | None = None, persistence_flags: dict | None = None) -> None:
        self.persistent_memory = PersistentMemoryStore.from_storage_root(root_dir, persistent_memory_db_path) if persistent_memory_db_path else None
        self.store = ArtifactStore(root_dir, persistent_memory_store=self.persistent_memory)
        self.session_store = SessionStore()
        self.sqlite_index = SQLiteIndex(sqlite_path) if sqlite_path else None
        self.persistence_flags = persistence_flags or {}
        if self.persistent_memory is not None:
            self.session_store.set_persistent_memory_db_location(self.persistent_memory.db_path)
            db_entry = persistent_memory_database_manifest(self.persistent_memory.db_path)
            self.session_store.record(db_entry)
            if self.sqlite_index is not None:
                self.sqlite_index.insert_manifest(kind=db_entry["kind"], name=db_entry["name"], location=db_entry["location"], metadata=db_entry["metadata"])

    def persist(self, *, session_id: str, round_id: int, kind: str, name: str, payload) -> str:
        location = self.store.write_json(session_id=session_id, round_id=round_id, artifact_name=name, payload=payload)
        if kind == "snapshot" and name.startswith("memory_pass"):
            memory_version = getattr(payload, "memory_version", None)
            created_pass_id = getattr(payload, "created_pass_id", 0)
            entry = session_memory_snapshot_manifest(name, location, memory_version=str(memory_version or ""), round_id=round_id, pass_id=int(created_pass_id))
            self.session_store.record_memory_snapshot(entry)
            if self.persistent_memory is not None:
                self.persistent_memory.record_memory_snapshot_reference(
                    session_id=session_id,
                    round_id=round_id,
                    pass_id=int(created_pass_id),
                    memory_version=str(memory_version or ""),
                    snapshot_path=location,
                    metadata={"artifact_name": name},
                )
        else:
            entry = manifest_entry(kind, name, location)
            self.session_store.record(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=entry["kind"], name=entry["name"], location=location, metadata=entry["metadata"])
        return location

    def persist_bytes(self, *, session_id: str, round_id: int, kind: str, name: str, payload: bytes) -> str:
        location = self.store.write_bytes(session_id=session_id, round_id=round_id, artifact_name=name, payload=payload)
        entry = manifest_entry(kind, name, location)
        self.session_store.record(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=kind, name=name, location=location, metadata=entry["metadata"])
        return location

    def persist_visualization_bytes(self, *, session_id: str, kind: str, name: str, payload: bytes) -> str:
        location = self.store.write_visualization_bytes(session_id=session_id, artifact_name=name, payload=payload)
        entry = manifest_entry(kind, name, location)
        self.session_store.record(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=kind, name=name, location=location, metadata=entry["metadata"])
        return location

    def persist_session_json(self, *, session_id: str, kind: str, name: str, payload) -> str:
        location = self.store.write_session_json(session_id=session_id, artifact_name=name, payload=payload)
        entry = manifest_entry(kind, name, location)
        self.session_store.record(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=kind, name=name, location=location, metadata=entry["metadata"])
        return location

    def persist_session_bytes(self, *, session_id: str, kind: str, name: str, payload: bytes) -> str:
        location = self.store.write_session_bytes(session_id=session_id, artifact_name=name, payload=payload)
        entry = manifest_entry(kind, name, location)
        self.session_store.record(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=kind, name=name, location=location, metadata=entry["metadata"])
        return location

    def load_persistent_memory(self, request: PersistentMemoryLoadRequest) -> PersistentMemoryLoadResult:
        if self.persistent_memory is None:
            return PersistentMemoryLoadResult(
                session_id=request.session_id,
                run_id=request.run_id,
                game_id=request.game_id,
                db_path="",
                loaded=False,
                priors={},
                metadata={"disabled": True},
            )
        return self.persistent_memory.load_priors(request)

    def flush_persistent_memory(self, request: PersistentMemoryFlushRequest) -> PersistentMemoryFlushResult:
        batch = request.batch
        def _eligible(rows):
            return tuple(
                row
                for row in rows
                if bool(dict(row.get("metadata", {})).get("allowed_for_durable_write", True))
            )
        filtered_request = PersistentMemoryFlushRequest(
            session_id=request.session_id,
            run_id=request.run_id,
            game_id=request.game_id,
            flush_id=request.flush_id,
            session_snapshot_path=request.session_snapshot_path,
            metadata=request.metadata,
            batch=type(batch)(
                session_id=batch.session_id,
                run_id=batch.run_id,
                game_id=batch.game_id,
                round_id=batch.round_id,
                pass_id=batch.pass_id,
                batch_id=batch.batch_id,
                source_memory_version=batch.source_memory_version,
                skills=_eligible(batch.skills),
                skill_stats=_eligible(batch.skill_stats) if self.persistence_flags.get("persist_skill_stats", True) else (),
                candidate_outcomes=_eligible(batch.candidate_outcomes) if self.persistence_flags.get("persist_candidate_outcomes", True) else (),
                failure_patterns=_eligible(batch.failure_patterns) if self.persistence_flags.get("persist_failure_patterns", True) else (),
                recovery_patterns=_eligible(batch.recovery_patterns) if self.persistence_flags.get("persist_recovery_patterns", True) else (),
                poi_patterns=_eligible(batch.poi_patterns) if self.persistence_flags.get("persist_poi_patterns", True) else (),
                trigger_patterns=_eligible(batch.trigger_patterns) if self.persistence_flags.get("persist_trigger_patterns", True) else (),
                consequence_patterns=_eligible(batch.consequence_patterns) if self.persistence_flags.get("persist_consequence_patterns", True) else (),
                entity_signatures=_eligible(batch.entity_signatures) if self.persistence_flags.get("persist_entity_signatures", True) else (),
                area_signatures=_eligible(batch.area_signatures) if self.persistence_flags.get("persist_area_signatures", True) else (),
                mechanic_hypotheses=_eligible(batch.mechanic_hypotheses) if self.persistence_flags.get("persist_mechanic_hypotheses", True) else (),
                mechanic_graph_nodes=_eligible(batch.mechanic_graph_nodes),
                mechanic_graph_edges=_eligible(batch.mechanic_graph_edges),
                durable_dependency_paths=_eligible(batch.durable_dependency_paths),
                deterministic_supported_paths=_eligible(batch.deterministic_supported_paths),
                llm_supported_paths=_eligible(batch.llm_supported_paths),
                deterministic_llm_agreements=_eligible(batch.deterministic_llm_agreements),
                repeated_validated_hypotheses=_eligible(batch.repeated_validated_hypotheses),
                contradicted_llm_proposals=_eligible(batch.contradicted_llm_proposals),
                deterministic_hypothesis_proposals=_eligible(batch.deterministic_hypothesis_proposals),
                llm_hypothesis_proposals=_eligible(batch.llm_hypothesis_proposals),
                proposal_validation_state=_eligible(batch.proposal_validation_state),
                proposal_agreement_groups=_eligible(batch.proposal_agreement_groups),
                proposal_outcome_summaries=_eligible(batch.proposal_outcome_summaries),
                ranker_state=_eligible(batch.ranker_state) if self.persistence_flags.get("persist_ranker_state", True) else (),
                metadata=batch.metadata,
            ),
        )
        result = self.store.flush_persistent_memory(filtered_request)
        entry = persistent_memory_flush_manifest(
            location=result.db_path,
            flush_id=result.flush_id,
            db_path=result.db_path,
            rows_written=result.rows_written,
        )
        self.session_store.record_persistent_memory_flush(entry)
        if self.sqlite_index is not None:
            self.sqlite_index.insert_manifest(kind=entry["kind"], name=entry["name"], location=entry["location"], metadata=entry["metadata"])
        return result

    def manifests(self) -> list[dict]:
        return list(self.session_store.manifests)

    def get_root_dir(self) -> str:
        return str(self.store.root_dir)
