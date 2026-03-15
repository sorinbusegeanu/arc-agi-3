from __future__ import annotations


def manifest_entry(kind: str, name: str, location: str, metadata: dict | None = None) -> dict:
    return {"kind": kind, "name": name, "location": location, "metadata": metadata or {}}


def session_memory_snapshot_manifest(name: str, location: str, *, memory_version: str, round_id: int, pass_id: int) -> dict:
    return manifest_entry("session_memory_snapshot", name, location, {"memory_version": memory_version, "round_id": round_id, "pass_id": pass_id})


def persistent_memory_flush_manifest(location: str, *, flush_id: str, db_path: str, rows_written: dict) -> dict:
    return manifest_entry("persistent_memory_flush", f"{flush_id}.json", location, {"db_path": db_path, "rows_written": rows_written})


def persistent_memory_database_manifest(db_path: str) -> dict:
    return manifest_entry("persistent_memory_db", "persistent_memory.sqlite", db_path, {"stable": True})
