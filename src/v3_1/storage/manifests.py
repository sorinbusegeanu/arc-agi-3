from __future__ import annotations


def manifest_entry(kind: str, name: str, location: str) -> dict:
    return {"kind": kind, "name": name, "location": location}

