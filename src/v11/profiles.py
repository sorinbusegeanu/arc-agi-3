"""Resolve live action sets and compose their reusable prompt capabilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


KNOWN_ACTION_SETS: dict[frozenset[str], str] = {
    frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4"}): "a1234",
    frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"}): "a12346",
    frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}): "a12345",
    frozenset({"ACTION6"}): "a6",
    frozenset({"ACTION5", "ACTION6"}): "a56",
    frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"}): "a123456",
    frozenset({"ACTION5"}): "a5",
    frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION5"}): "a1235",
    frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}): "a1234567",
}

CAPABILITY_ACTIONS: dict[str, frozenset[str]] = {
    "discrete_control": frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4"}),
    "context_action5": frozenset({"ACTION5"}),
    "coordinate_action6": frozenset({"ACTION6"}),
    "reversible_action7": frozenset({"ACTION7"}),
}
CAPABILITY_ORDER = tuple(CAPABILITY_ACTIONS)


@dataclass(frozen=True)
class ActionProfile:
    profile_id: str
    action_ids: tuple[str, ...]
    action_signature: str
    argument_schema_fingerprint: str
    capabilities: tuple[str, ...]


def _action_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        action_id = value.get("id", value.get("action"))
        return str(action_id) if action_id is not None else None
    return None


def _schema_fingerprint(legal_actions: Sequence[Any], action_ids: Sequence[str]) -> str:
    by_id: dict[str, Any] = {}
    for item in legal_actions:
        action_id = _action_id(item)
        if action_id is None or action_id == "RESET" or action_id not in action_ids:
            continue
        parameters = item.get("parameters") if isinstance(item, Mapping) else None
        by_id[action_id] = parameters if parameters is not None else {}
    canonical = json.dumps(by_id, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_action_profile(legal_actions: Any) -> ActionProfile:
    """Resolve by sorted environment action IDs; RESET and schema details do not split profiles."""
    if isinstance(legal_actions, Mapping):
        rows: list[Any] = [
            {"id": action_id, "parameters": value.get("parameters", {}) if isinstance(value, Mapping) else {}}
            for action_id, value in legal_actions.items()
        ]
    elif isinstance(legal_actions, Sequence) and not isinstance(legal_actions, (str, bytes)):
        rows = list(legal_actions)
    else:
        rows = []
    action_ids = tuple(sorted({
        action_id for item in rows
        if (action_id := _action_id(item)) is not None and action_id != "RESET"
    }))
    action_set = frozenset(action_ids)
    profile_id = KNOWN_ACTION_SETS.get(action_set, "generic")
    signature = ",".join(action_ids)
    capabilities = tuple(
        capability for capability in CAPABILITY_ORDER
        if action_set & CAPABILITY_ACTIONS[capability]
    )
    if len(capabilities) > 1:
        capabilities = (*capabilities, "cross_action_dependency")
    return ActionProfile(
        profile_id=profile_id,
        action_ids=action_ids,
        action_signature=signature,
        argument_schema_fingerprint=_schema_fingerprint(rows, action_ids),
        capabilities=capabilities,
    )


def compose_profile_prompt(
    profile: ActionProfile,
    capability_dir: Path | str,
) -> tuple[str, str]:
    """Return active capability instructions and a revision hash for their exact composition."""
    root = Path(capability_dir)
    components: list[tuple[str, str]] = []
    for capability in profile.capabilities:
        filename = "cross_action_dependency" if capability == "cross_action_dependency" else capability
        path = root / f"{filename}.md"
        if not path.is_file():
            raise FileNotFoundError(f"missing v11 profile capability prompt: {path}")
        components.append((filename, path.read_text(encoding="utf-8").strip()))
    prompt = "\n\n".join(text for _name, text in components)
    payload = json.dumps(components, ensure_ascii=False, separators=(",", ":"))
    revision = hashlib.sha256((profile.profile_id + "\0" + payload).encode("utf-8")).hexdigest()
    return prompt, revision


__all__ = [
    "ActionProfile", "CAPABILITY_ACTIONS", "KNOWN_ACTION_SETS",
    "compose_profile_prompt", "resolve_action_profile",
]
