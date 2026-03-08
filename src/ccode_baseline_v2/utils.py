"""utils.py — shared utility functions for ccode_baseline_v2."""
from __future__ import annotations


def to_action_key(action) -> str:
    """Convert any action representation to a canonical 'ACTIONn' string.

    ARC engine actions may be returned as:
      - A GameAction enum with a .name attribute ('ACTION1', 'ACTION2', ...)
      - A 1-based integer index (1 → 'ACTION1')
      - A string already in canonical form
    """
    if isinstance(action, str):
        name = action.strip()
        if name.upper().startswith("ACTION") or name.upper() in ("RESET",):
            return name.upper() if not name.startswith("ACTION") else name
        return name
    # Try .name attribute (GameAction enum)
    name = getattr(action, "name", None)
    if name and isinstance(name, str):
        return name
    # Fallback: assume 1-based integer index
    try:
        return f"ACTION{int(action)}"
    except (ValueError, TypeError):
        return str(action)
