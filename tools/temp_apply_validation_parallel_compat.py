from __future__ import annotations

from pathlib import Path

p = Path("src/v6/memory/v63_validation_parallel_completion.py")
text = p.read_text(encoding="utf-8")
old = '''    config: Any,\n    candidate_links: dict[str, set[str]] | None = None,\n) -> tuple[Any, Any, Any]:\n'''
new = '''    config: Any,\n    candidate_links: dict[str, set[str]] | None = None,\n    role_links: dict[str, dict[str, set[str]]] | None = None,\n    transfer_rate_cache: dict[Any, Any] | None = None,\n    future_role_rate_cache: dict[Any, Any] | None = None,\n) -> tuple[Any, Any, Any]:\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("validation cache signature not found")
    text = text.replace(old, new, 1)
old2 = '''        config=config,\n        candidate_links=candidate_links,\n    )\n'''
new2 = '''        config=config,\n        candidate_links=candidate_links,\n        role_links=role_links,\n        transfer_rate_cache=transfer_rate_cache,\n        future_role_rate_cache=future_role_rate_cache,\n    )\n'''
# replace the occurrence inside _build_functional_from_cache, not the earlier worker call
marker = text.find("def _build_functional_from_cache(")
if marker < 0:
    raise RuntimeError("cache function not found")
tail = text[marker:]
if new2 not in tail:
    if old2 not in tail:
        raise RuntimeError("cache fallback call not found")
    tail = tail.replace(old2, new2, 1)
    text = text[:marker] + tail
p.write_text(text, encoding="utf-8")
