from __future__ import annotations


def aggregate_exhaustion_signals(retries: dict[str, dict], *, threshold: int) -> dict[str, dict]:
    exhausted = exhaustion_snapshot(retries, threshold=threshold)
    signals: dict[str, dict] = {}
    for scope, keys in exhausted.items():
        for key in keys:
            pattern_key = f"exhaustion:{scope}:{key}"
            signals[pattern_key] = {
                "pattern_key": pattern_key,
                "scope": scope,
                "count": 1,
                "metadata": {"source": "session_exhaustion"},
            }
    return signals


def exhaustion_snapshot(retries: dict[str, dict], *, threshold: int) -> dict[str, list[str]]:
    exhausted: dict[str, list[str]] = {"candidate": [], "target": [], "area": []}
    for key, row in retries.items():
        scope = str(row.get("scope", "candidate"))
        recent_failures = int(row.get("recent_failures", 0))
        failures = int(row.get("failures", 0))
        if recent_failures >= threshold or failures >= threshold + 1:
            exhausted.setdefault(scope, []).append(key)
    for scope in exhausted:
        exhausted[scope] = sorted(exhausted[scope])
    return exhausted


def exhausted_candidates(retries: dict[str, dict], threshold: int) -> set[str]:
    return set(exhaustion_snapshot(retries, threshold=threshold).get("candidate", []))
