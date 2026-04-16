from __future__ import annotations

from typing import Any

_REVERSE = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "UP": "DOWN",
    "DOWN": "UP",
}


def analyze_trace_for_redundancy(saved_trace) -> dict[str, Any]:
    actions = tuple(str(item) for item in getattr(saved_trace, "action_trace", ()))
    immediate_reversal_indices: list[int] = []
    oscillation_windows: list[tuple[int, int]] = []
    duplicate_prefix_lengths: list[int] = []
    repeated_subpaths: list[tuple[int, int]] = []
    stall_windows: list[tuple[int, int]] = []

    for idx in range(1, len(actions)):
        if _REVERSE.get(actions[idx - 1]) == actions[idx]:
            immediate_reversal_indices.append(idx - 1)

    for idx in range(3, len(actions)):
        if actions[idx - 3 : idx - 1] == actions[idx - 1 : idx + 1]:
            oscillation_windows.append((idx - 3, idx + 1))

    for window in range(2, max(3, min(8, len(actions) // 2 + 1))):
        seen: dict[tuple[str, ...], int] = {}
        for start in range(0, len(actions) - window + 1):
            key = actions[start : start + window]
            if key in seen:
                repeated_subpaths.append((seen[key], start))
            else:
                seen[key] = start

    blocked_meta = tuple(bool(item) for item in getattr(saved_trace, "blocked_flags", ()))
    if blocked_meta:
        start = None
        for idx, flag in enumerate(blocked_meta):
            if flag and start is None:
                start = idx
            if not flag and start is not None:
                if idx - start >= 2:
                    stall_windows.append((start, idx))
                start = None
        if start is not None and len(blocked_meta) - start >= 2:
            stall_windows.append((start, len(blocked_meta)))

    for prefix_len in range(1, max(1, len(actions) // 2 + 1)):
        if actions[:prefix_len] == actions[prefix_len : 2 * prefix_len]:
            duplicate_prefix_lengths.append(prefix_len)

    return {
        "immediate_reversal_indices": immediate_reversal_indices,
        "oscillation_windows": oscillation_windows,
        "duplicate_prefix_lengths": duplicate_prefix_lengths,
        "repeated_subpaths": repeated_subpaths,
        "stall_windows": stall_windows,
    }


def propose_shorter_trace_candidates(
    baseline_trace,
    redundancy_analysis: dict[str, Any],
) -> tuple[tuple[str, ...], ...]:
    baseline = tuple(str(item) for item in getattr(baseline_trace, "action_trace", ()))
    candidates: list[tuple[str, ...]] = []

    reversals = set(int(item) for item in redundancy_analysis.get("immediate_reversal_indices", ()))
    if reversals:
        pruned = [action for idx, action in enumerate(baseline) if idx not in reversals and idx - 1 not in reversals]
        if 0 < len(pruned) < len(baseline):
            candidates.append(tuple(pruned))

    for prefix_len in tuple(int(item) for item in redundancy_analysis.get("duplicate_prefix_lengths", ())):
        if 0 < prefix_len < len(baseline):
            trimmed = baseline[prefix_len:]
            if 0 < len(trimmed) < len(baseline):
                candidates.append(trimmed)

    for start, end in tuple(redundancy_analysis.get("oscillation_windows", ())):
        start_i = int(start)
        end_i = int(end)
        if 0 <= start_i < end_i <= len(baseline):
            collapsed = baseline[:start_i] + baseline[end_i:]
            if 0 < len(collapsed) < len(baseline):
                candidates.append(collapsed)

    for start, end in tuple(redundancy_analysis.get("stall_windows", ())):
        start_i = int(start)
        end_i = int(end)
        if 0 <= start_i < end_i <= len(baseline):
            collapsed = baseline[:start_i] + baseline[end_i:]
            if 0 < len(collapsed) < len(baseline):
                candidates.append(collapsed)

    dedup = sorted(set(candidates), key=lambda item: (len(item), item))
    return tuple(dedup)


def score_trace_redundancy(saved_trace) -> dict[str, Any]:
    analysis = analyze_trace_for_redundancy(saved_trace)
    counts = {
        "immediate_reversals": len(analysis.get("immediate_reversal_indices", ())),
        "oscillation_windows": len(analysis.get("oscillation_windows", ())),
        "duplicate_prefixes": len(analysis.get("duplicate_prefix_lengths", ())),
        "repeated_subpaths": len(analysis.get("repeated_subpaths", ())),
        "stall_windows": len(analysis.get("stall_windows", ())),
    }
    score = (
        counts["immediate_reversals"] * 1.0
        + counts["oscillation_windows"] * 1.5
        + counts["duplicate_prefixes"] * 2.0
        + counts["repeated_subpaths"] * 0.5
        + counts["stall_windows"] * 1.5
    )
    suggested = []
    for start, end in tuple(analysis.get("oscillation_windows", ())) + tuple(analysis.get("stall_windows", ())):
        suggested.append((int(start), int(end)))
    return {
        "redundancy_score": float(score),
        "counts": counts,
        "suggested_removable_windows": sorted(set(suggested)),
    }
