from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .executable_hypothesis_engine_types import ExecutableHypothesisV1, ExecutableProgramV1, TransitionEventV1
from .mechanic_synthesizer_config import MechanicSynthesizerConfig
from .mechanic_synthesizer_types import ActionSemanticsDraft, MechanicSynthesisReport, SynthesisCandidate
from .primitive_program_v1 import default_hypotheses


def synthesize(
    events: List[TransitionEventV1],
    fp_current: Dict[str, Any],
    available_actions_current: List[str],
    existing_hypotheses: List[ExecutableHypothesisV1],
    cfg: Optional[MechanicSynthesizerConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> MechanicSynthesisReport:
    cfg = cfg or MechanicSynthesizerConfig()
    window = events[-cfg.window_N :]
    diagnostics: Dict[str, Any] = {
        "window_used": len(window),
        "triggered": False,
        "trigger_reason": None,
        "candidates_before_prune": 0,
        "candidates_after_prune": 0,
        "ctx": ctx or {},
    }

    triggers = _should_trigger(window, existing_hypotheses, cfg)
    diagnostics.update(triggers)

    drafts = _action_semantics(window)
    candidates = _synthesize_candidates(drafts, available_actions_current, cfg)
    diagnostics["candidates_before_prune"] = len(candidates)

    candidates = _rank_and_prune(candidates, window, cfg)
    diagnostics["candidates_after_prune"] = len(candidates)

    if not candidates:
        fallback = _fallback_candidate()
        candidates = [fallback]
        diagnostics["fallback_used"] = True

    return MechanicSynthesisReport(candidates=candidates, diagnostics=diagnostics)


def _should_trigger(
    events: List[TransitionEventV1],
    hypotheses: List[ExecutableHypothesisV1],
    cfg: MechanicSynthesizerConfig,
) -> Dict[str, Any]:
    if len(events) < 3:
        return {"triggered": True, "trigger_reason": "insufficient_events"}
    non_falsified = [h for h in hypotheses if not h.fit_stats.get("falsified")]
    if not non_falsified:
        return {"triggered": True, "trigger_reason": "no_active_hypotheses"}
    best = max((h.fit_stats.get("avg_likelihood", 0.0) for h in hypotheses), default=0.0)
    if best < cfg.L_min:
        return {"triggered": True, "trigger_reason": "low_fit"}
    if _ambiguity_high(hypotheses, cfg):
        return {"triggered": True, "trigger_reason": "ambiguity_high"}
    unexplained_rate = _unexplained_rate(events, hypotheses)
    if unexplained_rate > cfg.R_max:
        return {"triggered": True, "trigger_reason": "unexplained_signature_rate"}
    return {"triggered": False, "trigger_reason": None}


def _ambiguity_high(hypotheses: List[ExecutableHypothesisV1], cfg: MechanicSynthesizerConfig) -> bool:
    if len(hypotheses) < 2:
        return False
    ranked = sorted(hypotheses, key=lambda h: (-h.confidence, h.hypothesis_id))
    top = ranked[0].confidence
    second = ranked[1].confidence
    return abs(top - second) < cfg.ambiguity_delta


def _unexplained_rate(events: List[TransitionEventV1], hypotheses: List[ExecutableHypothesisV1]) -> float:
    if not events:
        return 0.0
    best = max(hypotheses, key=lambda h: h.confidence, default=None)
    if best is None:
        return 1.0
    unexplained = 0
    for ev in events:
        pred = _predicted_signature(best)
        obs = _dominant_signature(ev.event_signature_histogram)
        if pred is None or obs is None or pred != obs:
            unexplained += 1
    return unexplained / float(len(events))


def _action_semantics(events: List[TransitionEventV1]) -> List[ActionSemanticsDraft]:
    by_action: Dict[str, List[TransitionEventV1]] = {}
    for ev in events:
        by_action.setdefault(ev.action_key, []).append(ev)
    drafts = []
    for action_key, evs in by_action.items():
        dominant_sig = _dominant_signature(_merge_hist([e.event_signature_histogram for e in evs]))
        noop_rate = sum(1 for e in evs if e.delta_metrics.get("changed_cells", 0) == 0) / float(len(evs))
        delta_bin = _bin_changed_cells(int(sum(e.delta_metrics.get("changed_cells", 0) for e in evs) / len(evs)))
        drafts.append(
            ActionSemanticsDraft(
                action_id=str(action_key),
                dominant_signature=dominant_sig,
                noop_rate=noop_rate,
                delta_bin=delta_bin,
                meta_effects={},
            )
        )
    return drafts


def _synthesize_candidates(
    drafts: List[ActionSemanticsDraft],
    available_actions: List[str],
    cfg: MechanicSynthesizerConfig,
) -> List[SynthesisCandidate]:
    candidates: List[SynthesisCandidate] = []
    for draft in drafts:
        action_id = draft.action_id.split("@", 1)[0]
        if available_actions and action_id not in available_actions:
            continue
        for candidate in _template_candidates(draft, cfg):
            candidates.append(candidate)
    if not candidates:
        candidates.append(_fallback_candidate())
    return candidates


def _template_candidates(draft: ActionSemanticsDraft, cfg: MechanicSynthesizerConfig) -> List[SynthesisCandidate]:
    candidates: List[SynthesisCandidate] = []
    sig = draft.dominant_signature
    delta_bin = draft.delta_bin

    if sig in {"translation", "gravity"}:
        candidates.append(_directional_move_candidate(draft))
    if sig in {"paint", "toggle"}:
        candidates.append(_click_apply_candidate(draft, sig))
    if sig in {"spawn", "despawn"}:
        candidates.append(_collect_candidate(draft))
    if sig is None:
        candidates.append(_noop_candidate(draft))

    if len(candidates) < cfg.beam_per_action_family:
        candidates.append(_global_tick_candidate(draft))

    return candidates[: cfg.beam_per_action_family]


def _directional_move_candidate(draft: ActionSemanticsDraft) -> SynthesisCandidate:
    program = ExecutableProgramV1(
        intent="move",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": [draft.delta_bin or "medium"]}],
        meta_effects=[],
    )
    return _wrap_candidate("move.avatar_4dir", "Directional move", program, draft)


def _click_apply_candidate(draft: ActionSemanticsDraft, sig: str) -> SynthesisCandidate:
    program = ExecutableProgramV1(
        intent="click_apply",
        gates=[{"requires_coord": True}],
        effects=[{"event_signatures": [sig], "delta_bins": [draft.delta_bin or "medium"]}],
        meta_effects=[],
    )
    return _wrap_candidate("click.apply", "Click/apply", program, draft)


def _collect_candidate(draft: ActionSemanticsDraft) -> SynthesisCandidate:
    program = ExecutableProgramV1(
        intent="collect",
        gates=[],
        effects=[{"event_signatures": ["despawn"], "delta_bins": [draft.delta_bin or "small"]}],
        meta_effects=[],
    )
    return _wrap_candidate("collect.target_on_contact", "Collect target", program, draft)


def _noop_candidate(draft: ActionSemanticsDraft) -> SynthesisCandidate:
    program = ExecutableProgramV1(
        intent="noop",
        gates=[],
        effects=[{"event_signatures": [], "delta_bins": ["tiny"], "noop": True}],
        meta_effects=[],
    )
    return _wrap_candidate("noop.effect", "No-op", program, draft)


def _global_tick_candidate(draft: ActionSemanticsDraft) -> SynthesisCandidate:
    program = ExecutableProgramV1(
        intent="tick",
        gates=[],
        effects=[{"event_signatures": [draft.dominant_signature or "translation"], "delta_bins": [draft.delta_bin or "small"]}],
        meta_effects=[],
    )
    return _wrap_candidate("global.tick", "Global tick", program, draft)


def _wrap_candidate(
    hypothesis_id: str,
    name: str,
    program: ExecutableProgramV1,
    draft: ActionSemanticsDraft,
) -> SynthesisCandidate:
    hyp = ExecutableHypothesisV1(
        hypothesis_id=_program_hash(hypothesis_id, program, draft),
        name=name,
        description=f"Synthesized from action {draft.action_id} behavior.",
        program_v1=program,
        params={"action_id": draft.action_id},
        confidence=0.0,
        fit_stats={"transitions_scored": 0, "avg_likelihood": 0.0, "falsified": False},
        predictions=[],
    )
    return SynthesisCandidate(hypothesis=hyp, origin="synth_v1", priority_score=0.0, diagnostics={})


def _rank_and_prune(
    candidates: List[SynthesisCandidate],
    events: List[TransitionEventV1],
    cfg: MechanicSynthesizerConfig,
) -> List[SynthesisCandidate]:
    scored: List[SynthesisCandidate] = []
    for cand in candidates:
        score = _priority_score(cand.hypothesis, events, cfg)
        cand.priority_score = score
        scored.append(cand)
    scored.sort(
        key=lambda c: (
            -c.priority_score,
            _complexity(c.hypothesis),
            c.hypothesis.hypothesis_id,
        )
    )
    return scored[: cfg.max_candidates_total]


def _priority_score(hyp: ExecutableHypothesisV1, events: List[TransitionEventV1], cfg: MechanicSynthesizerConfig) -> float:
    if not events:
        return 0.0
    scores = []
    pred_sig = _predicted_signature(hyp)
    pred_bin = _predicted_delta_bin(hyp)
    for ev in events:
        obs_sig = _dominant_signature(ev.event_signature_histogram)
        sig_score = 1.0 if pred_sig and obs_sig == pred_sig else 0.0
        obs_bin = _bin_changed_cells(int(ev.delta_metrics.get("changed_cells", 0)))
        delta_score = 1.0 if pred_bin and obs_bin == pred_bin else 0.0
        scores.append(0.5 * sig_score + 0.5 * delta_score)
    avg = sum(scores) / float(len(scores))
    penalty = cfg.complexity_penalty * _complexity(hyp)
    return max(0.0, avg - penalty)


def _complexity(hyp: ExecutableHypothesisV1) -> int:
    program = hyp.program_v1
    if isinstance(program, dict):
        gates = program.get("gates", [])
        effects = program.get("effects", [])
        meta = program.get("meta_effects", [])
    else:
        gates = program.gates
        effects = program.effects
        meta = program.meta_effects
    return len(gates) + len(effects) + len(meta)


def _predicted_signature(hyp: ExecutableHypothesisV1) -> Optional[str]:
    program = hyp.program_v1
    effects = program.get("effects") if isinstance(program, dict) else program.effects
    if not effects:
        return None
    first = effects[0]
    return (first.get("event_signatures") or [None])[0]


def _predicted_delta_bin(hyp: ExecutableHypothesisV1) -> Optional[str]:
    program = hyp.program_v1
    effects = program.get("effects") if isinstance(program, dict) else program.effects
    if not effects:
        return None
    first = effects[0]
    return (first.get("delta_bins") or [None])[0]


def _dominant_signature(hist: Dict[str, int]) -> Optional[str]:
    if not hist:
        return None
    return max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _merge_hist(hists: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for hist in hists:
        for key, val in hist.items():
            merged[key] = merged.get(key, 0) + int(val)
    return merged


def _bin_changed_cells(changed_cells: int) -> str:
    if changed_cells <= 0:
        return "tiny"
    if changed_cells <= 4:
        return "small"
    if changed_cells <= 20:
        return "medium"
    return "large"


def _program_hash(base_id: str, program: ExecutableProgramV1, draft: ActionSemanticsDraft) -> str:
    payload = f"{base_id}|{asdict(program)}|{draft.action_id}|{draft.dominant_signature}|{draft.delta_bin}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{base_id}::{digest}"


def _fallback_candidate() -> SynthesisCandidate:
    unknown = [h for h in default_hypotheses() if h.hypothesis_id == "unknown.mechanic"]
    hyp = unknown[0] if unknown else default_hypotheses()[0]
    return SynthesisCandidate(hypothesis=hyp, origin="fallback", priority_score=0.0, diagnostics={})


def asdict_report(report: MechanicSynthesisReport) -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "hypothesis": asdict(cand.hypothesis),
                "origin": cand.origin,
                "priority_score": cand.priority_score,
                "diagnostics": cand.diagnostics,
            }
            for cand in report.candidates
        ],
        "diagnostics": report.diagnostics,
    }
