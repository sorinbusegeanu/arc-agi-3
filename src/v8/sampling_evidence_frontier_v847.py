from __future__ import annotations

"""v8.47 persistent evidence-guided prefix exploration.

The v8.31 sequence sampler enumerated an explicit product tree with a fixed depth
and candidate count.  This layer replaces that final runtime behavior with lazy
one-action prefix expansion:

* there is no internal sequence-depth or candidate-count horizon;
* the actor's existing step budget remains the execution bound;
* changed/observable states are merged through a semantic transposition table;
* unchanged visual states retain path-specific nodes so delayed/hidden mechanics
  are not destroyed by an unsafe observation-only merge;
* frontier priority is driven by observed progress, prediction error, novelty,
  future-option change and structural change;
* compact frontier state is persisted under the existing trajectory runtime root
  and restored across normal process/run restarts.

This remains an interaction-control sidecar.  Canonical scientific memory and
transfer validation semantics are unchanged.
"""

import hashlib
import json
import multiprocessing.util as mp_util
import os
from dataclasses import dataclass, field
from pathlib import Path


_INSTALLED = False
_BASE_PREPARE_STEP = None
_BASE_ON_EXTERNAL_RESET = None
_BASE_DISCOVERY_ACTION = None
_BASE_OBSERVE_TRANSITION = None
_FINALIZER_INSTALLED = False

_TRAJECTORY_ROOT_ENV = "ARC_AGI3_V8_TRAJECTORY_ROOT"
_STATE_SCHEMA = 1
_STATE_DIR = "sampling_frontier_v847"


@dataclass(slots=True)
class EvidencePrefixNode:
    node_id: str
    level: int
    context: int
    anchor: tuple[int, ...]
    available_actions: set[int] = field(default_factory=set)
    tried_actions: set[int] = field(default_factory=set)
    expansions: int = 0
    failures: int = 0
    progress: bool = False
    prediction_error: float = 0.0
    novel: bool = False
    future_delta: float = 0.0
    changed_cells: int = 0
    latent: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": str(self.node_id),
            "level": int(self.level),
            "context": int(self.context),
            "anchor": [int(value) for value in self.anchor],
            "available_actions": sorted(int(value) for value in self.available_actions),
            "tried_actions": sorted(int(value) for value in self.tried_actions),
            "expansions": int(self.expansions),
            "failures": int(self.failures),
            "progress": bool(self.progress),
            "prediction_error": float(self.prediction_error),
            "novel": bool(self.novel),
            "future_delta": float(self.future_delta),
            "changed_cells": int(self.changed_cells),
            "latent": bool(self.latent),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "EvidencePrefixNode":
        return cls(
            node_id=str(raw.get("node_id", "")),
            level=int(raw.get("level", 0)),
            context=int(raw.get("context", 0)),
            anchor=tuple(int(value) for value in raw.get("anchor", ())),
            available_actions={int(value) for value in raw.get("available_actions", ())},
            tried_actions={int(value) for value in raw.get("tried_actions", ())},
            expansions=max(0, int(raw.get("expansions", 0))),
            failures=max(0, int(raw.get("failures", 0))),
            progress=bool(raw.get("progress", False)),
            prediction_error=max(0.0, float(raw.get("prediction_error", 0.0))),
            novel=bool(raw.get("novel", False)),
            future_delta=float(raw.get("future_delta", 0.0)),
            changed_cells=max(0, int(raw.get("changed_cells", 0))),
            latent=bool(raw.get("latent", False)),
        )


def _ensure_state_v847(sampler) -> dict[str, EvidencePrefixNode]:
    nodes = getattr(sampler, "_v847_nodes", None)
    if nodes is None:
        nodes = {}
        sampler._v847_nodes = nodes
        sampler._v847_dirty = False
        sampler._v847_active_expansion = None
        sampler._v847_loaded_root = None
        sampler._v847_actor_id = 0
    return nodes


def _anchor_digest(anchor: tuple[int, ...]) -> str:
    digest = hashlib.blake2b(digest_size=8, person=b"v847pf")
    for value in anchor:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b",")
    return digest.hexdigest()


def _canonical_id(level: int, context: int) -> str:
    return f"C:{int(level)}:{int(context)}"


def _latent_id(level: int, context: int, anchor: tuple[int, ...]) -> str:
    return f"L:{int(level)}:{int(context)}:{_anchor_digest(tuple(anchor))}"


def _merge_node_v847(target: EvidencePrefixNode, incoming: EvidencePrefixNode) -> None:
    if len(incoming.anchor) < len(target.anchor):
        target.anchor = tuple(incoming.anchor)
    target.available_actions.update(int(value) for value in incoming.available_actions)
    target.tried_actions.update(int(value) for value in incoming.tried_actions)
    target.expansions = max(int(target.expansions), int(incoming.expansions))
    target.failures = max(int(target.failures), int(incoming.failures))
    target.progress = bool(target.progress or incoming.progress)
    target.prediction_error = max(float(target.prediction_error), float(incoming.prediction_error))
    target.novel = bool(target.novel or incoming.novel)
    target.future_delta = max(float(target.future_delta), float(incoming.future_delta))
    target.changed_cells = max(int(target.changed_cells), int(incoming.changed_cells))
    target.latent = bool(target.latent and incoming.latent)


def _upsert_node_v847(sampler, node: EvidencePrefixNode) -> EvidencePrefixNode:
    nodes = _ensure_state_v847(sampler)
    current = nodes.get(str(node.node_id))
    if current is None:
        nodes[str(node.node_id)] = node
        sampler._v847_dirty = True
        return node
    before = current.to_dict()
    _merge_node_v847(current, node)
    if current.to_dict() != before:
        sampler._v847_dirty = True
    return current


def _register_current_v847(
    sampler,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> EvidencePrefixNode:
    return _upsert_node_v847(
        sampler,
        EvidencePrefixNode(
            node_id=_canonical_id(level, context),
            level=int(level),
            context=int(context),
            anchor=tuple(int(value) for value in history),
            available_actions={int(value) for value in actions},
        ),
    )


def _expandable_actions(node: EvidencePrefixNode) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in node.available_actions - node.tried_actions))


def _select_expandable_action_v847(actions: tuple[int, ...]) -> int:
    """Select one frontier action; later environment layers may refine ordering."""
    if not actions:
        raise ValueError("frontier action selection requires at least one action")
    return int(actions[0])


def _priority_key_v847(node: EvidencePrefixNode):
    """Lexicographic evidence priority; no synthetic scalar reward is introduced."""
    return (
        int(bool(node.progress)),
        float(node.prediction_error),
        int(bool(node.novel)),
        float(node.future_delta),
        int(node.changed_cells),
        -int(node.failures),
        -int(node.expansions),
        -len(node.anchor),
        str(node.node_id),
    )


def _best_expansion_v847(sampler):
    nodes = _ensure_state_v847(sampler)
    candidates = [node for node in nodes.values() if _expandable_actions(node)]
    if not candidates:
        return None
    node = max(candidates, key=_priority_key_v847)
    action = _select_expandable_action_v847(_expandable_actions(node))
    return node, action


def _productive_v847(
    *,
    before_context: int,
    after_context: int,
    changed_cells: int,
    future_delta: float,
) -> bool:
    return bool(
        int(after_context) != int(before_context)
        or int(changed_cells) > 0
        or float(future_delta) != 0.0
    )


def _record_expansion_v847(
    sampler,
    *,
    source_node_id: str,
    action: int,
    before_level: int,
    before_context: int,
    after_level: int,
    after_context: int,
    after_actions: tuple[int, ...],
    history_after: tuple[int, ...],
    changed_cells: int,
    terminal_state: str,
    level_advanced: bool,
    prediction_error: float,
    future_delta: float,
) -> EvidencePrefixNode | None:
    nodes = _ensure_state_v847(sampler)
    source = nodes.get(str(source_node_id))
    if source is None:
        source = _register_current_v847(
            sampler,
            level=int(before_level),
            context=int(before_context),
            actions=(int(action),),
            history=tuple(history_after[:-1]),
        )
    source.available_actions.add(int(action))
    source.tried_actions.add(int(action))
    source.expansions += 1
    if str(terminal_state) == "GAME_OVER":
        source.failures += 1
    sampler._v847_dirty = True

    if str(terminal_state) == "GAME_OVER" or not tuple(after_actions):
        return None

    productive = _productive_v847(
        before_context=int(before_context),
        after_context=int(after_context),
        changed_cells=int(changed_cells),
        future_delta=float(future_delta),
    )
    latent = bool(
        not productive
        and int(after_level) == int(before_level)
        and int(after_context) == int(before_context)
    )
    anchor = tuple(int(value) for value in history_after)
    node_id = (
        _latent_id(after_level, after_context, anchor)
        if latent
        else _canonical_id(after_level, after_context)
    )
    novel = node_id not in nodes
    destination = EvidencePrefixNode(
        node_id=node_id,
        level=int(after_level),
        context=int(after_context),
        anchor=anchor,
        available_actions={int(value) for value in after_actions},
        progress=bool(level_advanced or str(terminal_state) == "WIN"),
        prediction_error=max(0.0, float(prediction_error)),
        novel=bool(novel),
        future_delta=float(future_delta),
        changed_cells=max(0, int(changed_cells)),
        latent=latent,
    )
    return _upsert_node_v847(sampler, destination)


def _state_root_v847() -> Path | None:
    raw = os.environ.get(_TRAJECTORY_ROOT_ENV)
    if raw is None or not str(raw).strip():
        return None
    return Path(raw) / _STATE_DIR


def _game_token(game_id: str) -> str:
    return hashlib.blake2b(
        str(game_id).encode("utf-8"), digest_size=8, person=b"v847game"
    ).hexdigest()


def _state_path_v847(sampler) -> Path | None:
    root = getattr(sampler, "_v847_state_root", None)
    actor_id = int(getattr(sampler, "_v847_actor_id", 0))
    if root is None or actor_id <= 0:
        return None
    return Path(root) / f"{_game_token(sampler.game_id)}-{actor_id}.json"


def _action_learning_metrics_v847(
    nodes: dict[str, EvidencePrefixNode],
) -> dict[str, int]:
    from v8.action_targeting_v810 import native_action_id

    result = {
        "click_frontier_nodes": 0,
        "click_frontier_expandable": 0,
        "suppressed_click_noop_frontiers": 0,
    }
    for node in nodes.values():
        click_available = {
            int(value)
            for value in node.available_actions
            if int(native_action_id(int(value))) == 6
        }
        click_anchor = bool(
            node.anchor and int(native_action_id(int(node.anchor[-1]))) == 6
        )
        if click_available or click_anchor:
            result["click_frontier_nodes"] += 1
        if click_available - node.tried_actions:
            result["click_frontier_expandable"] += 1
        if bool(node.latent) and click_anchor:
            result["suppressed_click_noop_frontiers"] += 1
    return result


def _write_action_learning_metrics_v847(
    path: Path,
    *,
    game_id: str,
    nodes: dict[str, EvidencePrefixNode],
) -> None:
    metrics_path = path.with_suffix(".metrics")
    try:
        if metrics_path.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            return
    except OSError:
        pass
    metrics_payload = {
        "schema": _STATE_SCHEMA,
        "game_id": str(game_id),
        **_action_learning_metrics_v847(nodes),
    }
    metrics_temporary = metrics_path.with_name(
        f".{metrics_path.name}.{os.getpid()}.tmp"
    )
    metrics_temporary.write_text(
        json.dumps(metrics_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(metrics_temporary, metrics_path)


def _save_sampler_state_v847(sampler) -> None:
    nodes = _ensure_state_v847(sampler)
    path = _state_path_v847(sampler)
    if path is None or not bool(getattr(sampler, "_v847_dirty", False)):
        return
    payload = {
        "schema": _STATE_SCHEMA,
        "game_id": str(sampler.game_id),
        "nodes": [nodes[key].to_dict() for key in sorted(nodes)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    _write_action_learning_metrics_v847(
        path,
        game_id=str(sampler.game_id),
        nodes=nodes,
    )
    sampler._v847_dirty = False


def _load_sampler_state_v847(sampler) -> None:
    nodes = _ensure_state_v847(sampler)
    root = getattr(sampler, "_v847_state_root", None)
    if root is None:
        sampler._v847_loaded_root = None
        return
    root = Path(root)
    token = _game_token(sampler.game_id)
    if root.exists():
        for path in sorted(root.glob(f"{token}-*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if int(raw.get("schema", 0)) != _STATE_SCHEMA:
                continue
            if str(raw.get("game_id", "")) != str(sampler.game_id):
                continue
            source_nodes: dict[str, EvidencePrefixNode] = {}
            for item in raw.get("nodes", ()):
                if not isinstance(item, dict):
                    continue
                try:
                    node = EvidencePrefixNode.from_dict(item)
                except (TypeError, ValueError):
                    continue
                if node.node_id:
                    source_nodes[node.node_id] = node
                    _upsert_node_v847(sampler, node)
            try:
                _write_action_learning_metrics_v847(
                    path,
                    game_id=str(sampler.game_id),
                    nodes=source_nodes,
                )
            except OSError:
                # The frontier itself remains usable on read-only or transiently
                # contended roots; reporting falls back to bounded legacy rules.
                pass
    sampler._v847_dirty = False
    sampler._v847_loaded_root = str(root)


def _save_live_frontiers_v847() -> None:
    try:
        from v8 import decision_point_sampling_v821 as sampling

        seen = set()
        for sampler in tuple(sampling._SAMPLERS.values()):
            identity = id(sampler)
            if identity in seen:
                continue
            seen.add(identity)
            if hasattr(sampler, "_v847_nodes"):
                _save_sampler_state_v847(sampler)
    except BaseException:
        return


def _ensure_process_finalizer_v847() -> None:
    global _FINALIZER_INSTALLED
    if _FINALIZER_INSTALLED:
        return
    mp_util.Finalize(None, _save_live_frontiers_v847, exitpriority=20)
    _FINALIZER_INSTALLED = True


def _sampler_for_v847(job):
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    key = (int(job.actor_id), str(job.game_id))
    sampler = sampling._SAMPLERS.get(key)
    if isinstance(sampler, portfolio.PortfolioSampler):
        _save_sampler_state_v847(sampler)
    else:
        sampler = portfolio.PortfolioSampler(str(job.game_id), seed=int(job.seed))
        sampling._SAMPLERS[key] = sampler
        if len(sampling._SAMPLERS) > 64:
            for stale in tuple(sampling._SAMPLERS)[:-64]:
                victim = sampling._SAMPLERS.get(stale)
                if victim is not None and hasattr(victim, "_v847_nodes"):
                    _save_sampler_state_v847(victim)
                sampling._SAMPLERS.pop(stale, None)

    _ensure_state_v847(sampler)
    root = _state_root_v847()
    root_text = None if root is None else str(root)
    if getattr(sampler, "_v847_loaded_root", None) != root_text:
        sampler._v847_nodes.clear()
        sampler._v847_state_root = root
        sampler._v847_actor_id = int(job.actor_id)
        _load_sampler_state_v847(sampler)
    else:
        sampler._v847_state_root = root
        sampler._v847_actor_id = int(job.actor_id)

    sampler.begin_lease(int(job.seed))
    sampler._v847_active_expansion = None
    _ensure_process_finalizer_v847()
    return sampler


def _next_mode_v847(self) -> str:
    from v8 import sampling_portfolio_v831 as portfolio

    cycle = portfolio._WARM_MODES if self.saw_progress else portfolio._COLD_MODES
    return str(cycle[self.decision_count % len(cycle)])


def _consume_sequence_mode_v847(self) -> None:
    from v8 import sampling_portfolio_v831 as portfolio

    self.decision_count += 1
    self.mode_counts["SEQUENCE"] = int(self.mode_counts.get("SEQUENCE", 0)) + 1
    portfolio._set_mode("SEQUENCE")


def _control_busy_v847(self) -> bool:
    return bool(
        self.pending_sequence is not None
        or self.base.pending_reset is not None
        or self.base.replay_actions
        or self.base.replay_target is not None
        or self.base.verification is not None
        or self.active_sequence
        or getattr(self, "_v832_persist_action", None) is not None
        or bool(getattr(self, "_v833_random_rollout", False))
        or bool(getattr(self, "_v833_transfer_rollout", False))
    )


def _prepare_step_v847(self, env) -> bool:
    _ensure_state_v847(self)
    if not _control_busy_v847(self) and _next_mode_v847(self) == "SEQUENCE":
        selected = _best_expansion_v847(self)
        if selected is not None:
            node, action = selected
            _consume_sequence_mode_v847(self)
            self.pending_sequence = (
                tuple(node.anchor),
                (int(node.level), int(node.context)),
                (int(action),),
            )
            self._v847_active_expansion = (str(node.node_id), int(action))
    return bool(_BASE_PREPARE_STEP(self, env))


def _on_external_reset_v847(self) -> None:
    self._v847_active_expansion = None
    return _BASE_ON_EXTERNAL_RESET(self)


def _schedule_next_sequence_v847(self, _point_key) -> bool:
    """v8.32 stall recovery only marks availability; it never chains a fixed tree."""
    return _best_expansion_v847(self) is not None


def _discovery_action_v847(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    mode = str(getattr(portfolio._PORTFOLIO_STATE, "mode", "PROGRESS"))
    if mode != "SEQUENCE":
        return _BASE_DISCOVERY_ACTION(
            self,
            level=int(level),
            context=int(context),
            actions=tuple(actions),
            history=tuple(history),
        )

    available = tuple(sorted({int(value) for value in actions}))
    if not available:
        return None
    node = _register_current_v847(
        self,
        level=int(level),
        context=int(context),
        actions=available,
        history=tuple(history),
    )
    local = _expandable_actions(node)
    if not local:
        # A remote evidence frontier can only be replayed at the next loop's
        # prepare_step boundary.  Avoid falling back into v8.31's product tree.
        portfolio._set_mode("NOVELTY")
        try:
            return _BASE_DISCOVERY_ACTION(
                self,
                level=int(level),
                context=int(context),
                actions=available,
                history=tuple(history),
            )
        finally:
            portfolio._set_mode(None)

    action = _select_expandable_action_v847(local)
    self._v847_active_expansion = (str(node.node_id), action)
    self.active_point = (int(level), int(context))
    self.active_anchor = tuple(history)
    self.active_sequence_full = (action,)
    self.active_sequence.clear()
    self.base.current = sampling.Intervention(
        "SEQUENCE",
        (int(level), int(context)),
        action,
        tuple(history),
    )
    portfolio._set_source(context, "SEQUENCE_FRONTIER", (action,))
    return action


def _observe_sequence_v847(self, intervention, **kwargs) -> None:
    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_portfolio_v831 as portfolio

    active = getattr(self, "_v847_active_expansion", None)
    source_node_id = str(active[0]) if active is not None else _canonical_id(
        int(kwargs.get("before_level", 0)), int(kwargs.get("before_context", 0))
    )
    action = int(kwargs.get("action", intervention.action))
    before_level = int(kwargs.get("before_level", 0))
    before_context = int(kwargs.get("before_context", 0))
    after_level = int(kwargs.get("after_level", before_level))
    after_context = int(kwargs.get("after_context", before_context))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    history_after = tuple(int(value) for value in kwargs.get("history_after", ()))
    changed_cells = int(kwargs.get("changed_cells", 0))
    terminal_state = str(kwargs.get("terminal_state", ""))
    terminal_polarity = int(kwargs.get("terminal_polarity", 0))
    level_advanced = bool(kwargs.get("level_advanced", after_level > before_level))
    prediction_error = float(kwargs.get("prediction_error", 0.0))
    future_delta = float(kwargs.get("future_delta", 0.0))
    success = bool(level_advanced or terminal_state == "WIN")
    productive = _productive_v847(
        before_context=before_context,
        after_context=after_context,
        changed_cells=changed_cells,
        future_delta=future_delta,
    )

    self.base.current = None
    before_points = set(self.base.points)
    after_key = (after_level, after_context)
    novel = after_key not in before_points
    bad = bool(
        terminal_state == "GAME_OVER"
        or (not success and not productive)
    )
    priority = self.base.signal_priority(
        success=success,
        positive=terminal_polarity > 0,
        prediction_error=prediction_error,
        novel=novel,
        future_delta=future_delta,
        bad=bad,
    )
    if terminal_state != "GAME_OVER" and after_actions:
        self.base.register_point(
            level=after_level,
            context=after_context,
            anchor=history_after,
            actions=after_actions,
            priority=priority,
        )

    _record_expansion_v847(
        self,
        source_node_id=source_node_id,
        action=action,
        before_level=before_level,
        before_context=before_context,
        after_level=after_level,
        after_context=after_context,
        after_actions=after_actions,
        history_after=history_after,
        changed_cells=changed_cells,
        terminal_state=terminal_state,
        level_advanced=level_advanced,
        prediction_error=prediction_error,
        future_delta=future_delta,
    )

    self.active_sequence.clear()
    self.active_sequence_full = ()
    self.active_point = None
    self.active_anchor = ()
    self.pending_sequence = None
    self._v847_active_expansion = None

    if success:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        persistence._clear_persistence_v832(self)
    elif terminal_state != "GAME_OVER" and productive and action in after_actions:
        persistence._arm_persistence_v832(self, action, (before_level, before_context))

    portfolio._set_mode(None)


def _observe_transition_v847(self, **kwargs) -> None:
    intervention = self.base.current
    if intervention is not None and str(intervention.kind) == "SEQUENCE":
        _observe_sequence_v847(self, intervention, **kwargs)
        return
    return _BASE_OBSERVE_TRANSITION(self, **kwargs)


def frontier_telemetry_v847(game_id: str) -> dict[str, object]:
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    game = str(game_id)
    samplers = [
        sampler
        for (_actor_id, owner), sampler in tuple(sampling._SAMPLERS.items())
        if owner == game and isinstance(sampler, portfolio.PortfolioSampler)
    ]
    nodes = [
        node
        for sampler in samplers
        for node in _ensure_state_v847(sampler).values()
    ]
    return {
        "game_id": game,
        "actors": len(samplers),
        "frontier_nodes": len(nodes),
        "latent_nodes": sum(1 for node in nodes if node.latent),
        "expandable_nodes": sum(1 for node in nodes if _expandable_actions(node)),
        "max_prefix_length": max((len(node.anchor) for node in nodes), default=0),
        "fixed_depth_limit": None,
        "fixed_candidate_limit": None,
    }


def install_sampling_evidence_frontier_v847() -> None:
    global _INSTALLED
    global _BASE_PREPARE_STEP, _BASE_ON_EXTERNAL_RESET
    global _BASE_DISCOVERY_ACTION, _BASE_OBSERVE_TRANSITION
    if _INSTALLED:
        return

    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    cls = portfolio.PortfolioSampler
    _BASE_PREPARE_STEP = cls.prepare_step
    _BASE_ON_EXTERNAL_RESET = cls.on_external_reset
    _BASE_DISCOVERY_ACTION = cls.discovery_action
    _BASE_OBSERVE_TRANSITION = cls.observe_transition

    sampling._sampler_for = _sampler_for_v847
    cls.prepare_step = _prepare_step_v847
    cls.on_external_reset = _on_external_reset_v847
    cls.discovery_action = _discovery_action_v847
    cls.observe_transition = _observe_transition_v847
    cls._schedule_next_sequence = _schedule_next_sequence_v847
    _INSTALLED = True
