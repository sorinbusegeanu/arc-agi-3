from __future__ import annotations

"""v8.31 adaptive sampling portfolio with bounded sequence exploration.

Cold DISCOVERY is no longer a single-action anti-repeat policy.  The portfolio
reuses v8.21 reset/replay decision points and mixes six bounded strategies:
sequence search, novelty search, progress-prefix expansion, local transfer probes,
exact-context memory and an unconditional random exploration floor.
"""

import itertools
import threading
from collections import deque
from dataclasses import dataclass


_INSTALLED = False
_BASE_PLAN_CHAIN = None
_BASE_SAMPLER_FOR = None
_BASE_DISCOVERY_ACTOR = None
_PORTFOLIO_STATE = threading.local()

_MAX_SEQUENCE_DEPTH = 4
_MAX_SEQUENCE_BRANCH = 6
_MAX_SEQUENCE_CANDIDATES = 512

# Twenty deterministic slots make the proportions exact without adding another
# random source.  RANDOM is 2/20 in both phases: memory can never suppress the
# final ten percent of exploratory decisions.
_COLD_MODES = (
    "SEQUENCE", "NOVELTY", "SEQUENCE", "PROGRESS", "RANDOM",
    "SEQUENCE", "TRANSFER", "NOVELTY", "SEQUENCE", "MEMORY",
    "PROGRESS", "SEQUENCE", "RANDOM", "NOVELTY", "SEQUENCE",
    "TRANSFER", "PROGRESS", "SEQUENCE", "NOVELTY", "MEMORY",
)
_WARM_MODES = (
    "PROGRESS", "MEMORY", "SEQUENCE", "NOVELTY", "RANDOM",
    "PROGRESS", "MEMORY", "PROGRESS", "SEQUENCE", "MEMORY",
    "PROGRESS", "NOVELTY", "RANDOM", "MEMORY", "PROGRESS",
    "SEQUENCE", "MEMORY", "PROGRESS", "MEMORY", "PROGRESS",
)


@dataclass(slots=True)
class SequenceFrontier:
    anchor: tuple[int, ...]
    actions: tuple[int, ...]
    candidates: tuple[tuple[int, ...], ...]
    next_index: int = 0


def _build_sequences(action_ids) -> tuple[tuple[int, ...], ...]:
    actions = tuple(sorted({int(value) for value in action_ids}))[:_MAX_SEQUENCE_BRANCH]
    if not actions:
        return ()
    rows: list[tuple[int, ...]] = []
    for depth in range(1, _MAX_SEQUENCE_DEPTH + 1):
        for candidate in itertools.product(actions, repeat=depth):
            rows.append(tuple(int(value) for value in candidate))
            if len(rows) >= _MAX_SEQUENCE_CANDIDATES:
                return tuple(rows)
    return tuple(rows)


def _set_mode(mode: str | None) -> None:
    if mode is None:
        try:
            delattr(_PORTFOLIO_STATE, "mode")
        except AttributeError:
            pass
    else:
        _PORTFOLIO_STATE.mode = str(mode)


def _set_source(context: int, source: str, actions=()) -> None:
    from v8 import sampling_progress_control_v829 as v829

    v829._set_selection(int(context), str(source), tuple(int(value) for value in actions))


class PortfolioSampler:
    """Interaction-control portfolio layered over the v8.21 decision sampler."""

    def __init__(self, game_id: str, *, seed: int = 0, max_points: int = 256) -> None:
        from v8 import decision_point_sampling_v821 as sampling

        self.game_id = str(game_id)
        self.base = sampling.DecisionPointSampler(
            self.game_id,
            seed=int(seed),
            max_points=int(max_points),
        )
        self.sequence_frontiers: dict[tuple[int, int], SequenceFrontier] = {}
        self.pending_sequence: tuple[tuple[int, ...], tuple[int, int], tuple[int, ...]] | None = None
        self.active_sequence: deque[int] = deque()
        self.active_sequence_full: tuple[int, ...] = ()
        self.active_point: tuple[int, int] | None = None
        self.active_anchor: tuple[int, ...] = ()
        self.decision_count = 0
        self.saw_progress = False
        self.mode_counts: dict[str, int] = {}

    def begin_lease(self, seed: int) -> None:
        self.base.begin_lease(int(seed))
        self.pending_sequence = None
        self.active_sequence.clear()
        self.active_sequence_full = ()
        self.active_point = None
        self.active_anchor = ()
        _set_mode(None)

    def _choose_mode(self) -> str:
        cycle = _WARM_MODES if self.saw_progress else _COLD_MODES
        mode = str(cycle[self.decision_count % len(cycle)])
        self.decision_count += 1
        self.mode_counts[mode] = int(self.mode_counts.get(mode, 0)) + 1
        _set_mode(mode)
        return mode

    def _frontier(
        self,
        *,
        level: int,
        context: int,
        actions: tuple[int, ...],
        history: tuple[int, ...],
    ) -> SequenceFrontier:
        key = (int(level), int(context))
        available = tuple(sorted({int(value) for value in actions}))[:_MAX_SEQUENCE_BRANCH]
        row = self.sequence_frontiers.get(key)
        if row is None:
            row = SequenceFrontier(
                tuple(history),
                available,
                _build_sequences(available),
            )
            self.sequence_frontiers[key] = row
        else:
            if len(history) < len(row.anchor):
                row.anchor = tuple(history)
            if available and available != row.actions:
                merged = tuple(sorted(set(row.actions) | set(available)))[:_MAX_SEQUENCE_BRANCH]
                row.actions = merged
                row.candidates = _build_sequences(merged)
                row.next_index = 0
        self.base.register_point(
            level=int(level),
            context=int(context),
            anchor=tuple(history),
            actions=tuple(actions),
            priority=3,
        )
        return row

    @staticmethod
    def _next_candidate(row: SequenceFrontier) -> tuple[int, ...] | None:
        if row.next_index >= len(row.candidates):
            return None
        candidate = tuple(row.candidates[row.next_index])
        row.next_index += 1
        return candidate

    def _activate_candidate(
        self,
        *,
        point_key: tuple[int, int],
        anchor: tuple[int, ...],
        candidate: tuple[int, ...],
    ) -> None:
        self.active_point = (int(point_key[0]), int(point_key[1]))
        self.active_anchor = tuple(anchor)
        self.active_sequence_full = tuple(candidate)
        self.active_sequence = deque(int(value) for value in candidate)

    def _schedule_next_sequence(self, point_key: tuple[int, int]) -> bool:
        row = self.sequence_frontiers.get((int(point_key[0]), int(point_key[1])))
        if row is None:
            return False
        candidate = self._next_candidate(row)
        if candidate is None:
            return False
        self.pending_sequence = (tuple(row.anchor), point_key, tuple(candidate))
        return True

    def prepare_step(self, env) -> bool:
        if self.pending_sequence is None:
            return bool(self.base.prepare_step(env))
        anchor, target, candidate = self.pending_sequence
        self.pending_sequence = None
        env.reset()
        self.base.replay_actions = deque(int(value) for value in anchor)
        self.base.replay_target = target
        self.base.current = None
        self._activate_candidate(
            point_key=target,
            anchor=anchor,
            candidate=candidate,
        )
        _set_mode("SEQUENCE")
        return True

    def on_external_reset(self) -> None:
        self.base.on_external_reset()
        self.active_sequence.clear()
        self.active_sequence_full = ()
        self.active_point = None
        self.active_anchor = ()
        _set_mode(None)

    def forced_action(
        self,
        *,
        level: int,
        context: int,
        actions: tuple[int, ...],
        history: tuple[int, ...],
    ) -> int | None:
        from v8 import decision_point_sampling_v821 as sampling

        forced = self.base.forced_action(
            level=int(level),
            context=int(context),
            actions=tuple(actions),
            history=tuple(history),
        )
        if forced is not None:
            _set_source(context, "REPLAY", (forced,))
            return int(forced)

        if self.active_sequence:
            action = int(self.active_sequence[0])
            available = {int(value) for value in actions}
            if action not in available:
                point = self.active_point
                self.active_sequence.clear()
                self.active_sequence_full = ()
                if point is not None:
                    self._schedule_next_sequence(point)
                return None
            self.active_sequence.popleft()
            point = self.active_point or (int(level), int(context))
            self.base.current = sampling.Intervention(
                "SEQUENCE",
                point,
                action,
                tuple(history),
            )
            _set_mode("SEQUENCE")
            _set_source(context, "SEQUENCE", (action,))
            return action

        self._choose_mode()
        return None

    def discovery_action(
        self,
        *,
        level: int,
        context: int,
        actions: tuple[int, ...],
        history: tuple[int, ...],
    ) -> int | None:
        from v8 import decision_point_sampling_v821 as sampling

        available = tuple(sorted({int(value) for value in actions}))
        if not available:
            return None
        mode = str(getattr(_PORTFOLIO_STATE, "mode", "PROGRESS"))

        if mode == "SEQUENCE":
            row = self._frontier(
                level=int(level),
                context=int(context),
                actions=available,
                history=tuple(history),
            )
            candidate = self._next_candidate(row)
            if candidate is not None:
                self._activate_candidate(
                    point_key=(int(level), int(context)),
                    anchor=tuple(row.anchor),
                    candidate=candidate,
                )
                action = int(self.active_sequence.popleft())
                self.base.current = sampling.Intervention(
                    "SEQUENCE",
                    (int(level), int(context)),
                    action,
                    tuple(history),
                )
                _set_source(context, "SEQUENCE", (action,))
                return action

        if mode == "RANDOM":
            self.base.register_point(
                level=int(level),
                context=int(context),
                anchor=tuple(history),
                actions=available,
                priority=2,
            )
            action = int(available[self.base.rng.randrange(len(available))])
            self.base.current = sampling.Intervention(
                "RANDOM",
                (int(level), int(context)),
                action,
                tuple(history),
            )
            _set_source(context, "RANDOM", (action,))
            return action

        if mode == "TRANSFER" and self.base.transfer_action in available:
            action = int(self.base.transfer_action)
            self.base.current = sampling.Intervention(
                "TRANSFER",
                (int(level), int(context)),
                action,
                tuple(history),
            )
            _set_source(context, "TRANSFER_PROBE", (action,))
            return action

        action = self.base.discovery_action(
            level=int(level),
            context=int(context),
            actions=available,
            history=tuple(history),
        )
        if action is not None:
            source = "NOVELTY" if mode == "NOVELTY" else "PROGRESS_PREFIX"
            _set_source(context, source, (action,))
            return int(action)

        # Exhausted point coverage must not become a deterministic dead end.
        action = int(available[self.base.rng.randrange(len(available))])
        self.base.current = sampling.Intervention(
            "RANDOM",
            (int(level), int(context)),
            action,
            tuple(history),
        )
        _set_source(context, "RANDOM_FALLBACK", (action,))
        return action

    def observe_transition(self, **kwargs) -> None:
        intervention = self.base.current
        if intervention is None or str(intervention.kind) != "SEQUENCE":
            before_level = int(kwargs.get("before_level", 0))
            after_level = int(kwargs.get("after_level", before_level))
            terminal_state = str(kwargs.get("terminal_state", ""))
            if after_level > before_level or terminal_state == "WIN":
                self.saw_progress = True
            self.base.observe_transition(**kwargs)
            return

        self.base.current = None
        before_level = int(kwargs.get("before_level", 0))
        before_context = int(kwargs.get("before_context", 0))
        after_level = int(kwargs.get("after_level", before_level))
        after_context = int(kwargs.get("after_context", before_context))
        terminal_state = str(kwargs.get("terminal_state", ""))
        level_advanced = bool(kwargs.get("level_advanced", False))
        after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
        history_after = tuple(int(value) for value in kwargs.get("history_after", ()))
        success = bool(level_advanced or terminal_state == "WIN")

        if terminal_state != "GAME_OVER" and after_actions:
            priority = 6 if success else (
                4 if int(after_context) != int(before_context) else 1
            )
            self.base.register_point(
                level=after_level,
                context=after_context,
                anchor=history_after,
                actions=after_actions,
                priority=priority,
            )

        if success:
            self.saw_progress = True
            self.base.transfer_action = int(kwargs.get("action", intervention.action))
            self.base.transfer_from_level = max(
                int(self.base.transfer_from_level),
                before_level,
            )
            self.active_sequence.clear()
            self.active_sequence_full = ()
            self.active_point = None
            self.active_anchor = ()
            _set_mode(None)
            return

        point = self.active_point or intervention.point_key
        if terminal_state != "GAME_OVER" and self.active_sequence:
            # Continue the selected prefix even when the first action was a no-op.
            # This is the key difference from the v8.29 anti-repeat policy.
            return

        self.active_sequence.clear()
        self.active_sequence_full = ()
        self.active_anchor = ()
        self.active_point = None
        if point is not None:
            self._schedule_next_sequence(point)
        _set_mode(None)


def _sampler_for_v831(job) -> PortfolioSampler:
    from v8 import decision_point_sampling_v821 as sampling

    key = (int(job.actor_id), str(job.game_id))
    sampler = sampling._SAMPLERS.get(key)
    if not isinstance(sampler, PortfolioSampler):
        sampler = PortfolioSampler(str(job.game_id), seed=int(job.seed))
        sampling._SAMPLERS[key] = sampler
        if len(sampling._SAMPLERS) > 64:
            for stale in tuple(sampling._SAMPLERS)[:-64]:
                sampling._SAMPLERS.pop(stale, None)
    sampler.begin_lease(int(job.seed))
    return sampler


def _plan_chain_v831(self, context_signature, action_ids, **kwargs):
    """Only the MEMORY portfolio slot may autonomously consume an M7 plan."""
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_progress_control_v829 as v829

    if not sampling._decision_mode_enabled():
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)
    if not str(getattr(v829._CONTROL_STATE, "game_id", "")):
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)
    if str(getattr(_PORTFOLIO_STATE, "mode", "MEMORY")) != "MEMORY":
        return ()
    return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)


def portfolio_telemetry_v831(game_id: str) -> dict[str, object]:
    from v8 import decision_point_sampling_v821 as sampling

    game = str(game_id)
    rows = [
        row
        for (actor_id, owner), row in tuple(sampling._SAMPLERS.items())
        if owner == game and isinstance(row, PortfolioSampler)
    ]
    modes: dict[str, int] = {}
    frontiers = candidates = 0
    for row in rows:
        frontiers += len(row.sequence_frontiers)
        candidates += sum(frontier.next_index for frontier in row.sequence_frontiers.values())
        for mode, count in row.mode_counts.items():
            modes[mode] = int(modes.get(mode, 0)) + int(count)
    return {
        "game_id": game,
        "actors": len(rows),
        "modes": modes,
        "sequence_frontiers": int(frontiers),
        "sequence_candidates_started": int(candidates),
        "random_floor": 0.10,
    }


def install_sampling_portfolio_v831() -> None:
    global _INSTALLED, _BASE_PLAN_CHAIN, _BASE_SAMPLER_FOR, _BASE_DISCOVERY_ACTOR
    if _INSTALLED:
        return

    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_progress_control_v829 as v829

    _BASE_SAMPLER_FOR = sampling._sampler_for
    sampling._sampler_for = _sampler_for_v831

    # v8.28 deliberately called v8.29's wrapper through sampling._BASE_ACTOR_WORKER.
    # Preserve that public identity and actor-local state binding, but change its
    # lower delegate from the old flat actor to the v8.21 reset/replay actor.
    _BASE_DISCOVERY_ACTOR = v829._BASE_DISCOVERY_ACTOR
    v829._BASE_DISCOVERY_ACTOR = sampling._decision_actor_worker

    # Preserve the historical v8.29/v8.24 public planner hook identities.  Insert
    # portfolio authority underneath v8.29 so compatibility callers are unchanged.
    _BASE_PLAN_CHAIN = v829._BASE_PLAN_CHAIN
    v829._BASE_PLAN_CHAIN = _plan_chain_v831

    _INSTALLED = True
