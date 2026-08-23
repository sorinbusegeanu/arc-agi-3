from __future__ import annotations

"""v8.57 bounded multi-state click exploration and action-space classification.

v8.48 deliberately bounded ACTION6 exploration, but its observable click sweep kept
one rendered color class and treated each coordinate as tried after one click.  That
is sufficient for many binary/erase click games but cannot systematically discover
three-state mechanics where a productive cell must be clicked twice.

This layer preserves the v8.48 public authorities and exact-coordinate action codec.
It only changes the private scan policy:

* expose every observable game-cell centre, independent of rendered color;
* sweep every coordinate once before any repeat;
* allow exactly one second click for coordinates whose first click produced an
  observable structural change or level progress;
* reset the bounded repeat budget at a real reset/level boundary;
* classify click+movement action spaces as click-only in reporting only after
  repeated observed evidence shows all non-click actions are non-productive.
"""

from v8.learning_blockers_v055 import pack_action_choice


_INSTALLED = False
_BASE_EXACT_CLICK_PAGES = None
_BASE_PREPARE_STEP = None
_BASE_FORCED_ACTION = None
_BASE_SPACE_TYPE = None

_MIN_NONCLICK_NOOP_EVIDENCE = 8


def _all_cell_click_tokens(env, grid) -> tuple[int, ...]:
    from v8 import click_exploration_v848 as click

    rows = click._canonical_click_cells(env, grid)
    if not rows:
        return ()
    # Coordinates, not rendered colors, are the intervention identity.  Keeping
    # all visible cells prevents a persistent color choice from hiding valid
    # targets after ACTION6 changes a cell's state/color.
    return tuple(
        pack_action_choice(6, int(x), int(y))
        for _color, x, y in rows
    )


def _exact_click_pages_v857(env, grid) -> tuple[tuple[int, ...], ...]:
    from v8 import click_exploration_v848 as click

    canonical = _all_cell_click_tokens(env, grid)
    if canonical:
        size = click._coverage_page_size()
        return tuple(
            tuple(canonical[start : start + size])
            for start in range(0, len(canonical), size)
        )
    return tuple(
        click.exact_click_coverage_page(grid, page)
        for page in range(click._coverage_page_count(grid))
    )


def _sync_repeat_state(sampler, env) -> bool:
    stamp = (
        int(getattr(env, "reset_count", 0)),
        int(getattr(env, "last_levels_completed", 0)),
    )
    changed = getattr(sampler, "_v857_repeat_stamp", None) != stamp
    if changed:
        sampler._v857_repeat_stamp = stamp
        sampler._v857_repeat_tried = set()
        sampler._v857_repeat_available = ()
    elif not hasattr(sampler, "_v857_repeat_tried"):
        sampler._v857_repeat_tried = set()
        sampler._v857_repeat_available = ()
    return changed


def _repeat_candidates(sampler, env) -> tuple[int, ...]:
    from v8 import click_exploration_v848 as click

    tokens = _all_cell_click_tokens(env, env._last_grid)
    if not tokens:
        return ()

    first_pass_tried = getattr(sampler, "_v848_scan_tried", set())
    if any(int(token) not in first_pass_tried for token in tokens):
        return ()

    productive = {
        int(value)
        for value in getattr(env, "_v848_replayable_clicks", ())
    }
    repeated = getattr(sampler, "_v857_repeat_tried", set())
    return tuple(
        int(token)
        for token in tokens
        if int(token) in productive
        and int(token) not in repeated
        and click._valid_exact_click(int(token), env._last_grid)
    )


def _sampler_prepare_step_v857(self, env) -> bool:
    from v8 import click_exploration_v848 as click

    changed = _sync_repeat_state(self, env)
    if changed:
        # Let v8.48 initialize its first-pass scan state for the new episode/level.
        return bool(_BASE_PREPARE_STEP(self, env))

    if not click._sampler_explicit_control_v848(self):
        tokens = _all_cell_click_tokens(env, env._last_grid)
        tried = getattr(self, "_v848_scan_tried", set())
        remaining = tuple(int(token) for token in tokens if int(token) not in tried)
        if remaining:
            self.pending_sequence = None
            self.base.pending_reset = None
            self._v848_scan_available = remaining
            self._v857_repeat_available = ()
            return False

        repeat = _repeat_candidates(self, env)
        if repeat:
            self.pending_sequence = None
            self.base.pending_reset = None
            self._v848_scan_available = ()
            self._v857_repeat_available = repeat
            return False

    self._v857_repeat_available = ()
    return bool(_BASE_PREPARE_STEP(self, env))


def _sampler_forced_action_v857(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    from v8 import click_exploration_v848 as click
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    if not click._sampler_explicit_control_v848(self):
        available = {int(value) for value in actions}

        # First pass: all cell centres once, independent of color.
        first = tuple(
            int(value)
            for value in getattr(self, "_v848_scan_available", ())
            if int(value) in available
        )
        repeat = tuple(
            int(value)
            for value in getattr(self, "_v857_repeat_available", ())
            if int(value) in available
        )
        candidates = first if first else repeat
        if candidates:
            action = min(candidates)
            if first:
                self._v848_scan_tried.add(action)
                source = "CLICK_SCAN"
            else:
                self._v857_repeat_tried.add(action)
                source = "CLICK_SCAN_REPEAT"
            self.base.current = sampling.Intervention(
                "CLICK_SCAN",
                (int(level), int(context)),
                action,
                tuple(history),
            )
            portfolio._set_mode("NOVELTY")
            portfolio._set_source(context, source, (action,))
            return action

    return _BASE_FORCED_ACTION(
        self,
        level=int(level),
        context=int(context),
        actions=tuple(actions),
        history=tuple(history),
    )


def _space_type_v857(row: dict[str, object] | None) -> str:
    if row is None:
        return "unknown"
    native = row.get("native_types", set())
    if not isinstance(native, set) or not native:
        return "unknown"
    has_click = 6 in native
    nonclick = {int(value) for value in native if int(value) != 6}
    if not has_click:
        return "movement"
    if not nonclick:
        return "click"

    # Native availability alone cannot distinguish a true mixed game from an ARC
    # game that exposes A1-A4 but implements them as no-ops.  Reclassify only after
    # enough actual non-click executions show no structural or level effect.
    movement_attempts = max(0, int(row.get("movement_actions_executed", 0)))
    evidence_required = max(_MIN_NONCLICK_NOOP_EVIDENCE, 2 * len(nonclick))
    if (
        movement_attempts >= evidence_required
        and int(row.get("movement_productive", 0)) == 0
        and int(row.get("movement_level_advances", 0)) == 0
    ):
        return "click"
    return "mixed"


def install_click_state_learning_v857() -> None:
    global _INSTALLED, _BASE_EXACT_CLICK_PAGES, _BASE_PREPARE_STEP
    global _BASE_FORCED_ACTION, _BASE_SPACE_TYPE
    if _INSTALLED:
        return

    from v8 import action_learning_report_v849 as report
    from v8 import click_exploration_v848 as click
    from v8 import sampling_evidence_frontier_v847 as frontier
    from v8 import sampling_persistence_v832 as persistence

    # v8.48's environment wrappers resolve _exact_click_pages dynamically.
    _BASE_EXACT_CLICK_PAGES = click._exact_click_pages
    click._exact_click_pages = _exact_click_pages_v857

    # Compose beneath the stable public sampler authorities, exactly as v8.48 does.
    _BASE_PREPARE_STEP = frontier._BASE_PREPARE_STEP
    frontier._BASE_PREPARE_STEP = _sampler_prepare_step_v857
    _BASE_FORCED_ACTION = persistence._BASE_FORCED_ACTION
    persistence._BASE_FORCED_ACTION = _sampler_forced_action_v857

    # Reporting-only behavioral classification.  Sampling remains environment
    # neutral and click-capable logic still keys off observed ACTION6 availability.
    _BASE_SPACE_TYPE = report._space_type
    report._space_type = _space_type_v857
    _INSTALLED = True
