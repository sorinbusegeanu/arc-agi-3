from __future__ import annotations

"""v8.57 bounded multi-state click exploration and action-space classification.

v8.48 deliberately bounded ACTION6 exploration, but its observable click sweep kept
one rendered color class and treated each coordinate as tried after one click. That
is sufficient for many binary/erase click games but cannot systematically discover
three-state mechanics where a productive cell must be clicked twice.

This layer keeps the installed v8.48 sampler authorities unchanged. It composes
beneath their private delegates and adds two bounded phases after v8.48's original
one-color sweep:

* visit every remaining observable game-cell centre, independent of rendered color;
* click each productive coordinate at most one additional time per reset/level.

It also reclassifies click+movement action spaces as click-only in reporting only
after repeated observed evidence shows all non-click actions are non-productive.
"""

from v8.learning_blockers_v055 import pack_action_choice


_INSTALLED = False
_BASE_EXACT_CLICK_PAGES = None
_LOWER_PREPARE_STEP = None
_LOWER_FORCED_ACTION = None
_BASE_SPACE_TYPE = None

_MIN_NONCLICK_NOOP_EVIDENCE = 8


def _all_cell_click_tokens(env, grid) -> tuple[int, ...]:
    from v8 import click_exploration_v848 as click

    rows = click._canonical_click_cells(env, grid)
    if not rows:
        return ()
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


def _sync_state(sampler, env) -> None:
    stamp = (
        int(getattr(env, "reset_count", 0)),
        int(getattr(env, "last_levels_completed", 0)),
    )
    if getattr(sampler, "_v857_scan_stamp", None) != stamp:
        sampler._v857_scan_stamp = stamp
        sampler._v857_additional_available = ()
        sampler._v857_repeat_available = ()
        sampler._v857_repeat_tried = set()
        return
    if not hasattr(sampler, "_v857_repeat_tried"):
        sampler._v857_repeat_tried = set()
    if not hasattr(sampler, "_v857_additional_available"):
        sampler._v857_additional_available = ()
    if not hasattr(sampler, "_v857_repeat_available"):
        sampler._v857_repeat_available = ()


def _repeat_candidates(sampler, env, tokens: tuple[int, ...]) -> tuple[int, ...]:
    from v8 import click_exploration_v848 as click

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


def _sampler_prepare_fallback_v857(self, env) -> bool:
    """Run only after v8.48 has exhausted its original selected-color sweep."""
    from v8 import click_exploration_v848 as click

    _sync_state(self, env)
    self._v857_additional_available = ()
    self._v857_repeat_available = ()

    if not click._sampler_explicit_control_v848(self):
        tokens = _all_cell_click_tokens(env, env._last_grid)
        tried = getattr(self, "_v848_scan_tried", set())
        remaining = tuple(int(token) for token in tokens if int(token) not in tried)
        if remaining:
            self.pending_sequence = None
            self.base.pending_reset = None
            self._v857_additional_available = remaining
            return False

        repeat = _repeat_candidates(self, env, tokens)
        if repeat:
            self.pending_sequence = None
            self.base.pending_reset = None
            self._v857_repeat_available = repeat
            return False

    return bool(_LOWER_PREPARE_STEP(self, env))


def _sampler_forced_fallback_v857(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
) -> int | None:
    """Supply v8.57 phases beneath v8.48's installed forced-action wrapper."""
    from v8 import click_exploration_v848 as click
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    if not click._sampler_explicit_control_v848(self):
        available = {int(value) for value in actions}
        additional = tuple(
            int(value)
            for value in getattr(self, "_v857_additional_available", ())
            if int(value) in available
        )
        repeat = tuple(
            int(value)
            for value in getattr(self, "_v857_repeat_available", ())
            if int(value) in available
        )
        candidates = additional if additional else repeat
        if candidates:
            action = min(candidates)
            if additional:
                self._v848_scan_tried.add(action)
                source = "CLICK_SCAN_ALL_STATES"
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

    return _LOWER_FORCED_ACTION(
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
    global _INSTALLED, _BASE_EXACT_CLICK_PAGES
    global _LOWER_PREPARE_STEP, _LOWER_FORCED_ACTION, _BASE_SPACE_TYPE
    if _INSTALLED:
        return

    from v8 import action_learning_report_v849 as report
    from v8 import click_exploration_v848 as click

    # Environment availability uses all observable cell states. v8.48 still owns
    # ArcGridEnvironment.available_actions and its public/private wrapper identity.
    _BASE_EXACT_CLICK_PAGES = click._exact_click_pages
    click._exact_click_pages = _exact_click_pages_v857

    # Compose *inside* v8.48. The stable delegates installed into v8.32/v8.47 remain
    # exactly v8.48 functions, satisfying the historical runtime authority contract.
    _LOWER_PREPARE_STEP = click._BASE_SAMPLER_PREPARE_STEP
    click._BASE_SAMPLER_PREPARE_STEP = _sampler_prepare_fallback_v857
    _LOWER_FORCED_ACTION = click._BASE_SAMPLER_FORCED_ACTION
    click._BASE_SAMPLER_FORCED_ACTION = _sampler_forced_fallback_v857

    _BASE_SPACE_TYPE = report._space_type
    report._space_type = _space_type_v857
    _INSTALLED = True
