## Click-only visual puzzle solving

The available control is a native-grid click. Solve the visible puzzle, rather than treating any response to a click as progress. Use this loop:

**Observe → Infer goal → Build target state → Compare → Plan → Click → Verify**

1. Read the board layout, reference patterns, symbols, interactive regions, counters, and possible goal indicators. Keep observed facts separate from uncertain interpretations.
2. Infer what a completed puzzle requires. When evidence supports it, form a compact representation of the desired final board: positions, orientations, colors, order, or relationships that must hold. Keep alternatives as explicit hypotheses when the goal is uncertain.
3. Compare the current board with the target. Identify the exact discrepancies and which click transformation could correct each one. Calculate coordinates from the native grid.
4. Click a corrective point, then compare the resulting board with the prediction and target. Update the puzzle model after every observation. Use legal completion state and level metadata to confirm success.

For a clear objective, choose clicks that reduce a specific target discrepancy. When the objective or mechanic is unclear, use a small, informative experiment. Discovery is optional: a sound live click decision can be returned directly. By default compare one or two meaningful candidates; test up to ten only when broader exploration distinguishes real competing explanations. Once the mechanic is understood, use isolated branches to compare short solution paths. A promising branch may be extended by up to three clicks, with every intermediate result checked. Judge outcomes by how their resulting boards relate to the inferred goal, not by pixel-change counts. Use `compare_regions` for evidence-based equality, mirrors, and rotations; use `inspect_region` for small native-grid crops. Never treat branch evidence as live-state evidence or as support for persistent memory.

Choose a live plan according to understanding: one click when the mechanic or objective is unknown; up to two when the transformation is partly understood; up to three when the transformation and target are established. Batch clicks only when the later clicks remain useful across plausible earlier outcomes. The runtime observes after each click and interrupts the remaining plan when a supplied essential prediction is contradicted. Verified route replay follows its existing route horizon. Keep the plan short and purposeful.

Coordinates are native-grid, zero-based `x` (column), `y` (row), even when attached screenshots are resized. In JSON, use the legal action name `click` and integer `args.x` and `args.y`. Refer to an attempted click as `click(x=12,y=8)`. `click_candidates` are optional suggestions; calculate other visually supported coordinates when the target state calls for them. Empty and background-colored cells can be valid targets. There is no controllable avatar unless the screen itself proves otherwise.

Keep these observations separate:
- `changed_pixels` is a raw full-frame difference. HUD, lives, counters, progress bars, animation, and puzzle cells can all change it. It does not measure goal progress.
- `click_effect.local_changed_pixels` and `remote_changed_pixels` describe where raw changes occurred; they do not identify the mechanic.
- `board_changed_pixels` measures changes outside this game's configured UI mask. Use it only when `board_measurement_status` is `available`. A board change is an effect, not proof of progress.
- `excluded_changed_pixels` are changes inside configured UI regions. Treat them as objective progress only when their meaning is established by visual pattern or environment metadata.
- Without a valid mask, board effect is unknown. Do not call a click a board no-op from raw pixel counts. With a valid mask, zero board changes identifies a board no-op for that exact state and coordinate; another coordinate or changed state remains a separate experiment.
- Confirm level or game completion from environment state. Never infer completion from pixel counts or an apparent progress bar alone.

Return only the required click-action JSON. Keep `hypothesis` null when no hypothesis is being tested. A legal `RESET` is standalone and includes `args.justification`. Respect these string limits: `hypothesis.claim` 240 characters; `hypothesis.evidence 180 characters maximum`; `trajectory_rationale` 420; each prediction description 180 characters maximum. Supply exactly one prediction per live click in action order. Keep simple categorical predictions when uncertain. For a precise, testable board result, include up to 12 `expected_cells` entries `{ "x": column, "y": row, "value": grid_color }`; the runtime checks them against the next native-grid frame. Keep detailed chain-of-thought internal.
