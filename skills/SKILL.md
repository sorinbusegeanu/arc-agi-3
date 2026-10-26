# Choose actions

Use the available game images, legal actions, recent outcomes, and read-only memory to choose up to the requested number of actions.

Use game memory for established mechanics and strategies. Use global memory as exploration guidance. Follow current observations when they conflict with memory. Do not create or update memory.

Use this internal reasoning procedure before acting:

1. **Observe:** Identify the player, visually distinct objects, relevant changes between labeled screens, and the current game state.
2. **Hypothesize:** List plausible objectives, object mechanics, and completion conditions without treating appearance or proximity as proof.
3. **Evaluate:** Compare each hypothesis with the observed transitions and scoped game memory. Keep supported observations separate from explanations; reject hypotheses contradicted by transitions.
4. **Experiment:** Prefer an action that distinguishes competing hypotheses or advances a demonstrated solution. Prioritize learning about mechanics and completion conditions when the solution is uncertain.
5. **Act:** Return legal actions with a concise assessment, rationale, and first-action prediction using the required structured JSON contract.

Track two kinds of progress separately: progress toward completing the level and progress toward understanding it. Prefer informative experiments over movement through already explored states. Use one action when testing an uncertain interaction; use longer plans only for demonstrated, predictable behavior. Keep detailed chain-of-thought internal; provide only the brief assessment and rationale fields required by the JSON contract.

Prefer a demonstrated strategy when its stated conditions match the current state. Treat recorded positions as historical unless the condition matches; do not turn an unchanged frame into a claim about walls or unreachable areas. Avoid repeating an action that had no visible effect from the same state, and avoid cycling through observed states. For uncertain behavior, choose one action and inspect the result. Use longer plans only for behavior already demonstrated to work.

Before choosing an action, inspect the explicitly marked CURRENT SCREEN. Locate the player and relevant targets in that image, then compare with the preceding labeled screen to confirm what moved and in which direction. Treat coordinates, grid size, adjacency, overlap, and blockage as uncertain unless the images support them; do not infer a grid by measuring a resized image. If an action causes no visible change, do not repeat it from the same state. Choose a different, visually supported experiment and verify its result before drawing a rule.

When context provides `visible_regions_current_screen` or `observed_pixel_region_motion`, use those native-frame pixel-class bounds and measured shifts to ground location comparisons. They identify visual regions only, not semantic objects or grid cells; match them to the labeled images before reasoning about object identity or game rules. The ineffective-action list covers the active level's full observed history, even when the corresponding screenshot is outside the recent image window.

Apply object roles, goal identity, and completion rules only within their stated level or state scope. A rule confirmed on earlier levels remains uncertain on a new level unless memory says it was verified there too. If the apparent target is blocked from distinct approaches, test another visible candidate or condition with one action instead of repeating the same approach. During stagnation, the harness may limit the next model plan to one action; use that action as a deliberate experiment and inspect its result.

When recent outcomes show repeated states or ineffective actions, change the experiment instead of cycling. `RESET` restarts the whole game and is available without a reset count limit when supported by the environment. If the level is stuck and no useful unexplored action remains, choose `RESET` alone with a short justification; verified routes may replay completed levels afterward. The current screen and observed outcomes override uncertain memory hypotheses.

Use these exact lowercase names for movement: `up`, `down`, `left`, `right`. Use other legal action names exactly as listed. Match each action's required arguments.

Return only the required structured JSON object, with concise evidence-based state labels and rationale. `actions` is the only action list; do not include additional action directives elsewhere. Return no more actions than requested. If you choose `RESET`, it must be the only action and include its required justification in `args`.
