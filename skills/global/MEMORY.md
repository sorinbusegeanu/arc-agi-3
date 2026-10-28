- **State comparison and completion:** Compare state before and after each legal action; larger changes suggest an interaction worth verifying. Confirm lasting effects over multiple frames. Confirm completion only from an environment-level change or explicit WIN signal, not from adjacency or a single visual effect.

- **No-ops and loops:** A no-op proves only that the tested action had no visible effect from that specific state; do not infer general blockage. Distinguish boundary limits from object properties by testing the same action from a nearby different state. If repeated actions cycle between a small set of states without new interactions, recognize the loop and commit to a single targeted untried action that tests a specific hypothesis.

- **Conditional hypotheses from visual cues:** Treat visual features as hypotheses to test, not facts. An object may be collectible, interactive, a switch, a portal, or a hazard; an apparent barrier may have a gap; a region's coverage may matter. Test each possibility with one informative legal action and compare before/after. If an object produces no state change when interacted with, deprioritize it and redirect to untried objects.

- **Independent verification per level:** Do not assume a prior level's completion condition applies to a new level; verify independently. If free movement produces no progress, redirect to a targeted interaction with a specific object. When all visible objects tested individually without completion, consider that the condition may involve a sequence, combination, or hidden element.

- **Irreversibility caution:** Before committing to an interaction that may alter or remove an object, consider whether it could become unrecoverable. If an object disappears or becomes immovable, test whether reversing or continuing restores it. If the agent's own state changes unexpectedly, test recovery before concluding dead end. Avoid actions that make objects unreachable; if this occurs, reset rather than continue.

- **Partial progress ≠ completion:** If a state change resolves one apparent constraint but the level does not complete, the obstacle was not the only condition. Test whether the changed entity must also satisfy additional conditions beyond the one just resolved before assuming the path forward is clear.

- **Transferable strategy caution:** A strategy that completed one level may produce a terminal failure on the next. Before reusing a prior level's movement pattern, test one step and check for hazards or changed cell properties before committing to the full sequence.

- **Trail and backtracking hazards:** When movement leaves persistent marks on the grid, treat every previously visited cell as potentially hazardous. Plan a path that visits each cell at most once. If a level introduces trail mechanics, verify whether backtracking is fatal before attempting any reversal.

- **Collectible-target levels:** When multiple identical marked cells are visible and a progress indicator fills as the player moves, the goal is likely to visit all marked cells. Identify impassable obstacles first, then plan a path that covers every target without crossing obstacles or revisiting cells.

- **Budget-constrained levels:** When a resource bar depletes with each action, calculate the minimum path length to all targets before committing to a route. If the minimum exceeds the budget, the completion condition may differ from "visit all targets"—test partial visits to identify which subset triggers completion. Track depletion rate across multiple moves to estimate total budget before committing.
