# OSS v1 ARC Engine Adaptor & GPT‑OSS Agent

## 1. Action representation
- Actions are integer IDs defined by `arcengine/enums.py`:
  - UP      → 1
  - DOWN    → 2
  - LEFT    → 3
  - RIGHT   → 4
  - HOLD    → 5
  - CLICK(x,y) → 6 (complex action requiring coordinates)

## 2. Grid representation for GPT‑OSS input
- 4 × 4 ARC board encoded as a **16‑character hex string** (`0–9`, `A–F`), no separators or spaces.
  Example: `"0123ABCD4567EF89"`.
- The prompt will also include:
  - List of available action IDs (e.g., `[1,2,4,6]`).
  - Current state name (`WIN`, `GAME_OVER`, or `NOT_FINISHED`).

## 3. Prompt template
```
You are an ARC game solver.

Board: {hex_grid}
Available actions: {action_list}
State: {state_name}

Choose the next action.  
Output one of:
- A single integer ID (e.g., "1").
- For a click, `CLICK x y` where `x` and `y` are grid coordinates (0‑based).
Respond *only* with that string.
```

## 4. Adaptor (`src/oss_v1/adaptor.py`)
```python
class ArcEngineAdaptor:
    def __init__(self, game_id: str, seed: int = 0) -> None
    def reset(self) -> np.ndarray        # returns current grid as a NumPy array
    def step(self, action: int) -> np.ndarray   # forwards to GameAction.from_id(action)
    def observe(self) -> np.ndarray   # current grid snapshot
    def available_actions(self) -> list[int]
```

## 5. Agent (`src/oss_v1/agent.py`)
- Calls the adaptor to get board state and available actions.
- Formats them into the prompt template above.
- Sends the prompt to GPT‑OSS via Ollama (`http://localhost:11434/api/chat`).
- Parses LLM response:
  - Strip whitespace, uppercase.
  - If starts with `"CLICK"`, split and parse two integers → action ID 6 plus coordinates.
  - Otherwise, interpret as an integer ID (1–6).
- Calls `adaptor.step(action_id)` and loops until terminal state.

## 6. Implementation notes
- Use the existing `ArcGridEnvironment` from `src/v6/environment/arc_adapter.py`; wrap it to expose the minimal interface above.
- LLM calls are synchronous; keep token budget < 200 tokens for the prompt + response.
- Logging: capture each step’s board string, prompt sent, LLM output, and chosen action in a CSV (optional).

## 7. Next question
Please confirm the grid representation and let me know if you’d like to include any extra metadata (e.g., `levels_completed`) or change anything else before we implement.
