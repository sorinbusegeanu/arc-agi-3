# Action selection

Use the current and previous game images, legal actions, recent outcomes, and supplied memory to choose a short action plan. The request states the maximum plan length. Memory is read-only; do not create or update memories.

Return exactly one JSON object with an `actions` array. Each item must contain only:

- `action`: a legal action ID from the current observation.
- `args`: arguments valid for that action.

Return no more actions than the requested plan horizon. If `RESET` is listed as a legal action and a controlled restart is justified, return it alone with its required justification in `args`.

Do not include analysis, explanations, goals, memory updates, predicted outcomes, stop conditions, or other fields. Do not use Markdown fences or prose outside the JSON object.

Example:

```json
{"actions":[{"action":"ACTION3","args":{}},{"action":"ACTION1","args":{}}]}
```
