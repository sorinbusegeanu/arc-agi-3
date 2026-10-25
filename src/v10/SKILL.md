# Action selection

Use the current and previous game images, legal actions, recent outcomes, and supplied memory to choose a short action plan. The request states the maximum plan length. Memory is read-only; do not create or update memories.

The four standard movement actions are named `up`, `down`, `left`, and `right`. Use these exact lowercase words in your response. Other legal actions keep their listed names. The harness converts movement words to the environment's action IDs.

Return exactly one JSON object with an `actions` array. Each item must contain only:

- `action`: a legal movement name (`up`, `down`, `left`, `right`) or another legal action name from the current observation.
- `args`: arguments valid for that action.

Return no more actions than the requested plan horizon. If `RESET` is listed as a legal action and a controlled restart is justified, return it alone with its required justification in `args`.

Do not include analysis, explanations, goals, memory updates, predicted outcomes, stop conditions, or other fields. Do not use Markdown fences or prose outside the JSON object.

Example:

```json
{"actions":[{"action":"left","args":{}},{"action":"up","args":{}}]}
```
