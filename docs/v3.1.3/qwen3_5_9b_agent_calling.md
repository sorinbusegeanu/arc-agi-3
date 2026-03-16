# Qwen 3.5 9B Agent Calling

## Contract

- use OpenAI-compatible Chat Completions
- disable thinking mode
- require strict JSON-only output
- keep calls single-turn
- keep calls gated
- advisory LLM calls must use focused subgraph prompts
- do not dump the full mechanic graph state
- keep one target question per call

## Runtime Guidance

- use low-temperature settings for advisory mechanic reasoning
- do not send prior assistant generations back into the prompt
- do not send raw frames
- do not send long logs
- do not use streaming
- keep prompt budget below active local context limits
- increase timeout for local inference beyond the old 5-second path
- prompt compression is required before increasing model context length

## Focused Prompt Rules

- `hypothesis_for_exit` must center on one exit and its local prerequisite neighborhood
- `resolve_contradiction` must center on one contradicted relation or path
- `suggest_experiment` must center on only the small competing hypothesis set
- include only the focused node ids, edges, paths, contradictions, and exit attempts needed for that question
- export only compact symbolic summaries, not full internal graph rows

## Allowed Use

- mechanic hypothesis generation
- ambiguity resolution
- experiment suggestion

## Disallowed Use

- frame-level perception
- low-level action selection
- free-form explanation mode
- chain-of-thought output
