# Bootstrap Explorer (LLM Stack Agentic)

## Quick Start

Terminal render:

```bash
python -m llm_stack_agentic.lsa_entrypoint \
  --agent bootstrap_explorer \
  --adapter-config src/llm_stack_agentic/adapter_config_default.json \
  --adapter-args '{"render_mode":"terminal"}' \
  --game-id <game> --seed 0
```

No render:

```bash
python -m llm_stack_agentic.lsa_entrypoint \
  --agent bootstrap_explorer \
  --adapter-config src/llm_stack_agentic/adapter_config_default.json \
  --adapter-args '{"render_mode":null}' \
  --game-id <game> --seed 0
```

Notes:
- Default adapter is offline and uses `ENVIRONMENTS_DIR` (defaults to `/home/zodrak/zod/environment_files`).
- Outputs go to stdout unless you pass `--outdir`.
