# Controller (LLM Stack Agentic)

## Quick Start

Run controller with default adapter (offline):

```bash
python -m llm_stack_agentic.lsa_entrypoint \
  --agent controller \
  --adapter-config src/llm_stack_agentic/adapter_config_default.json \
  --controller-config-json '{"max_actions_total":200,"probe_steps":4,"top_k_pois_per_round":3,"per_poi_step_budget":12,"max_rounds":5}' \
  --game-id <game> --seed 0
```

Notes:
- `env_parallel_mode` must be `sequential` (anything else is a hard error).
- Artifacts are written under `runs/lsa_controller/<game>_<seed>` unless `controller_config.outdir` is set.
