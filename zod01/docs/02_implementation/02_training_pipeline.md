# zod01 Training Implementation (Module 17)

## Implemented Components
- Data collection runner: `zod01/training/collect_data.py`
- Data quality gate: `zod01/training/check_data_quality.py`
- Dataset builder: `zod01/training/dataset_builder.py`
- Ranker trainer: `zod01/training/train_ranker.py`
- Critic trainer: `zod01/training/train_critic.py`
- Mechanic trainer: `zod01/training/train_mechanic.py`
- Ablation evaluator: `zod01/training/eval_ablation.py`
- Runtime model hooks in agent/controller:
  - `--use-ranker`
  - `--use-learned-critic`
  - `--use-mechanic-classifier`

## Logged Training Fields (per-step)
Each step log now includes:
- `episode_id, game_id, seed, variant_id, step_idx`
- `state_hash, raw_obs_hash`
- `available_actions`
- `chosen_action`
- `next_state_hash`
- `delta_tokens, delta_kind`
- `terminal, won, actions_used`
- candidate-level `features`, critic/safety tags and scores (`candidate_debug`)

## End-to-End Commands

### 1) Collect data
```bash
UV_CACHE_DIR=/tmp/uv-cache OPERATION_MODE=offline ENVIRONMENTS_DIR=arc-agi/test_environment_files \
uv run python zod01/training/collect_data.py \
  --games bt11-fd9df0622a1b \
  --seeds 0,1,2,3 \
  --variants baseline \
  --max-actions 80 \
  --run-dir zod01/runs
```

### 2) Check prerequisites
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python zod01/training/check_data_quality.py \
  --summary zod01/runs/<run_id>/summary.json
```

### 3) Build datasets
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python zod01/training/dataset_builder.py \
  --log-dir zod01/runs/<run_id>/logs \
  --out-dir zod01/datasets/<run_id>
```

### 4) Train module models
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python zod01/training/train_ranker.py \
  --dataset zod01/datasets/<run_id>/ranker.jsonl \
  --out zod01/models/ranker.json

UV_CACHE_DIR=/tmp/uv-cache uv run python zod01/training/train_critic.py \
  --dataset zod01/datasets/<run_id>/critic.jsonl \
  --out zod01/models/critic.json

UV_CACHE_DIR=/tmp/uv-cache uv run python zod01/training/train_mechanic.py \
  --dataset zod01/datasets/<run_id>/mechanic.jsonl \
  --out zod01/models/mechanic.json
```

### 5) Run learned ablations
```bash
UV_CACHE_DIR=/tmp/uv-cache OPERATION_MODE=offline ENVIRONMENTS_DIR=arc-agi/test_environment_files \
uv run python zod01/training/eval_ablation.py \
  --games bt11-fd9df0622a1b \
  --seeds 0,1,2 \
  --max-actions 80
```

## Notes
- `train_ranker.py` intentionally fails if there are zero positive labels (no successful episodes).
- Game-level split is applied in ranker training to reduce leakage.
- Learned modules are additive; heuristics remain active as fallback.
