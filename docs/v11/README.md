# Hydra v11

V11 is an isolated fork of the v10.0.29 live ARC harness. It adds visual click and mixed click/movement support while leaving the v10 implementation and its default state untouched. The detailed feature contract and implementation gates are in [`DESIGN.md`](DESIGN.md).

## Run

Use `PYTHONPATH=src` plus the ARC environment paths used by this repository:

```sh
PYTHONPATH=src:other_repos/arc-agi:other_repos/ARCEngine \
  /home/zodrak/cuda13.x/bin/python -m v11.cli live \
  --game-id ez01 --run-dir runs/v11/ez01-run
```

The v11 CLI is also available through `src/start_live_v11.sh`. Its model options match v10, including opt-in `--vllm`, `--model`, and `--base-url`.

## Isolated instructions and memory

V11 reads its instructions from `skills/v11/SKILL.md` and stores game/global memory and verified routes under `skills/v11/state/` by default. It discovers prior successful runs only under `runs/v11/`. V10 memory and routes are not imported automatically. Set `V11_STATE_DIR` or pass `--state-dir` to choose another state root; choosing a v10 state location explicitly opts into shared state.

The v11 runtime reports version `11.0.0`. Data records whose structure is unchanged may retain their existing `v10.*` schema identifiers for compatibility; the manifest and metrics identify the producing runtime as `v11_version`.

## Tests

Run the v11-specific suite with:

```sh
PYTHONPATH=src /home/zodrak/cuda13.x/bin/python -m pytest -q tests/test_v11_*.py
```

Run the frozen v10 suite separately with `tests/test_v10_*.py` to check that v11 work did not affect v10.
