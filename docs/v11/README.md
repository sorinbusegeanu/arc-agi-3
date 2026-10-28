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

V11 reads general instructions from `skills/v11/SKILL.md` and composes only the capability fragments for the runtime action set. Game memory, approved profile method memory, candidate proposals, and verified routes live under `skills/v11/state/` by default. Profile memory is isolated by action set; review and discovery can queue candidates, but only an explicit promotion after cross-game and held-out evaluation can change it. The first live run in a state root removes the obsolete `global/MEMORY.md` once and records a migration marker. It discovers prior successful runs only under `runs/v11/`. V10 memory and routes are not imported automatically. Set `V11_STATE_DIR` or pass `--state-dir` to choose another state root.

The v11 runtime reports version `11.1.0`. Data records whose structure is unchanged may retain their existing `v10.*` schema identifiers for compatibility; the manifest and metrics identify the producing runtime as `v11_version`.

## Profile memory review

List pending candidates with `python -m v11.cli profiles candidates --profile a6`. To promote a method, curate replacement Markdown and provide an evaluation JSON containing the profile ID, candidate revision, baseline revision, distinct held-out game IDs, measured metrics, and `"improved": true`; pass those with `profiles promote`. Promotion requires matching evidence-backed candidates from at least two source games. Reject unsupported proposals with `profiles reject --candidate-ids ID1,ID2 --reason "..."`.

## Tests

Run the v11-specific suite with:

```sh
PYTHONPATH=src /home/zodrak/cuda13.x/bin/python -m pytest -q tests/test_v11_*.py
```

Run the frozen v10 suite separately with `tests/test_v10_*.py` to check that v11 work did not affect v10.
