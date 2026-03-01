## Codex Spec: `src/my_agi_games` package for new ARCEngine games + variants (no code)

### Reference example

Use the existing downloaded game as the style/compatibility baseline:

* A game is a single Python module defining `sprites`, `levels`, and an `ARCBaseGame` subclass with `step()` and `on_set_level()` logic. 
* Each packaged “game instance” has a `metadata.json` containing `game_id`, `title`, `default_fps`, etc. 

---

# 0) Goals

1. Add a **separate package** `src/my_agi_games` that can:

   * define new base games (classic-mechanic inspired)
   * generate **many deterministic variants** per base game
   * export variants in the **same form** as existing environment files (a folder containing a python game file + metadata)

2. Keep the engine-facing interface identical: “game module + metadata”.

3. Enforce: deterministic generation, strict train/eval seed splits, reproducible IDs, and a uniform parameter schema.

---

# 1) Repository layout (must implement)

Create:

* `src/my_agi_games/README.md`
* `src/my_agi_games/__init__.py`
* `src/my_agi_games/registry.py`
* `src/my_agi_games/specs/` (human-readable design specs per game)
* `src/my_agi_games/games/<base_game_name>/` (one folder per base game)

  * `design.md` (mechanic + params)
  * `base.py` (base game implementation template)
  * `assets.py` (sprites + palette helpers)
  * `variant.py` (variant parameter sampling)
  * `export.py` (export adapter)
* `src/my_agi_games/tools/`

  * `export_variants.py` (CLI entry)
  * `list_games.py`
  * `validate_variant.py`
* `tests/my_agi_games/`

  * determinism tests
  * export format tests
  * smoke-run tests (step loop)

---

# 2) “Base game” contract (must document + enforce)

Each base game implementation must provide:

## 2.1 Class contract

* Subclass `ARCBaseGame`
* Constructor must accept:

  * `variant_spec` (dict-like)
  * `seed` (int)
* Must set:

  * `game_name` (stable base name like `"sokoban"`)
  * camera config (fixed, standard default)
* Must implement:

  * `on_set_level(level)`
  * `step()`
* Must only rely on `variant_spec + seed` for behavior/layout (no global randomness).

(Keep the code readable; do not obfuscate identifiers like the reference file.) 

## 2.2 Variant determinism

Given `(base_game_name, seed, variant_spec)`:

* Generated level layouts and all initial placements are identical across runs.

## 2.3 Action semantics

* Must use the standard ARCEngine action set; if ACTION6 (click) is used, define exactly how `(x,y)` maps to grid coordinates via camera display mapping.

## 2.4 Win/Lose semantics

* Must call engine win/lose mechanisms consistently.
* Must define terminal conditions clearly in `design.md`.

---

# 3) Variant specification schema (uniform across all games)

Define a project-wide schema `VariantSpec` used by all base games.

Required fields:

* `schema_version`: string
* `base_game`: string
* `seed`: int
* `variant_id`: string (stable)
* `split`: `"train"` or `"eval"`
* `difficulty`: int (0..N)
* `palette_id`: int
* `layout_id`: int
* `params`: dict (game-specific knobs)

Rules:

* `variant_id` must be derived deterministically from `(base_game, seed, params)` (hash or canonical string).
* `split` must be derived from seed ranges (see §6).
* `difficulty` must be derived deterministically from seed or explicitly set by generator.

---

# 4) Export format (must match existing environment_files style)

Implement an exporter that writes variants to:

* `environment_files/<base_game>/<variant_hash>/`

  * `<base_game>.py` (or `game.py`; choose one and make consistent)
  * `metadata.json`

Metadata must include at least:

* `game_id` in the format `<base_game>-<variant_hash>` 
* `title` (human readable)
* `default_fps`
* `tags` (optional)
* `local_dir` (absolute path written at export time)
* `date_downloaded` or `date_generated` (ISO string)

The generated Python file must be importable and contain the `ARCBaseGame` subclass plus any required sprite/level definitions. 

---

# 5) Registry (base games + variant construction)

Implement `src/my_agi_games/registry.py`:

* `list_base_games() -> [BaseGameInfo]`
* `get_base_game(name) -> BaseGameFactory`
* `make_variant(name, seed, overrides) -> VariantSpec`
* `build_game_instance(variant_spec) -> ARCBaseGame`

Registry rules:

* Adding a new game requires only:

  * adding its folder under `games/<name>/`
  * registering it in one registry table
* No other code path should hardcode base game names.

---

# 6) Train/Eval splitting contract (must be enforced)

Use seed ranges (project-wide):

* Train seeds: `[0, 999_999]`
* Eval seeds: `[1_000_000, 1_099_999]`

Exporter and tools must:

* Reject seeds outside configured ranges (unless `--allow-custom-seed-range` is explicitly provided)
* Write `split` into the `VariantSpec` and metadata tags.

---

# 7) Validation tools (must implement)

## 7.1 Determinism validator

`validate_variant.py` must:

* construct the same variant twice
* run K fixed actions (or random but seeded action sequence)
* assert frame hashes match at each step
* assert terminal signals match

## 7.2 Export validator

Must check output folder contains:

* python file
* metadata.json
* metadata matches `VariantSpec` fields and `game_id` format 

## 7.3 Smoke runner

Given exported variant folder:

* import module
* instantiate game
* run a short episode
* ensure no exceptions

---

# 8) Standard parameter knobs (so variants are meaningful)

Each base game must support a subset of standardized knobs, even if not all are used:

Common knobs:

* `grid_size_id` (mapped to concrete sizes)
* `object_density`
* `obstacle_density`
* `num_targets`
* `num_enemies`
* `enemy_speed_id`
* `time_limit_id`
* `reward_shaping_profile` (optional; mostly for your RL wrapper)
* `palette_id`
* `distractor_profile_id`

Each game’s `design.md` must state:

* which knobs are used
* valid discrete ranges
* which knobs affect layout vs dynamics

---

# 9) Initial base games to implement (per your ranked list)

Codex should implement base games incrementally, one folder at a time, starting with those that share infrastructure:

Phase 1 (movement + grid layout):

1. Maze Collect-All
2. Key–Door Puzzle
3. Sokoban

Phase 2 (click mechanic):
4) Lights-Out / Toggle Puzzle (ACTION6 core)

Phase 3 (enemies / dynamics):
5) Enemy Avoidance (Pacman-lite)

(Do not implement all 10+ at once; ensure generator/export/determinism pipeline is stable first.)

---

# 10) Non-negotiable constraints

* No global RNG usage; only seed-local RNG.
* No hidden state outside the game instance.
* No cross-variant shared mutable state (sprites must be cloned per level, like the reference). 
* All generated variants must be reproducible solely from `(base_game, seed, overrides)`.

---

# 11) Integration points with existing project runner

Codex must:

* ensure exported variants land under `environment_files/...` in the same structure the existing runner already consumes (as shown by the ft09 example metadata `local_dir`). 
* add no changes to the RL pipeline here, only environment generation.

---

If you want this spec tightened into “one-file-per-prompt” Codex instructions, say whether the export python filename should be `<base_game>.py` or `game.py` (pick one), and whether you want `environment_files/` written inside the repo or to a configurable external directory.

