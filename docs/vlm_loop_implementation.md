# `src/vlm_loop` Implementation

This document describes the current implementation of `src/vlm_loop` as it exists in code on 2026-03-26. It is an implementation note, not a design spec. Where behavior is heuristic, fixed, or asymmetric across backends, that is described directly.

## Overview

`src/vlm_loop` implements a minimal vision-language closed loop for grid or arcade-style environments. The loop:

1. creates or resets an environment
2. records PNG frames from the live observation stream
3. builds short MP4 clips from those frames
4. sends a prompt plus sampled frames to a VLM backend
5. validates the model against a strict stage JSON contract
6. executes exactly one 5-action plan at a time
7. repeats until win, game over, or step budget exhaustion
8. optionally carries a compact `next_run_hint` from one failed episode into the next

The public entrypoints are [cli.py](/home/zodrak/zod/src/vlm_loop/cli.py), [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py), and [config.py](/home/zodrak/zod/src/vlm_loop/config.py).

## Module Map

- [cli.py](/home/zodrak/zod/src/vlm_loop/cli.py): argparse entrypoint for `run-seeds`, `run-loop`, and `replay-sequence`.
- [config.py](/home/zodrak/zod/src/vlm_loop/config.py): builds the immutable `LoopConfig`.
- [models.py](/home/zodrak/zod/src/vlm_loop/models.py): dataclasses for sequences, episode results, model analysis summaries, and config.
- [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py): top-level orchestration for seed runs, iterative VLM episodes, stage execution, summaries, and artifact writing.
- [env_runner.py](/home/zodrak/zod/src/vlm_loop/env_runner.py): environment session wrapper, action execution, frame capture, and `episode.json` generation.
- [action_schema.py](/home/zodrak/zod/src/vlm_loop/action_schema.py): action canonicalization, allowed-action extraction, and sequence validation.
- [frame_writer.py](/home/zodrak/zod/src/vlm_loop/frame_writer.py): converts observations into upscaled PNGs.
- [video_builder.py](/home/zodrak/zod/src/vlm_loop/video_builder.py): builds MP4s from selected frame ranges using `ffmpeg`.
- [prompt_builder.py](/home/zodrak/zod/src/vlm_loop/prompt_builder.py): loads prompt config and formats stage prompts.
- [response_parser.py](/home/zodrak/zod/src/vlm_loop/response_parser.py): extracts JSON from backend responses and validates stage contracts.
- [vlm_client.py](/home/zodrak/zod/src/vlm_loop/vlm_client.py): backend-specific HTTP request construction and response recording.
- [prompt_config.json](/home/zodrak/zod/src/vlm_loop/prompt_config.json): live stage definitions and default backend parameters.

## CLI and Configuration

The CLI in [cli.py](/home/zodrak/zod/src/vlm_loop/cli.py) exposes three commands:

- `run-seeds`: generate random long action sequences and execute them without VLM analysis.
- `run-loop`: run the staged closed loop for `--iters` episodes.
- `replay-sequence`: execute one explicit comma-separated action sequence and print the resulting `EpisodeResult` as JSON.

`build_config(...)` in [config.py](/home/zodrak/zod/src/vlm_loop/config.py) mostly forwards CLI values directly into `LoopConfig`. Important defaults from [models.py](/home/zodrak/zod/src/vlm_loop/models.py):

- `max_steps`: total per-episode environment action budget
- `max_iterations`: number of episodes for `run-loop`
- `max_sequences_per_iter`: seed-mode sequence count
- `agents_per_iteration`: thread-pool width for seed execution only
- `max_prompt_frames`: number of PNG frames attached to each model request
- `initial_bootstrap_enabled`: whether episode 0 executes a fixed bootstrapping prefix before the first analysis stage

The CLI merges config from flags and [prompt_config.json](/home/zodrak/zod/src/vlm_loop/prompt_config.json). CLI values win when explicitly provided. Booleans coming from JSON are normalized by `_config_bool(...)` in [cli.py](/home/zodrak/zod/src/vlm_loop/cli.py).

## Environment Integration

Environment creation is delegated to `v3_1.execution.env_factory.build_env(...)`, called from [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py) and [env_runner.py](/home/zodrak/zod/src/vlm_loop/env_runner.py). This means `vlm_loop` does not own environment semantics; it operates against the normalized adapter returned by `build_env(...)`.

`EnvSession` in [env_runner.py](/home/zodrak/zod/src/vlm_loop/env_runner.py) is the execution wrapper. It:

- creates the normalized adapter
- resets the environment with a seed
- keeps the latest `observation`, `info`, `done`, `truncated`, and `allowed_actions`
- writes PNG frames for both per-segment and episode-global timelines
- executes validated action sequences and writes `episode.json`

Allowed actions are sourced from:

1. `adapter.env.available_actions()` if present
2. `adapter.available_actions()`
3. `info["available_actions"]` as a fallback

Action naming is canonicalized in [action_schema.py](/home/zodrak/zod/src/vlm_loop/action_schema.py) using `v3_1.execution.env_factory.normalize_action_lookup(...)`. Internally, the loop operates on uppercased action names such as `UP`, `DOWN`, `LEFT`, `RIGHT`, and converts them back to env-facing values only at execution time in `resolve_env_action(...)` in [env_runner.py](/home/zodrak/zod/src/vlm_loop/env_runner.py).

## Frame and Video Pipeline

Frames are written by `write_frame_png(...)` in [frame_writer.py](/home/zodrak/zod/src/vlm_loop/frame_writer.py).

- 2D integer grids are mapped through the ARC color table.
- RGB or RGBA observations are clipped to `uint8`.
- Every image is upscaled 10x with nearest-neighbor resampling before being saved as `frame_%06d.png`.

Videos are built by `build_episode_video(...)` in [video_builder.py](/home/zodrak/zod/src/vlm_loop/video_builder.py).

- It selects either all frames or an explicit index slice.
- It materializes a temporary sequential frame directory using symlinks when possible.
- It calls `ffmpeg` with `libx264` and `yuv420p`.

The loop builds several MP4 types:

- `start_video.mp4`: episode prefix from frame `0` through the initial bootstrap window
- `iter_XXX.mp4`: per-segment local frames written in the segment directory
- `update_iter_XXX.mp4`: the same segment but sliced from the episode-global frame timeline
- `episode.mp4`: full episode video

## Stage Model

The controller only supports five stage ids, enforced by `_load_stage_config_map(...)` in [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py):

- `start_poi`
- `start_poi_actions`
- `update_poi`
- `update_poi_actions`
- `episode_review`

The required first four stages must exist in [prompt_config.json](/home/zodrak/zod/src/vlm_loop/prompt_config.json). `episode_review` is optional.

Stage roles are summarized by `STAGE_ROLE_MAP` in [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py):

- `start_poi`: analyze from the episode-start video
- `start_poi_actions`: plan from the episode-start video
- `update_poi`: analyze from the most recent executed segment
- `update_poi_actions`: plan from the most recent executed segment
- `episode_review`: review the completed full episode

## Episode Algorithm

The main closed loop lives in `LoopController.run_loop()` and `_run_episode()` in [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py).

For each episode:

1. create `episode_{index}` and `frames/`
2. reset `EnvSession` with seed `config.seed + episode_index`
3. capture frame `0`
4. optionally run the initial bootstrap prefix only for `initial_bootstrap_episode_index`
5. build `start_video.mp4`
6. run alternating analysis and action stages until terminal
7. after termination, build `episode.mp4`
8. if outcome is `game_over` or `timeout`, optionally run `episode_review`
9. write `batch_prompt.json`, `batch_response.json`, `episode_summary.json`, and later the session summary

Within each non-terminal iteration:

1. choose `start_poi` / `start_poi_actions` on the first pass, otherwise `update_poi` / `update_poi_actions`
2. build the prompt from stage config, prior outputs, and extra context
3. call the backend
4. parse and validate the returned JSON
5. extract exactly one 5-action sequence
6. reject immediately if the action list is missing, the wrong length, contains unknown actions, or exceeds remaining step budget
7. execute the accepted sequence in the live environment
8. record segment metadata and decide whether the episode ended

The terminal conditions are:

- `win`: `_session_is_win(...)` detects success via `info["win"]`, `info["won"]`, `info["is_success"]`, or success-like `state` strings
- `game_over`: environment returned `done` but success was not detected
- `timeout`: truncated, step budget exhausted, or model action sequence rejected

## Bootstrap Behavior

Bootstrap is implemented in `_bootstrap_actions_from_action_set(...)` in [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py).

- If `initial_bootstrap_actions` is configured, the loop uses that prefix truncated to `initial_bootstrap_num_actions`.
- Otherwise it cycles through the currently available action set until it has `count` actions.
- Bootstrap runs before the first `start_poi` analysis, so the initial video may contain several actions worth of motion instead of only the reset frame.

## Prompt Construction

`build_prompt(...)` in [prompt_builder.py](/home/zodrak/zod/src/vlm_loop/prompt_builder.py) formats the stage prompt from:

- stage-local `task_prompt` or the CLI `--task-prompt`
- `action_set`
- prior stage outputs
- `previous_target_json`
- `next_run_hint`
- `episode_outcome`

The implementation uses Python `.format(...)` directly against `user_prompt_template`, so unsupported template variables raise a `ValueError`.

If `next_run_hint` is absent, the builder removes a line whose stripped content is exactly `Next run hint:`. This is a narrow cleanup rule, not a general templating system.

`build_prompt_record(...)` stores a JSON-safe copy of the final prompt inputs into `batch_prompt.json`.

## Backend Requests

`analyze_episode(...)` in [vlm_client.py](/home/zodrak/zod/src/vlm_loop/vlm_client.py) constructs a request, sends it with `requests.post(...)`, retries up to `retry_count + 1` times, and writes the raw transaction record to `raw_response_{stage}.json`.

### vLLM path

For `backend == "vllm"`:

- endpoint: `{vllm_url}/v1/chat/completions`
- request format: OpenAI-style chat completion
- attached media: sampled PNG frames encoded as `data:image/png;base64,...`
- response constraint: `response_format = {"type": "json_object"}`
- extra body fields: `enable_thinking`, `greedy`, `top_k`, `repetition_penalty`, `presence_penalty`

Notably, `video_path` is passed through the controller but ignored in `_build_vllm_payload(...)`; only sampled frames are sent.

### Ollama path

For `backend == "ollama"`:

- endpoint: `{ollama_url}/api/chat`
- request format: Ollama chat API
- attached media: sampled PNG frames in `messages[1].images`
- response constraint: `format = "json"`
- option fields: `num_ctx`, `num_predict`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `presence_penalty`

The `greedy` argument is accepted by the controller but is not used in `_build_ollama_payload(...)`.

### Frame sampling

Both backends use `_sample_frame_paths(...)` in [vlm_client.py](/home/zodrak/zod/src/vlm_loop/vlm_client.py).

- If the frame count is already small enough, all frames are attached.
- If more frames exist than allowed, the sampler chooses evenly distributed indices from first to last inclusive.
- If `max_prompt_frames <= 1`, only the first frame is sent.

## Response Parsing and Contract Validation

`extract_response_text(...)` in [vlm_client.py](/home/zodrak/zod/src/vlm_loop/vlm_client.py) strips leading `<think>` or `<reasoning>` blocks before higher-level summarization.

`parse_backend_contract_object(...)` in [response_parser.py](/home/zodrak/zod/src/vlm_loop/response_parser.py):

- locates the backend-specific message content
- accepts either an already-parsed object or a JSON string
- tries whole-string JSON parsing first
- falls back to extracting the first balanced JSON object
- falls back again to a partial root-object parser for truncated outputs

`validate_stage_contract(...)` then enforces stage-specific schemas:

- `start_poi` and `update_poi` must return `sprite` and `poi` objects
- `start_poi_actions` and `update_poi_actions` must return an `actions` list
- `episode_review` must return string fields for the review plus an optional `next_run_hint` object with exactly `sprite_description`, `target`, `hud`, and `avoid`

Action stages undergo a second acceptance pass through `extract_action_sequence(...)`:

- list must exist at the target field
- minimum length must be 5
- maximum length must be 5
- every action must be in the current allowed action set

Any failure is recorded into `batch_response.json` and, in the episode loop, also into `rejected_returned_sequences`.

## Carried State Across Episodes

`run_loop()` in [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py) carries forward a small amount of prior-episode context:

- `previous_episode_outcome`
- `previous_episode_review`
- `previous_episode_action_sequence`
- `previous_episode_video_path`
- `next_run_hint`

In practice, only `next_run_hint` is injected back into the next prompt flow, and only into the first `start_poi` stage. The code stores the other prior values in `episode_summary.json` for inspection, but does not feed them back into the later prompt builder.

If an episode is won, the carried hint is cleared.

## Seed and Replay Modes

`run-seeds()` in [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py) is separate from the analyzed episode loop.

- It queries the environment for allowed actions.
- It generates `max_sequences_per_iter` random full-length sequences using `_build_long_seed_sequences(...)`.
- It executes them in parallel threads through `_run_episodes(...)`.
- It writes videos and a seed-style summary, but does not call the VLM.

`replay_sequence(...)` executes one explicit action list inside `iter_replay/`, builds a video, rewrites `episode.json`, and returns the `EpisodeResult`.

## Artifact Layout

At session scope, [loop_controller.py](/home/zodrak/zod/src/vlm_loop/loop_controller.py) writes:

- `runs/vlm_loop/<session_id>/session_summary.json`

Per episode it writes:

- `episode_summary.json`
- `batch_prompt.json`
- `batch_response.json`
- `raw_response_<stage>.json`
- `parsed_contract_<stage>.json`
- `start_video.mp4`
- `update_iter_XXX.mp4`
- `episode.mp4`
- `frames/frame_*.png`

Per executed segment it writes:

- `iter_XXX/episode.json`
- `iter_XXX/episode.mp4`
- `iter_XXX/frames/frame_*.png`

The summary objects include both accepted and rejected action sequences, selected POI history, segment metadata, bootstrap details, and carried hint data.

## Important Implementation Constraints

- The controller is not a tree search. It executes one accepted 5-step sequence at a time.
- The stage set is fixed in code; prompt config cannot add arbitrary new stages.
- The loop treats model output as authoritative only after strict contract validation.
- All action plans are currently hard-clamped to length 5 in both prompt wording and runtime validation.
- Media transport is frame-based for both backends; `video_path` is not uploaded to the model.
- Seed-mode parallelism exists, but the main analyzed `run-loop` path is episode-serial and intra-episode serial.
- `analyze_episode(...)` raises `RuntimeError("vllm request failed: ...")` on final failure even when the selected backend is Ollama; this is just the current error string.
