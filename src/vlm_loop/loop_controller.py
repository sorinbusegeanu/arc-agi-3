from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from v3_1.execution.env_factory import build_env

from .action_schema import extract_available_actions
from .env_runner import EnvSession, run_episode
from .models import ActionSequence, EpisodeResult, LoopConfig
from .prompt_builder import build_prompt, build_prompt_record, load_prompt_config
from .response_parser import build_model_analysis_result, extract_action_sequence, parse_backend_contract_object, validate_stage_contract
from .video_builder import build_episode_video
from .vlm_client import analyze_episode, extract_response_text

STAGE_ROLE_MAP = {
    "start_poi": {"analysis_from": "video_start"},
    "start_poi_actions": {"planning_from": "video_start"},
    "update_poi": {"analysis_from": "video_end"},
    "update_poi_actions": {"planning_from": "video_end"},
    "episode_review": {"review_from": "episode_end"},
}


class LoopController:
    def __init__(
        self,
        *,
        env_factory_path: str | None,
        env_id: str | None,
        env_root: str | None,
        config: LoopConfig,
        session_id: str,
    ) -> None:
        self.env_factory_path = env_factory_path
        self.env_id = env_id
        self.env_root = env_root
        self.config = config
        self.session_id = session_id
        self.session_dir = Path(config.output_root) / session_id
        self.prompt_config = load_prompt_config(config.prompt_config_path)
        self.stage_config_by_id = self._load_stage_config_map()
        self._sequence_counter = 0

    def run_seeds(self) -> dict[str, Any]:
        seed_sequences = self.generate_seed_sequences()
        results = self._run_iteration(iteration_index=0, sequences=seed_sequences, analyze=False)
        return self._write_summary(iteration_summaries=[results], accepted=[], rejected=[])

    def run_loop(self) -> dict[str, Any]:
        episode_summaries: list[dict[str, Any]] = []
        accepted_all: list[dict[str, Any]] = []
        rejected_all: list[dict[str, Any]] = []
        next_run_hint: dict[str, Any] | None = None
        previous_episode_outcome = None
        previous_episode_review = None
        previous_episode_action_sequence: list[str] = []
        previous_episode_video_path = None
        winning_episode_index: int | None = None

        for episode_index in range(self.config.max_iterations):
            episode_summary = self._run_episode(
                episode_index=episode_index,
                next_run_hint=next_run_hint,
                previous_episode_outcome=previous_episode_outcome,
                previous_episode_review=previous_episode_review,
                previous_episode_action_sequence=previous_episode_action_sequence,
                previous_episode_video_path=previous_episode_video_path,
            )
            episode_summaries.append(episode_summary)
            accepted_all.extend(episode_summary.get("accepted_returned_sequences", []))
            rejected_all.extend(episode_summary.get("rejected_returned_sequences", []))

            previous_episode_outcome = episode_summary.get("episode_outcome")
            previous_episode_review = episode_summary.get("episode_review_output")
            previous_episode_action_sequence = list(episode_summary.get("episode_action_sequence_full", []))
            previous_episode_video_path = episode_summary.get("episode_full_video_path")

            if episode_summary.get("episode_won"):
                next_run_hint = None
                if winning_episode_index is None:
                    winning_episode_index = episode_index
            else:
                next_run_hint_value = episode_summary.get("next_run_hint_out")
                next_run_hint = next_run_hint_value if isinstance(next_run_hint_value, dict) else None

        return self._write_summary(
            iteration_summaries=episode_summaries,
            accepted=accepted_all,
            rejected=rejected_all,
            winning_episode_index=winning_episode_index,
        )

    def replay_sequence(self, actions: list[str]) -> EpisodeResult:
        sequence = ActionSequence(
            sequence_id=self._next_sequence_id(),
            actions=actions,
            source="replay",
            parent_sequence_id=None,
        )
        iter_dir = self.session_dir / "iter_replay"
        result = self._run_episodes(iter_dir=iter_dir, sequences=[sequence])[0]
        result.video_path = build_episode_video(result.frame_dir, fps=self.config.fps)
        self._rewrite_episode_json(result)
        return result

    def generate_seed_sequences(self) -> list[ActionSequence]:
        adapter = build_env(self.env_factory_path, env_id=self.env_id, env_root=self.env_root, seed=self.config.seed)
        try:
            actions = extract_available_actions(adapter)
        finally:
            if hasattr(adapter.env, "close"):
                try:
                    adapter.env.close()
                except Exception:
                    pass
        sequences = self._build_long_seed_sequences(actions)
        unique: list[ActionSequence] = []
        seen: set[tuple[str, ...]] = set()
        for actions_row in sequences:
            key = tuple(actions_row)
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                ActionSequence(
                    sequence_id=self._next_sequence_id(),
                    actions=list(actions_row),
                    source="seed",
                    parent_sequence_id=None,
                )
            )
            if len(unique) >= self.config.max_sequences_per_iter:
                break
        return unique

    def _run_episode(
        self,
        *,
        episode_index: int,
        next_run_hint: dict[str, Any] | None,
        previous_episode_outcome: str | None,
        previous_episode_review: dict[str, Any] | None,
        previous_episode_action_sequence: list[str],
        previous_episode_video_path: str | None,
    ) -> dict[str, Any]:
        episode_dir = self.session_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_frames_dir = episode_dir / "frames"
        prompt_records: list[dict[str, Any]] = []
        response_records: list[dict[str, Any]] = []
        accepted_returned_sequences: list[dict[str, Any]] = []
        rejected_returned_sequences: list[dict[str, Any]] = []
        selected_poi_history: list[dict[str, Any]] = []
        segment_metadata: list[dict[str, Any]] = []

        episode_outcome = "timeout"
        episode_done = False
        episode_won = False
        episode_step_count = 0
        episode_action_sequence_full: list[str] = []
        last_selected_poi: dict[str, Any] | None = None
        last_start_poi_output: dict[str, Any] | None = None
        last_update_poi_output: dict[str, Any] | None = None
        last_episode_review_output: dict[str, Any] | None = None
        episode_full_video_path = ""
        start_video_path = ""
        last_update_video_path = ""
        frame_counter_global_for_episode = 0
        iter_index = 0
        episode_review_error: str | None = None
        used_initial_bootstrap = False
        initial_bootstrap_actions: list[str] = []
        start_video_frame_count = 1

        session = EnvSession(
            env_factory_path=self.env_factory_path,
            env_id=self.env_id,
            env_root=self.env_root,
            seed=self.config.seed + episode_index,
            render_terminal=self.config.render_terminal,
        )
        try:
            session.reset(seed=self.config.seed + episode_index)
            start_frame_path, _ = session.capture_current_frame(
                frame_dir=str(episode_frames_dir),
                frame_index=frame_counter_global_for_episode,
            )
            frame_counter_global_for_episode += 1
            if (
                self.config.initial_bootstrap_enabled
                and episode_index == self.config.initial_bootstrap_episode_index
            ):
                used_initial_bootstrap = True
                initial_bootstrap_actions = self._bootstrap_actions_from_action_set(
                    action_set=list(session.allowed_actions),
                    count=self.config.initial_bootstrap_num_actions,
                    configured_actions=self.config.initial_bootstrap_actions,
                )
                if initial_bootstrap_actions:
                    bootstrap_sequence = ActionSequence(
                        sequence_id=self._next_sequence_id(),
                        actions=initial_bootstrap_actions,
                        source="bootstrap",
                        parent_sequence_id=None,
                    )
                    bootstrap_dir = episode_dir / "bootstrap"
                    bootstrap_dir.mkdir(parents=True, exist_ok=True)
                    _, frame_counter_global_for_episode = session.run_sequence(
                        sequence=bootstrap_sequence,
                        episode_dir=str(bootstrap_dir),
                        episode_frame_dir=str(episode_frames_dir),
                        episode_frame_index=frame_counter_global_for_episode,
                        max_steps=self.config.initial_bootstrap_num_actions,
                    )
                    start_video_frame_count = frame_counter_global_for_episode
            start_video_path = build_episode_video(
                str(episode_frames_dir),
                fps=self.config.fps,
                output_name="start_video.mp4",
                frame_index_start=0,
                frame_index_end=max(0, start_video_frame_count - 1),
            )

            while not episode_done:
                if iter_index == 0:
                    analysis_stage_id = "start_poi"
                    action_stage_id = "start_poi_actions"
                    stage_video_path = start_video_path
                    stage_frame_dir = str(episode_frames_dir)
                    analysis_prior_stage_outputs = {}
                    analysis_extra_context = {"next_run_hint": json.loads(json.dumps(next_run_hint))} if isinstance(next_run_hint, dict) else {}
                else:
                    analysis_stage_id = "update_poi"
                    action_stage_id = "update_poi_actions"
                    stage_video_path = last_update_video_path
                    stage_frame_dir = str(episode_frames_dir)
                    analysis_prior_stage_outputs = {}
                    analysis_extra_context = {"previous_target_json": last_selected_poi}

                analysis_output = self._run_stage(
                    episode_dir=episode_dir,
                    episode_index=episode_index,
                    iter_index=iter_index,
                    stage_id=analysis_stage_id,
                    frame_dir=stage_frame_dir,
                    video_path=stage_video_path,
                    action_set=[],
                    prior_stage_outputs_json=analysis_prior_stage_outputs,
                    extra_context=analysis_extra_context,
                    prompt_records=prompt_records,
                    response_records=response_records,
                )
                if analysis_stage_id == "start_poi":
                    last_start_poi_output = analysis_output
                else:
                    last_update_poi_output = analysis_output
                if isinstance(analysis_output.get("poi"), dict):
                    last_selected_poi = json.loads(json.dumps(analysis_output["poi"]))
                    selected_poi_history.append(
                        {
                            "iter_index": iter_index,
                            "stage_id": analysis_stage_id,
                            "poi": last_selected_poi,
                            "target_reached": bool(analysis_output.get("target_reached", False)),
                        }
                    )

                action_output = self._run_stage(
                    episode_dir=episode_dir,
                    episode_index=episode_index,
                    iter_index=iter_index,
                    stage_id=action_stage_id,
                    frame_dir=stage_frame_dir,
                    video_path=stage_video_path,
                    action_set=list(session.allowed_actions),
                    prior_stage_outputs_json={analysis_stage_id: analysis_output},
                    extra_context={},
                    prompt_records=prompt_records,
                    response_records=response_records,
                )
                current_plan_actions, rejection_reason = extract_action_sequence(
                    action_output,
                    field="actions",
                    allowed_actions=list(session.allowed_actions),
                    min_length=5,
                    max_length=5,
                )
                if current_plan_actions is None:
                    rejected_returned_sequences.append(
                        {
                            "episode_index": episode_index,
                            "iter_index": iter_index,
                            "stage_id": action_stage_id,
                            "rejected_action_sequence": action_output.get("actions") if isinstance(action_output.get("actions"), list) else None,
                            "action_sequence_rejection_reason": rejection_reason,
                        }
                    )
                    episode_outcome = "timeout"
                    episode_done = True
                    break
                if episode_step_count + len(current_plan_actions) > self.config.max_steps:
                    rejected_returned_sequences.append(
                        {
                            "episode_index": episode_index,
                            "iter_index": iter_index,
                            "stage_id": action_stage_id,
                            "rejected_action_sequence": current_plan_actions,
                            "action_sequence_rejection_reason": "insufficient_remaining_steps",
                        }
                    )
                    episode_outcome = "timeout"
                    episode_done = True
                    break

                sequence = ActionSequence(
                    sequence_id=self._next_sequence_id(),
                    actions=current_plan_actions,
                    source="model",
                    parent_sequence_id=None,
                )
                accepted_returned_sequences.append(
                    {
                        **sequence.to_dict(),
                        "episode_index": episode_index,
                        "iter_index": iter_index,
                        "stage_id": action_stage_id,
                    }
                )
                segment_dir = episode_dir / f"iter_{iter_index:03d}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                segment_start_frame_index = frame_counter_global_for_episode
                selected_poi_before_execution = json.loads(json.dumps(last_selected_poi)) if last_selected_poi is not None else None
                result, frame_counter_global_for_episode = session.run_sequence(
                    sequence=sequence,
                    episode_dir=str(segment_dir),
                    episode_frame_dir=str(episode_frames_dir),
                    episode_frame_index=frame_counter_global_for_episode,
                    max_steps=self.config.max_steps - episode_step_count,
                )
                result.video_path = build_episode_video(result.frame_dir, fps=self.config.fps, output_name=f"iter_{iter_index:03d}.mp4")
                self._rewrite_episode_json(result)
                segment_end_frame_index = frame_counter_global_for_episode - 1
                last_update_video_path = build_episode_video(
                    str(episode_frames_dir),
                    fps=self.config.fps,
                    output_name=f"update_iter_{iter_index:03d}.mp4",
                    frame_index_start=segment_start_frame_index,
                    frame_index_end=segment_end_frame_index,
                )
                segment_metadata.append(
                    {
                        "iter_index": iter_index,
                        "stage_id": action_stage_id,
                        "actions": list(result.actions),
                        "frame_index_start": segment_start_frame_index,
                        "frame_index_end": segment_end_frame_index,
                        "segment_video_path": last_update_video_path,
                        "selected_poi_before_execution": selected_poi_before_execution,
                        "terminal": bool(result.done or result.truncated),
                    }
                )
                episode_action_sequence_full.extend(result.actions)
                episode_step_count += result.step_count

                if self._session_is_win(session):
                    episode_outcome = "win"
                    episode_done = True
                    episode_won = True
                elif result.done and not self._session_is_win(session):
                    episode_outcome = "game_over"
                    episode_done = True
                elif result.truncated or episode_step_count >= self.config.max_steps:
                    episode_outcome = "timeout"
                    episode_done = True
                else:
                    episode_done = False
                iter_index += 1

            try:
                episode_full_video_path = build_episode_video(str(episode_frames_dir), fps=self.config.fps, output_name="episode.mp4")
            except Exception as exc:
                episode_review_error = f"full_episode_video_failed: {exc}"
                episode_full_video_path = ""

            next_run_hint_out: dict[str, Any] | None = None
            if episode_outcome in {"game_over", "timeout"}:
                if not episode_full_video_path:
                    episode_review_error = episode_review_error or "episode_review skipped: full episode video unavailable"
                elif "episode_review" in self.stage_config_by_id:
                    last_episode_review_output = self._run_stage(
                        episode_dir=episode_dir,
                        episode_index=episode_index,
                        iter_index=iter_index,
                        stage_id="episode_review",
                        frame_dir=str(episode_frames_dir),
                        video_path=episode_full_video_path,
                        action_set=[],
                        prior_stage_outputs_json={},
                        extra_context={"episode_outcome": episode_outcome},
                        prompt_records=prompt_records,
                        response_records=response_records,
                    )
                    next_run_hint_value = last_episode_review_output.get("next_run_hint")
                    next_run_hint_out = json.loads(json.dumps(next_run_hint_value)) if isinstance(next_run_hint_value, dict) else None
                else:
                    episode_review_error = "episode_review stage not configured"
            if episode_won:
                next_run_hint_out = None

            with open(episode_dir / "batch_prompt.json", "w", encoding="utf-8") as handle:
                json.dump(prompt_records, handle, indent=2)
            with open(episode_dir / "batch_response.json", "w", encoding="utf-8") as handle:
                json.dump(response_records, handle, indent=2)

            episode_summary = {
                "episode_index": episode_index,
                "episode_outcome": episode_outcome,
                "episode_done": episode_done,
                "episode_won": episode_won,
                "episode_step_count": episode_step_count,
                "episode_action_sequence_full": episode_action_sequence_full,
                "selected_poi_history": selected_poi_history,
                "next_run_hint_in": next_run_hint,
                "next_run_hint_out": next_run_hint_out,
                "episode_full_video_path": episode_full_video_path,
                "episode_dir": str(episode_dir),
                "episode_frames_dir": str(episode_frames_dir),
                "start_video_path": start_video_path,
                "last_update_video_path": last_update_video_path,
                "frame_counter_global_for_episode": frame_counter_global_for_episode,
                "used_initial_bootstrap": used_initial_bootstrap,
                "initial_bootstrap_num_actions": len(initial_bootstrap_actions),
                "initial_bootstrap_actions": initial_bootstrap_actions,
                "start_video_frame_count": start_video_frame_count,
                "last_selected_poi": last_selected_poi,
                "last_start_poi_output": last_start_poi_output,
                "last_update_poi_output": last_update_poi_output,
                "last_episode_review_output": last_episode_review_output,
                "segment_metadata": segment_metadata,
                "accepted_returned_sequences": accepted_returned_sequences,
                "rejected_returned_sequences": rejected_returned_sequences,
                "previous_episode_outcome": previous_episode_outcome,
                "previous_episode_review": previous_episode_review,
                "previous_episode_action_sequence": previous_episode_action_sequence,
                "previous_episode_video_path": previous_episode_video_path,
                "episode_review_output": last_episode_review_output,
                "episode_review_error": episode_review_error,
            }
            with open(episode_dir / "episode_summary.json", "w", encoding="utf-8") as handle:
                json.dump(episode_summary, handle, indent=2)
            return episode_summary
        finally:
            session.close()

    def _run_stage(
        self,
        *,
        episode_dir: Path,
        episode_index: int,
        iter_index: int,
        stage_id: str,
        frame_dir: str,
        video_path: str,
        action_set: list[str],
        prior_stage_outputs_json: dict[str, Any],
        extra_context: dict[str, Any],
        prompt_records: list[dict[str, Any]],
        response_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stage_config = self.stage_config_by_id[stage_id]
        system_prompt = str(stage_config.get("system_prompt") or self.prompt_config.get("default_system_prompt") or "")
        prompt = build_prompt(
            stage_config=stage_config,
            action_set=action_set,
            prior_stage_outputs=prior_stage_outputs_json,
            extra_context=extra_context,
            task_prompt=self.config.task_prompt,
        )
        prompt_record = build_prompt_record(
            sequence_id=f"episode_{episode_index:03d}",
            stage_id=stage_id,
            system_prompt=system_prompt,
            prompt=prompt,
            action_set=action_set if stage_id in {"start_poi_actions", "update_poi_actions"} else [],
            prior_stage_outputs=prior_stage_outputs_json,
            extra_context=extra_context,
            stage_role=STAGE_ROLE_MAP.get(stage_id, {}),
        )
        prompt_record["episode_index"] = episode_index
        prompt_record["iter_index"] = iter_index
        if stage_id != "start_poi":
            prompt_record["extra_context"].pop("next_run_hint", None)
        if stage_id != "update_poi":
            prompt_record["extra_context"].pop("previous_target_json", None)
        prompt_records.append(prompt_record)
        if self.config.debug:
            print(f"[debug] episode={episode_index} iter={iter_index} stage={stage_id} prompt:")
            print(prompt)
        raw_record = analyze_episode(
            backend=self.config.llm_backend,
            ollama_url=self.config.ollama_url,
            ollama_model=self.config.ollama_model,
            ollama_num_ctx=self.config.ollama_num_ctx,
            vllm_url=self.config.vllm_url,
            vllm_model=self.config.vllm_model,
            disable_thinking=self.config.disable_thinking,
            greedy=self.config.greedy,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            presence_penalty=self.config.presence_penalty,
            out_seq_length=self.config.out_seq_length,
            frame_dir=frame_dir,
            video_path=video_path,
            system_prompt=system_prompt,
            prompt=prompt,
            metadata={
                "episode_index": episode_index,
                "iter_index": iter_index,
                "stage_id": stage_id,
                "next_run_hint": extra_context.get("next_run_hint") if stage_id == "start_poi" else None,
                "previous_target_json": extra_context.get("previous_target_json") if stage_id == "update_poi" else None,
                "action_set": action_set if stage_id in {"start_poi_actions", "update_poi_actions"} else [],
            },
            timeout_sec=self.config.timeout_sec,
            retry_count=self.config.retry_count,
            output_path=str(episode_dir / f"raw_response_{stage_id}.json"),
            max_prompt_frames=self.config.max_prompt_frames,
        )
        raw_text = extract_response_text(raw_record)
        if self.config.debug:
            print(f"[debug] episode={episode_index} iter={iter_index} stage={stage_id} response:")
            print(raw_text)
        payload = validate_stage_contract(
            parse_backend_contract_object(raw_record, backend=self.config.llm_backend),
            stage_id=stage_id,
        )
        with open(episode_dir / f"parsed_contract_{stage_id}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        response_entry = {
            "episode_index": episode_index,
            "iter_index": iter_index,
            "stage_id": stage_id,
            "stage_role": STAGE_ROLE_MAP.get(stage_id, {}),
            "output": payload,
        }
        if stage_id == "start_poi":
            response_entry["next_run_hint"] = extra_context.get("next_run_hint") if extra_context.get("next_run_hint") else None
        if stage_id == "update_poi":
            response_entry["previous_target_json"] = extra_context.get("previous_target_json")
            response_entry["analysis"] = build_model_analysis_result(stage_outputs={"update_poi": payload}, raw_text=raw_text).to_dict()
        elif stage_id == "start_poi":
            response_entry["analysis"] = build_model_analysis_result(stage_outputs={"start_poi": payload}, raw_text=raw_text).to_dict()
        elif stage_id in {"start_poi_actions", "update_poi_actions"}:
            accepted_action_sequence, action_sequence_rejection_reason = extract_action_sequence(
                payload,
                field="actions",
                allowed_actions=action_set,
                min_length=5,
                max_length=5,
            )
            response_entry["action_set"] = action_set
            response_entry["accepted_action_sequence"] = accepted_action_sequence
            response_entry["rejected_action_sequence"] = None if accepted_action_sequence is not None else payload.get("actions")
            response_entry["action_sequence_rejection_reason"] = action_sequence_rejection_reason
        else:
            response_entry["episode_outcome"] = extra_context.get("episode_outcome")
        response_records.append(response_entry)
        return payload

    def _session_is_win(self, session: EnvSession) -> bool:
        info = session.info if isinstance(session.info, dict) else {}
        state = str(info.get("state") or "").upper()
        return bool(
            info.get("win")
            or info.get("won")
            or info.get("is_success")
            or state in {"WIN", "WON", "SUCCESS", "GAMESTATE.WIN", "GAMESTATE.SUCCESS"}
        )

    def _bootstrap_actions_from_action_set(
        self,
        *,
        action_set: list[str],
        count: int,
        configured_actions: list[str],
    ) -> list[str]:
        if configured_actions:
            return list(configured_actions[:count])
        if not action_set or count <= 0:
            return []
        ordered = [str(action).upper() for action in action_set]
        return [ordered[index % len(ordered)] for index in range(count)]

    def _load_stage_config_map(self) -> dict[str, dict[str, Any]]:
        stages = self.prompt_config.get("stages", [])
        if not isinstance(stages, list):
            raise ValueError("prompt config stages must be a list")
        stage_map: dict[str, dict[str, Any]] = {}
        allowed_stage_ids = {"start_poi", "start_poi_actions", "update_poi", "update_poi_actions", "episode_review"}
        for stage in stages:
            stage_id = str(stage.get("stage_id") or "").strip()
            if not stage_id:
                raise ValueError("each prompt stage must define stage_id")
            if stage_id not in allowed_stage_ids:
                raise ValueError(f"unsupported stage id in prompt config: {stage_id}")
            stage_map[stage_id] = dict(stage)
        missing_required = {"start_poi", "start_poi_actions", "update_poi", "update_poi_actions"} - set(stage_map)
        if missing_required:
            raise ValueError(f"prompt config is missing required stage ids: {sorted(missing_required)}")
        return stage_map

    def _run_iteration(self, *, iteration_index: int, sequences: list[ActionSequence], analyze: bool) -> dict[str, Any]:
        if analyze:
            raise RuntimeError("analyzed iterations must use the episode-aware run_loop path")
        iter_dir = self.session_dir / f"iter_{iteration_index:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        episode_results = self._run_episodes(iter_dir=iter_dir, sequences=sequences[: self.config.max_sequences_per_iter])
        for result in episode_results:
            result.video_path = build_episode_video(result.frame_dir, fps=self.config.fps)
            self._rewrite_episode_json(result)
        return {
            "iteration_index": iteration_index,
            "phase": "seed",
            "episode_results": [result.to_dict() for result in episode_results],
            "tested_sequences": [sequence.to_dict() for sequence in sequences],
            "accepted_returned_sequences": [],
            "rejected_returned_sequences": [],
            "next_sequences": [],
            "artifact_dir": str(iter_dir),
        }

    def _build_long_seed_sequences(self, actions: list[str]) -> list[list[str]]:
        if not actions:
            return []
        sequences: list[list[str]] = []
        target_count = max(1, self.config.max_sequences_per_iter)
        target_length = max(1, self.config.max_steps)
        rng = random.Random(self.config.seed)
        for seq_index in range(target_count):
            sequence_rng = random.Random(rng.randint(0, 10**9) + seq_index)
            sequence = [sequence_rng.choice(actions) for _ in range(target_length)]
            sequences.append(sequence)
        return sequences

    def _run_episodes(self, *, iter_dir: Path, sequences: list[ActionSequence]) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        with ThreadPoolExecutor(max_workers=max(1, self.config.agents_per_iteration)) as executor:
            future_map = {}
            for sequence in sequences:
                episode_dir = iter_dir / sequence.sequence_id
                episode_dir.mkdir(parents=True, exist_ok=True)
                future = executor.submit(
                    run_episode,
                    env_factory_path=self.env_factory_path,
                    env_id=self.env_id,
                    env_root=self.env_root,
                    seed=self.config.seed,
                    render_terminal=self.config.render_terminal,
                    sequence=sequence,
                    episode_dir=str(episode_dir),
                    max_steps=self.config.max_steps,
                )
                future_map[future] = sequence.sequence_id
            for future in as_completed(future_map):
                results.append(future.result())
        results.sort(key=lambda item: item.sequence_id)
        return results

    def _rewrite_episode_json(self, result: EpisodeResult) -> None:
        path = Path(result.output_dir) / "episode.json"
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(result.to_dict())
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_summary(
        self,
        *,
        iteration_summaries: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        winning_episode_index: int | None = None,
    ) -> dict[str, Any]:
        summary = {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "flow": {
                "episode_start_chain": ["start_poi", "start_poi_actions"],
                "episode_continuation_chain": ["update_poi", "update_poi_actions"],
                "episode_end_chain": ["episode_review"],
            },
            "total_episodes_run": len(iteration_summaries),
            "winning_episode_index": winning_episode_index,
            "per_episode_outcomes": [
                {"episode_index": item.get("episode_index"), "episode_outcome": item.get("episode_outcome")}
                for item in iteration_summaries
            ],
            "carried_hints": [
                {
                    "episode_index": item.get("episode_index"),
                    "next_run_hint_in": item.get("next_run_hint_in"),
                    "next_run_hint_out": item.get("next_run_hint_out"),
                }
                for item in iteration_summaries
            ],
            "accepted_returned_sequences": accepted,
            "rejected_returned_sequences": rejected,
            "artifact_paths": [item.get("episode_dir") or item.get("artifact_dir") for item in iteration_summaries],
            "episodes": iteration_summaries,
        }
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with open(self.session_dir / "session_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def _next_sequence_id(self) -> str:
        value = self._sequence_counter
        self._sequence_counter += 1
        return f"seq_{value:04d}"
