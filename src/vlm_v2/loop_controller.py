from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .action_schema import action_sequence_to_letters
from .env_runner import run_branch
from .models import BranchPlan, BranchRunResult, ObjectActionProposal, StartLevelAnalysis, VLMV2Config
from .prompt_builder import (
    build_prompt_memory,
    build_prompt,
)
from .response_parser import parse_object_action_proposals, parse_start_level_analysis
from .vlm_client import call_ollama, extract_response_text


class LoopController:
    def __init__(self, *, config: VLMV2Config, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self.logger = logging.getLogger("vlm_v2")
        self.session_dir = Path(config.output_root) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.session_dir / "config.json", config.to_dict())

    def run_episode(self) -> dict[str, Any]:
        episode_dir = self.session_dir / "episode_000"
        episode_dir.mkdir(parents=True, exist_ok=True)

        total_actions_used = 0
        completed_episode_actions: list[str] = []
        saved_level_files: list[str] = []
        level_index = 0
        won = False
        stop_reason = "unknown"
        level_summaries: list[dict[str, Any]] = []

        while total_actions_used < self.config.max_actions_budget:
            level_dir = episode_dir / f"level_{level_index:03d}"
            level_dir.mkdir(parents=True, exist_ok=True)
            level_completed_before = level_index
            bootstrap_actions = list(self.config.bootstrap_actions)
            bootstrap_letters = action_sequence_to_letters(bootstrap_actions) if bootstrap_actions else ""

            base_branch = BranchPlan(
                branch_id=f"level_{level_index:03d}_base",
                object_name="bootstrap",
                current_level_actions=list(bootstrap_actions),
                source_prompt_stage="bootstrap",
            )
            base_full_actions = list(completed_episode_actions) + list(base_branch.current_level_actions)
            remaining_budget = self.config.max_actions_budget - total_actions_used
            base_result = run_branch(
                cfg=self.config,
                seed=self.config.seed,
                branch=base_branch,
                full_actions=base_full_actions,
                output_dir=str(level_dir / "base_run"),
                max_actions_allowed=remaining_budget,
            )
            total_actions_used += int(base_result.executed_count)
            self._write_json(level_dir / "base_result.json", base_result.to_dict())

            if base_result.won or base_result.level_advanced:
                saved_path = self._save_level_solution(
                    level_dir=level_dir,
                    level_start=level_completed_before,
                    level_end=base_result.levels_completed_after,
                    actions=base_branch.current_level_actions,
                )
                saved_level_files.append(saved_path)
                completed_episode_actions.extend(base_branch.current_level_actions)
                level_summaries.append(
                    {
                        "level_index": level_index,
                        "level_start": level_completed_before,
                        "level_end": base_result.levels_completed_after,
                        "winning_branch": base_branch.to_dict(),
                        "solution_file": saved_path,
                        "base_only": True,
                    }
                )
                if base_result.won:
                    level_index = int(base_result.levels_completed_after)
                    won = True
                    stop_reason = "win"
                    break
                level_index = int(base_result.levels_completed_after)
                continue

            if base_result.done or base_result.truncated:
                stop_reason = "terminated_before_prompting"
                break

            start_level_analysis = self._run_start_level_prompt(
                level_dir=level_dir,
                level_index=level_index,
                frame_dir=base_result.frame_dir,
                bootstrap_letters=bootstrap_letters,
            )
            initial_proposals = self._run_initial_action_prompt(
                level_dir=level_dir,
                level_index=level_index,
                frame_dir=base_result.frame_dir,
                bootstrap_letters=bootstrap_letters,
                analysis=start_level_analysis,
            )
            frontier = self._build_initial_frontier(
                level_index=level_index,
                bootstrap_actions=bootstrap_actions,
                proposals=initial_proposals,
            )
            if not frontier:
                stop_reason = "no_initial_object_actions"
                break

            success_result: BranchRunResult | None = None
            round_index = 0
            while frontier and total_actions_used < self.config.max_actions_budget:
                selected = self._select_frontier(frontier, completed_episode_actions, total_actions_used)
                if not selected:
                    stop_reason = "budget_exhausted_before_branch_round"
                    break
                next_frontier: list[BranchPlan] = []
                with ThreadPoolExecutor(max_workers=min(self.config.max_parallel_branches, len(selected))) as executor:
                    futures = {
                        executor.submit(
                            self._evaluate_branch_round,
                            level_dir=level_dir,
                            round_index=round_index,
                            branch=branch,
                            completed_episode_actions=list(completed_episode_actions),
                            level_index=level_index,
                            start_level_analysis=start_level_analysis,
                        ): branch
                        for branch in selected
                    }
                    for future in as_completed(futures):
                        outcome = future.result()
                        result = outcome["branch_result"]
                        total_actions_used += int(result.executed_count)
                        if result.won or result.level_advanced:
                            success_result = result
                            for pending in futures:
                                pending.cancel()
                            break
                        next_frontier.extend(outcome["next_frontier"])
                        if total_actions_used >= self.config.max_actions_budget:
                            break
                if success_result is not None:
                    break
                frontier = self._dedupe_frontier(next_frontier)
                round_index += 1

            if success_result is None:
                if total_actions_used >= self.config.max_actions_budget:
                    stop_reason = "action_budget_exhausted"
                elif stop_reason == "unknown":
                    stop_reason = "level_unsolved"
                break

            saved_path = self._save_level_solution(
                level_dir=level_dir,
                level_start=level_completed_before,
                level_end=success_result.levels_completed_after,
                actions=success_result.branch.current_level_actions,
            )
            saved_level_files.append(saved_path)
            completed_episode_actions.extend(success_result.branch.current_level_actions)
            level_summaries.append(
                {
                    "level_index": level_index,
                    "level_start": level_completed_before,
                    "level_end": success_result.levels_completed_after,
                    "winning_branch": success_result.branch.to_dict(),
                    "winning_result": success_result.to_dict(),
                    "solution_file": saved_path,
                    "base_only": False,
                }
            )
            if success_result.won:
                level_index = int(success_result.levels_completed_after)
                won = True
                stop_reason = "win"
                break
            level_index = int(success_result.levels_completed_after)

        if not won and stop_reason == "unknown":
            stop_reason = "action_budget_exhausted"

        summary = {
            "env_id": self.config.env_id,
            "episode_index": 0,
            "won": bool(won),
            "stop_reason": stop_reason,
            "levels_completed": int(level_index),
            "total_actions_used": int(total_actions_used),
            "max_actions_budget": int(self.config.max_actions_budget),
            "completed_episode_actions": list(completed_episode_actions),
            "completed_episode_actions_letters": action_sequence_to_letters(completed_episode_actions) if completed_episode_actions else "",
            "saved_level_files": list(saved_level_files),
            "level_summaries": level_summaries,
        }
        self._write_json(episode_dir / "episode_summary.json", summary)
        return summary

    def _run_start_level_prompt(
        self,
        *,
        level_dir: Path,
        level_index: int,
        frame_dir: str,
        bootstrap_letters: str,
    ) -> StartLevelAnalysis:
        conversation_scope = self._level_conversation_scope(level_index)
        prompt = build_prompt(
            self.config.start_level_prompt,
            memory=build_prompt_memory(
                action_list=bootstrap_letters,
                bootstrap_action_list=bootstrap_letters,
                bootstrap_sequence=bootstrap_letters,
                level_index=level_index,
            ),
        )
        raw_path = level_dir / "start_level_raw.json"
        record = call_ollama(
            ollama_url=self.config.ollama_url,
            ollama_model=self.config.ollama_model,
            ollama_num_ctx=self.config.ollama_num_ctx,
            system_prompt=self.config.system_prompt,
            prompt=prompt,
            frame_dir=frame_dir,
            max_prompt_frames=self.config.max_prompt_frames,
            timeout_sec=self.config.timeout_sec,
            retry_count=self.config.retry_count,
            output_path=str(raw_path),
            conversation_scope=conversation_scope,
            reset_context=True,
            metadata={"stage": "start_level_prompt", "level_index": level_index},
        )
        text = extract_response_text(record)
        analysis = parse_start_level_analysis(text)
        self._write_json(level_dir / "start_level_parsed.json", analysis.to_dict())
        self._debug_prompt_round(
            stage="start_level_prompt",
            prompt=prompt,
            answer_text=text,
            parsed_payload=analysis.to_dict(),
            extra={"level_index": level_index, "frame_dir": frame_dir},
        )
        return analysis

    def _run_initial_action_prompt(
        self,
        *,
        level_dir: Path,
        level_index: int,
        frame_dir: str,
        bootstrap_letters: str,
        analysis: StartLevelAnalysis,
    ) -> list[ObjectActionProposal]:
        conversation_scope = self._level_conversation_scope(level_index)
        prompt = build_prompt(
            self.config.get_list_of_objects_actions_prompt,
            memory=build_prompt_memory(
                action_list=bootstrap_letters,
                bootstrap_action_list=bootstrap_letters,
                bootstrap_sequence=bootstrap_letters,
                level_index=level_index,
                player=analysis.player,
                layout=analysis.layout,
                reasoning=analysis.reasoning,
                hud=analysis.hud,
                objects=analysis.to_dict().get("objects", []),
                objects_json=analysis.to_dict().get("objects", []),
                start_level_analysis_json=analysis.to_dict(),
            ),
        )
        raw_path = level_dir / "get_list_of_objects_actions_raw.json"
        record = call_ollama(
            ollama_url=self.config.ollama_url,
            ollama_model=self.config.ollama_model,
            ollama_num_ctx=self.config.ollama_num_ctx,
            system_prompt=self.config.system_prompt,
            prompt=prompt,
            frame_dir=frame_dir,
            max_prompt_frames=self.config.max_prompt_frames,
            timeout_sec=self.config.timeout_sec,
            retry_count=self.config.retry_count,
            output_path=str(raw_path),
            conversation_scope=conversation_scope,
            reset_context=False,
            metadata={"stage": "get_list_of_objects_actions", "level_index": level_index},
        )
        text = extract_response_text(record)
        proposals = parse_object_action_proposals(text)
        self._write_json(level_dir / "get_list_of_objects_actions_parsed.json", [item.to_dict() for item in proposals])
        self._debug_prompt_round(
            stage="get_list_of_objects_actions_prompt",
            prompt=prompt,
            answer_text=text,
            parsed_payload={"Actions": [item.to_dict() for item in proposals]},
            extra={"level_index": level_index, "frame_dir": frame_dir, "player": analysis.player},
        )
        return proposals

    def _evaluate_branch_round(
        self,
        *,
        level_dir: Path,
        round_index: int,
        branch: BranchPlan,
        completed_episode_actions: list[str],
        level_index: int,
        start_level_analysis: StartLevelAnalysis,
    ) -> dict[str, Any]:
        branch_dir = level_dir / f"round_{round_index:03d}" / _safe_token(branch.branch_id)
        branch_dir.mkdir(parents=True, exist_ok=True)
        full_actions = list(completed_episode_actions) + list(branch.current_level_actions)
        bootstrap_count = len(self.config.bootstrap_actions)
        bootstrap_letters = action_sequence_to_letters(self.config.bootstrap_actions) if self.config.bootstrap_actions else ""
        branch_letters = action_sequence_to_letters(branch.current_level_actions)
        generated_action_letters = action_sequence_to_letters(branch.current_level_actions[bootstrap_count:])
        result = run_branch(
            cfg=self.config,
            seed=self.config.seed,
            branch=branch,
            full_actions=full_actions,
            output_dir=str(branch_dir),
            max_actions_allowed=len(full_actions),
        )
        next_frontier: list[BranchPlan] = []
        if not result.won and not result.level_advanced and not result.done and not result.truncated:
            conversation_scope = self._level_conversation_scope(level_index)
            prompt = build_prompt(
                self.config.in_loop_prompt,
                memory=build_prompt_memory(
                    action_list=branch_letters,
                    bootstrap_action_list=bootstrap_letters,
                    level_index=level_index,
                    player=start_level_analysis.player,
                    branch_object=branch.object_name,
                    branch_sequence=branch_letters,
                    object_action_list=generated_action_letters,
                    start_level_analysis_json=start_level_analysis.to_dict(),
                ),
            )
            raw_path = branch_dir / "in_loop_raw.json"
            record = call_ollama(
                ollama_url=self.config.ollama_url,
                ollama_model=self.config.ollama_model,
                ollama_num_ctx=self.config.ollama_num_ctx,
                system_prompt=self.config.system_prompt,
                prompt=prompt,
                frame_dir=result.frame_dir,
                max_prompt_frames=self.config.max_prompt_frames,
                timeout_sec=self.config.timeout_sec,
                retry_count=self.config.retry_count,
                output_path=str(raw_path),
                conversation_scope=conversation_scope,
                reset_context=False,
                metadata={"stage": "in_loop_prompt", "level_index": level_index, "branch_id": branch.branch_id},
            )
            text = extract_response_text(record)
            proposals = parse_object_action_proposals(text)
            self._write_json(branch_dir / "in_loop_parsed.json", [item.to_dict() for item in proposals])
            self._debug_prompt_round(
                stage="in_loop_prompt",
                prompt=prompt,
                answer_text=text,
                parsed_payload={"Actions": [item.to_dict() for item in proposals]},
                extra={
                    "level_index": level_index,
                    "round_index": round_index,
                    "branch_id": branch.branch_id,
                    "branch_object": branch.object_name,
                    "frame_dir": result.frame_dir,
                },
            )
            next_frontier = [
                BranchPlan(
                    branch_id=f"{branch.branch_id}__{idx:03d}_{_safe_token(item.object_name)}",
                    object_name=item.object_name,
                    current_level_actions=list(branch.current_level_actions) + list(item.actions),
                    source_prompt_stage="in_loop_prompt",
                    parent_branch_id=branch.branch_id,
                    generation=int(branch.generation) + 1,
                )
                for idx, item in enumerate(proposals)
            ]
        return {"branch_result": result, "next_frontier": next_frontier}

    def _build_initial_frontier(
        self,
        *,
        level_index: int,
        bootstrap_actions: list[str],
        proposals: list[ObjectActionProposal],
    ) -> list[BranchPlan]:
        branches: list[BranchPlan] = []
        for idx, item in enumerate(proposals):
            branches.append(
                BranchPlan(
                    branch_id=f"level_{level_index:03d}_obj_{idx:03d}_{_safe_token(item.object_name)}",
                    object_name=item.object_name,
                    current_level_actions=list(bootstrap_actions) + list(item.actions),
                    source_prompt_stage="get_list_of_objects_actions",
                    generation=0,
                )
            )
        return self._dedupe_frontier(branches)

    def _select_frontier(
        self,
        frontier: list[BranchPlan],
        completed_episode_actions: list[str],
        total_actions_used: int,
    ) -> list[BranchPlan]:
        remaining = self.config.max_actions_budget - int(total_actions_used)
        selected: list[BranchPlan] = []
        reserved = 0
        for branch in frontier:
            estimated = len(completed_episode_actions) + len(branch.current_level_actions)
            if estimated <= 0:
                estimated = 1
            if reserved + estimated > remaining:
                continue
            selected.append(branch)
            reserved += estimated
            if len(selected) >= self.config.max_parallel_branches:
                break
        return selected

    def _dedupe_frontier(self, frontier: list[BranchPlan]) -> list[BranchPlan]:
        seen: set[tuple[str, ...]] = set()
        out: list[BranchPlan] = []
        for branch in frontier:
            key = tuple(branch.current_level_actions)
            if key in seen:
                continue
            seen.add(key)
            out.append(branch)
        return out

    def _save_level_solution(self, *, level_dir: Path, level_start: int, level_end: int, actions: list[str]) -> str:
        filename = f"{self.config.env_id}_level_{level_start}-level_{level_end}.txt"
        path = level_dir / filename
        letters = action_sequence_to_letters(actions) if actions else ""
        path.write_text(letters + "\n", encoding="utf-8")
        self._write_json(
            level_dir / f"{path.stem}.json",
            {
                "env_id": self.config.env_id,
                "level_start": int(level_start),
                "level_end": int(level_end),
                "actions": list(actions),
                "letters": letters,
            },
        )
        return str(path)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _level_conversation_scope(self, level_index: int) -> str:
        return f"{self.session_id}:episode_000:level_{int(level_index):03d}"

    def _debug_prompt_round(
        self,
        *,
        stage: str,
        prompt: str,
        answer_text: str,
        parsed_payload: Any,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.debug:
            return
        extra_json = json.dumps(dict(extra or {}), ensure_ascii=False, indent=2)
        parsed_json = json.dumps(parsed_payload, ensure_ascii=False, indent=2)
        message = (
            f"stage={stage}\n"
            f"extra={extra_json}\n"
            f"\nPrompt:\n{str(prompt).rstrip()}\n"
            f"\nAnswer:\n{str(answer_text).rstrip()}\n"
            f"\nParsed TOON:\n{parsed_json}\n"
        )
        self.logger.debug(message)


def _safe_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    return token.strip("_") or "item"
