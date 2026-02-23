from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .actions import build_action
from .controller import Controller
from .critic import Critic
from .env_adapter import EnvAdapter
from .explorer import Explorer
from .goal_detector import is_terminal_win
from .learned_models import ActionRankerModel, LearnedCriticModel, MechanicClassifierModel
from .logger import EpisodeLogger
from .mechanic_inference import MechanicInference
from .memory_episodic import EpisodicMemory
from .observation_parser import parse_observation
from .options import select_tool_and_click
from .planner import Planner
from .safety import SafetyGuard
from .semantic_memory import SemanticMemory
from .state_abstract import canonicalize_state
from .transition_diff import compute_delta
from .types import ControllerContext, EpisodeResult
from .world_model import EmpiricalWorldModel


class ZodAgent:
    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        max_actions: int = 80,
        variant_id: str = "baseline",
        log_dir: str = "zod01/logs",
        episode_id: str | None = None,
        use_ranker: bool = False,
        use_learned_critic: bool = False,
        use_mechanic_classifier: bool = False,
        ranker_model_path: str = "zod01/models/ranker.json",
        critic_model_path: str = "zod01/models/critic.json",
        mechanic_model_path: str = "zod01/models/mechanic.json",
        w_ranker: float = 0.5,
        w_risk: float = 0.5,
        w_safety: float = 1.0,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.max_actions = max_actions
        self.variant_id = variant_id
        self.user_episode_id = episode_id
        self.use_ranker = use_ranker
        self.use_learned_critic = use_learned_critic
        self.use_mechanic_classifier = use_mechanic_classifier

        self.env = EnvAdapter(game_id=game_id, seed=seed)
        self.memory = EpisodicMemory()
        self.model = EmpiricalWorldModel(self.memory)
        self.planner = Planner(self.memory)
        self.safety = SafetyGuard()
        self.critic = Critic()
        self.inference = MechanicInference()
        self.explorer = Explorer(self.memory, self.model)
        ranker_model = (
            ActionRankerModel.load(ranker_model_path)
            if use_ranker and Path(ranker_model_path).exists()
            else None
        )
        learned_critic_model = (
            LearnedCriticModel.load(critic_model_path)
            if use_learned_critic and Path(critic_model_path).exists()
            else None
        )
        mechanic_model = (
            MechanicClassifierModel.load(mechanic_model_path)
            if use_mechanic_classifier and Path(mechanic_model_path).exists()
            else None
        )
        self.controller = Controller(
            self.safety,
            self.critic,
            self.inference,
            use_ranker=use_ranker,
            use_learned_critic=use_learned_critic,
            use_mechanic_classifier=use_mechanic_classifier,
            ranker_model=ranker_model,
            learned_critic_model=learned_critic_model,
            mechanic_model=mechanic_model,
            w_ranker=w_ranker,
            w_risk=w_risk,
            w_safety=w_safety,
        )
        self.logger = EpisodeLogger(out_dir=log_dir)
        self.semantic_memory = SemanticMemory()
        self.goal_hashes: set[str] = set()
        self.option_queue: list[str] = []

        prior = self.semantic_memory.get(self._mechanic_signature())
        if prior is not None:
            self.inference.belief.movement_bias = float(prior.get("movement_bias", 0.0))
            self.inference.belief.interaction_bias = float(
                prior.get("interaction_bias", 0.0)
            )
            self.inference.belief.click_bias = float(prior.get("click_bias", 0.0))

    def _planner_action(self, current_hash: str) -> str | None:
        path = self.planner.plan_to_any_goal(current_hash, self.goal_hashes)
        if len(path) < 2:
            return None
        nxt = path[1]
        for (src, _), edge in self.memory.transitions.items():
            if src == current_hash and edge.dst_hash == nxt:
                return edge.action.name
        return None

    def _mechanic_signature(self) -> str:
        return self.game_id.split("-", 1)[0]

    @staticmethod
    def _raw_obs_hash(obs: Any) -> str:
        payload = str(obs.frame).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def run_episode(self) -> EpisodeResult:
        self.env.reset()
        log_path = self.logger.start(
            self.game_id, self.seed, self.variant_id, episode_id=self.user_episode_id
        )

        obs = parse_observation(self.env.observation)
        state = canonicalize_state(obs)
        self.memory.register_state(state.state_hash)
        self.logger.log(
            {
                "type": "reset",
                "episode_id": self.logger.episode_id,
                "game_id": self.game_id,
                "seed": self.seed,
                "variant_id": self.variant_id,
                "state_hash": state.state_hash,
                "raw_obs_hash": self._raw_obs_hash(self.env.observation),
                "state": state.state,
                "available_actions": list(obs.available_actions),
            }
        )

        won = False
        step_idx = -1
        for step_idx in range(self.max_actions):
            planner_name = self._planner_action(state.state_hash)
            planner_action = build_action(planner_name) if planner_name else None
            if not self.option_queue and "ACTION6" in obs.available_actions:
                # Option seed: interact then center-click when complex action is available.
                self.option_queue = [a.name for a in select_tool_and_click(32, 32)]
            if self.option_queue:
                planner_action = build_action(self.option_queue.pop(0), x=32, y=32)
            explorer_props = self.explorer.propose(state_hash=state.state_hash, available_actions=obs.available_actions)
            ctx = ControllerContext(
                step_idx=step_idx,
                recent_hashes=tuple(self.safety.recent),
                available_actions=obs.available_actions,
            )
            choice, candidates_debug = self.controller.choose(ctx, planner_action, explorer_props)
            result = self.env.step(choice.action)
            if not result.valid_action:
                # Always recover to the first valid action.
                fallback = build_action(obs.available_actions[0] if obs.available_actions else "RESET")
                result = self.env.step(fallback)
                choice = choice.__class__(action=fallback, source="fallback", score=-1.0, tags=("invalid",))

            new_obs = parse_observation(result.observation)
            new_state = canonicalize_state(new_obs)
            delta = compute_delta(state, new_state)
            won = is_terminal_win(new_state)

            self.memory.register_state(new_state.state_hash)
            self.memory.add_transition(
                src_hash=state.state_hash,
                action=choice.action,
                dst_hash=new_state.state_hash,
                delta=delta,
                won=won,
            )
            self.inference.update(choice.action.name, delta)
            self.safety.observe(new_state.state_hash, delta)
            if won:
                self.goal_hashes.add(new_state.state_hash)

            self.logger.log(
                {
                    "type": "step",
                    "episode_id": self.logger.episode_id,
                    "game_id": self.game_id,
                    "seed": self.seed,
                    "variant_id": self.variant_id,
                    "step_idx": step_idx,
                    "state_hash": state.state_hash,
                    "raw_obs_hash": self._raw_obs_hash(result.observation),
                    "available_actions": list(obs.available_actions),
                    "next_state_hash": new_state.state_hash,
                    "chosen_action": choice.action.name,
                    "source": choice.source,
                    "score": choice.score,
                    "use_ranker": self.use_ranker,
                    "use_learned_critic": self.use_learned_critic,
                    "use_mechanic_classifier": self.use_mechanic_classifier,
                    "tags": list(choice.tags),
                    "candidate_debug": candidates_debug,
                    "delta_tokens": list(delta.tags),
                    "delta_kind": "no_op" if delta.no_op else "state_change",
                    "won": won,
                    "terminal": won or (step_idx + 1 >= self.max_actions),
                    "actions_used": step_idx + 1,
                    "delta": asdict(delta),
                }
            )

            state = new_state
            obs = new_obs
            if won:
                break

        self.logger.log(
            {
                "type": "episode_end",
                "episode_id": self.logger.episode_id,
                "game_id": self.game_id,
                "seed": self.seed,
                "variant_id": self.variant_id,
                "won": won,
                "actions_used": max(0, step_idx + 1),
                "unique_states": self.memory.unique_states(),
                "log_path": str(log_path),
            }
        )
        traj_hash = self.logger.close()
        self.semantic_memory.put(
            self._mechanic_signature(),
            {
                "movement_bias": self.inference.belief.movement_bias,
                "interaction_bias": self.inference.belief.interaction_bias,
                "click_bias": self.inference.belief.click_bias,
            },
        )
        return EpisodeResult(
            episode_id=self.logger.episode_id,
            game_id=self.game_id,
            won=won,
            actions=step_idx + 1,
            unique_states=self.memory.unique_states(),
            trajectory_hash=traj_hash,
            log_path=str(log_path),
        )
