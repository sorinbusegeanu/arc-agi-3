from __future__ import annotations

from .feature_extractor import candidate_features
from .learned_models import ActionRankerModel, LearnedCriticModel, MechanicClassifierModel
from .critic import Critic
from .mechanic_inference import MechanicInference
from .safety import SafetyGuard
from .types import ActionProposal, ControllerContext, NormalizedAction


class Controller:
    def __init__(
        self,
        safety: SafetyGuard,
        critic: Critic,
        inference: MechanicInference,
        use_ranker: bool = False,
        use_learned_critic: bool = False,
        use_mechanic_classifier: bool = False,
        ranker_model: ActionRankerModel | None = None,
        learned_critic_model: LearnedCriticModel | None = None,
        mechanic_model: MechanicClassifierModel | None = None,
        w_ranker: float = 0.5,
        w_risk: float = 0.5,
        w_safety: float = 1.0,
    ) -> None:
        self.safety = safety
        self.critic = critic
        self.inference = inference
        self.use_ranker = use_ranker
        self.use_learned_critic = use_learned_critic
        self.use_mechanic_classifier = use_mechanic_classifier
        self.ranker_model = ranker_model
        self.learned_critic_model = learned_critic_model
        self.mechanic_model = mechanic_model
        self.w_ranker = w_ranker
        self.w_risk = w_risk
        self.w_safety = w_safety

    def choose(
        self,
        ctx: ControllerContext,
        planner_action: NormalizedAction | None,
        explorer_proposals: list[ActionProposal],
    ) -> tuple[ActionProposal, list[dict[str, object]]]:
        candidates: list[ActionProposal] = []

        if planner_action is not None:
            candidates.append(ActionProposal(action=planner_action, source="planner", score=1.0))

        candidates.extend(explorer_proposals[:6])
        if not candidates:
            fallback = NormalizedAction(name=ctx.available_actions[0] if ctx.available_actions else "RESET")
            return (
                ActionProposal(action=fallback, source="fallback", score=0.0, tags=("no-candidates",)),
                [],
            )

        best: ActionProposal | None = None
        best_score = float("-inf")
        debug_candidates: list[dict[str, object]] = []
        for cand in candidates:
            safety_penalty, safety_tags = self.safety.penalties(cand.action)
            critic_penalty, critic_tags = self.critic.score_penalty(cand)
            mech_bias = self.inference.bias_for(cand.action.name)
            if self.use_mechanic_classifier and self.mechanic_model is not None:
                mech_bias += self.mechanic_model.bias(cand.action.name)

            heuristic_score = cand.score + mech_bias - critic_penalty
            features = candidate_features(cand, ctx, heuristic_score, mech_bias, safety_penalty)

            rank_score = 0.0
            if self.use_ranker and self.ranker_model is not None:
                rank_score = self.ranker_model.score(features, cand.action.name)

            risk_score = critic_penalty
            if self.use_learned_critic and self.learned_critic_model is not None:
                risk_score += self.learned_critic_model.risk(cand.action.name)

            score = heuristic_score + self.w_ranker * rank_score - self.w_risk * risk_score - self.w_safety * safety_penalty
            tags = cand.tags + safety_tags + critic_tags
            merged = ActionProposal(action=cand.action, source=cand.source, score=score, tags=tags)
            debug_candidates.append(
                {
                    "action": cand.action.name,
                    "source": cand.source,
                    "features": features,
                    "heuristic_score": heuristic_score,
                    "rank_score": rank_score,
                    "risk_score": risk_score,
                    "safety_penalty": safety_penalty,
                    "final_score": score,
                    "tags": list(tags),
                }
            )
            if score > best_score:
                best_score = score
                best = merged

        assert best is not None
        if best.action.name not in ctx.available_actions and ctx.available_actions:
            return (
                ActionProposal(
                    action=NormalizedAction(name=ctx.available_actions[0]),
                    source="fallback",
                    score=-1.0,
                    tags=("invalid-filtered",),
                ),
                debug_candidates,
            )
        return best, debug_candidates
