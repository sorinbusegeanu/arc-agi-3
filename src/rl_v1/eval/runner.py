from rl_v1.eval.evaluator import Evaluator


def evaluate_model(cfg, model, mode: str = "policy_only"):
    cfg.acting.mode = "planner_act" if mode == "planner_act" else "policy_only"
    return Evaluator(cfg, model).evaluate()
