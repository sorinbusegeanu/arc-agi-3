from __future__ import annotations

from v6.contingency.contingency_learner import ContingencyLearner


class Predictor:
    def __init__(self, learner: ContingencyLearner) -> None:
        self.learner = learner

    def predict(self, context_signature: tuple, action: int) -> int | None:
        distribution = self.learner.distribution(context_signature, action)
        if not distribution:
            return None
        return max(distribution.items(), key=lambda item: (item[1], -item[0]))[0]

    def predict_multi_scale(self, context_signatures: dict[int, tuple], action: int) -> int | None:
        return self.learner.predict(context_signatures, action)
