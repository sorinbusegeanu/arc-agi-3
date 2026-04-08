from __future__ import annotations

from rl_v1.metrics.metric_keys import TRAINING_LOSS


class TrainingLossMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss_sum = 0.0
        self.total_loss_count = 0

    def update(self, loss_value: float) -> None:
        self.total_loss_sum += float(loss_value)
        self.total_loss_count += 1

    def compute(self) -> dict:
        return {
            TRAINING_LOSS: self.total_loss_sum / self.total_loss_count if self.total_loss_count else 0.0,
        }
