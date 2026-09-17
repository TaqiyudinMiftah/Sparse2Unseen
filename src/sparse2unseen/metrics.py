from __future__ import annotations

import math


class CountMetrics:
    def __init__(self) -> None:
        self.abs_error = 0.0
        self.sq_error = 0.0
        self.n = 0

    def update(self, pred_count: float, true_count: float) -> None:
        err = pred_count - true_count
        self.abs_error += abs(err)
        self.sq_error += err * err
        self.n += 1

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {"mae": float("nan"), "rmse": float("nan")}
        return {
            "mae": self.abs_error / self.n,
            "rmse": math.sqrt(self.sq_error / self.n),
        }
