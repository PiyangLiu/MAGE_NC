"""Small deterministic unit tests for the fresh selector."""
from __future__ import annotations

import numpy as np

from greedy_selector import select_minimum_validation_greedy


def toy_pool(n=96):
    x = np.linspace(-1.0, 1.0, n)[:, None]
    eta = np.c_[x, x * x]
    phi = 0.2 * x[:, 0] ** 2
    q = np.c_[x[:, 0], 1.0 + x[:, 0]]
    return {"eta": eta, "phi": phi, "q": q}


def main():
    candidate = toy_pool()
    validation = toy_pool(64)
    result = select_minimum_validation_greedy(
        candidate, validation, max_m=64, tau_grid=(0.05,),
        thresholds={"mean": 1.0, "max": 2.0},
        batch_add=8, coverage_count=8, neighbourhood_size=16,
        candidate_residual_weight=1.0)
    assert len(result.support_order) == len(set(result.support_order.tolist()))
    assert result.selected_m == next(h["M"] for h in result.history if h["target_met"])
    assert result.selected_m % 8 == 0
    print("selector smoke test passed", result.selected_m)


if __name__ == "__main__":
    main()
