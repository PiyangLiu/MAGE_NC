"""Finite-support, normalized log-sum-exp Massieu representation.

All quantities here are nondimensional.  For the PR experiment, the physical
potential and gradients are transformed to dimensionless coordinates first.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax


def predict(x: np.ndarray, xs: np.ndarray, fs: np.ndarray,
            gs: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Return potential and gradient of the *normalized* tangent LSE.

    The division by M is essential; omitting it changes phase pressures.
    """
    x = np.atleast_2d(np.asarray(x, float))
    intercept = fs - np.einsum("ij,ij->i", gs, xs)
    planes = x @ gs.T + intercept
    logits = planes / tau
    f = tau * (logsumexp(logits, axis=1) - np.log(len(xs)))
    grad = softmax(logits, axis=1) @ gs
    return f, grad


def greedy_select(xc: np.ndarray, fc: np.ndarray, gc: np.ndarray,
                  xv: np.ndarray, fv: np.ndarray, gv: np.ndarray,
                  tau: float, max_m: int, tolerance: float,
                  min_m: int = 8) -> tuple[np.ndarray, list[dict]]:
    """Deterministic error-directed selection, independent of test data.

    The selection error is max(|delta potential|/f_range,
    ||delta gradient||/max(norm(reference gradient), 0.1*gradient_range)).
    Each step adds the nearest unused candidate to the worst validation query.
    Strictly monotone stop is not assumed for normalized LSE.
    """
    xc, xv = np.asarray(xc), np.asarray(xv)
    f_range = max(float(np.ptp(fv)), 1e-12)
    g_range = max(float(np.linalg.norm(np.ptp(gv, axis=0))), 1e-12)
    center = np.mean(xc, axis=0)
    start = int(np.argmin(np.sum((xc-center)**2, axis=1)))
    selected = [start]
    used = np.zeros(len(xc), dtype=bool)
    used[start] = True
    trace = []
    for m in range(1, min(max_m, len(xc)) + 1):
        inds = np.asarray(selected, dtype=int)
        fp, gp = predict(xv, xc[inds], fc[inds], gc[inds], tau)
        ef = np.abs(fp-fv)/f_range
        eg = np.linalg.norm(gp-gv, axis=1)/np.maximum(np.linalg.norm(gv, axis=1), 0.1*g_range)
        metric = np.maximum(ef, eg)
        trace.append({"M": m, "max_validation_error": float(metric.max()),
                      "mean_validation_error": float(metric.mean()),
                      "max_potential_error_normalized": float(ef.max()),
                      "max_gradient_error_relative": float(eg.max())})
        if m >= min_m and metric.max() <= tolerance:
            return inds, trace
        if m == max_m or m == len(xc):
            return inds, trace
        worst = xv[int(np.argmax(metric))]
        distances = np.sum((xc-worst)**2, axis=1)
        distances[used] = np.inf
        next_index = int(np.argmin(distances))
        used[next_index] = True
        selected.append(next_index)
    raise RuntimeError("unreachable")
