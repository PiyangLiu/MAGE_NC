"""Fresh validation-error greedy support selector.

This module is deliberately self contained.  It does not import any of the
previous ``unified_validation_greedy`` implementations and it never receives
a precomputed support order or a requested final support count.

The same algorithm is used by every case outside Figure 2:

1. standardise the dual coordinates;
2. build a small deterministic coverage prefix (centroid, extrema, then
   farthest-point coverage);
3. evaluate the complete validation pool and identify its worst states;
4. rank unused candidates in the nearest-coordinate neighbourhood by the
   distance plus the candidate's own thermodynamic residual;
5. append a batch (eight by default), retune tau using validation only, and
   stop at the first nested prefix satisfying both validation gates.

The independent test pool is evaluated only after this function returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _as_float(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _relative(a: np.ndarray, b: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


def _scale_coordinates(candidate: Dict[str, np.ndarray], validation: Dict[str, np.ndarray]):
    both = np.vstack((_as_float(candidate["eta"]), _as_float(validation["eta"])))
    center = both.mean(axis=0)
    scale = both.std(axis=0)
    return center, np.where(scale > 1.0e-14, scale, 1.0)


def _score_arrays(phi: np.ndarray, q: np.ndarray, ref: Dict[str, np.ndarray],
                  weights: Sequence[float], composition_weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Equation-S43-style thermodynamic score used for selection.

    Potential error is in the physical Phi units.  Derivative errors are
    normalised by the validation range, so density/energy cannot hide the
    composition error.  The same expression is used for validation states
    and for a candidate's own residual.
    """
    phi_ref = _as_float(ref["phi"])
    q_ref = _as_float(ref["q"])
    q_range = np.ptp(q_ref, axis=0)
    q_range = np.where(q_range > 1.0e-14, q_range, 1.0)
    score = float(weights[0]) * (phi - phi_ref) ** 2
    score += float(weights[1]) * ((q[:, 0] - q_ref[:, 0]) / q_range[0]) ** 2
    if q.shape[1] > 1:
        comp = ((q[:, 1:] - q_ref[:, 1:]) / q_range[None, 1:]) ** 2
        if composition_weights is not None:
            cw = _as_float(composition_weights)
            if cw.size != comp.shape[1]:
                raise ValueError("composition_weights must match q[:, 1:]")
            comp = comp * cw[None, :]
        score += float(weights[2]) * np.mean(comp, axis=1)
    return np.asarray(score, dtype=float)


def evaluate_support(eta_support: np.ndarray, phi_support: np.ndarray,
                     q_support: np.ndarray, eta_query: np.ndarray, tau: float,
                     center: np.ndarray, scale: np.ndarray,
                     chunk_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the finite-support log-sum-exp potential and derivatives."""
    eta_support = _as_float(eta_support)
    phi_support = _as_float(phi_support)
    q_support = _as_float(q_support)
    eta_query = _as_float(eta_query)
    if len(eta_support) == 0 or tau <= 0.0:
        raise ValueError("support must be non-empty and tau must be positive")
    z_support = (eta_support - center) / scale
    z_query = (eta_query - center) / scale
    q_scaled = q_support * scale[None, :]
    intercept = phi_support - np.sum(q_scaled * z_support, axis=1)
    out_phi = np.empty(len(eta_query), dtype=float)
    out_q = np.empty((len(eta_query), q_support.shape[1]), dtype=float)
    for lo in range(0, len(eta_query), max(1, int(chunk_size))):
        hi = min(len(eta_query), lo + max(1, int(chunk_size)))
        logits = (z_query[lo:hi] @ q_scaled.T + intercept[None, :]) / float(tau)
        m = np.max(logits, axis=1, keepdims=True)
        w = np.exp(logits - m)
        den = np.sum(w, axis=1, keepdims=True)
        w /= den
        out_phi[lo:hi] = float(tau) * (m[:, 0] + np.log(den[:, 0]) - np.log(len(eta_support)))
        out_q[lo:hi] = (w @ q_scaled) / scale[None, :]
    return out_phi, out_q


class _IncrementalValidationState:
    """Exact log-sum-exp update for a fixed tau and a growing prefix."""

    def __init__(self, eta_query: np.ndarray, tau: float,
                 center: np.ndarray, scale: np.ndarray):
        self.query = _as_float(eta_query)
        self.tau = float(tau)
        self.center = center
        self.scale = scale
        self.z_query = (self.query - center) / scale
        self.m = np.full(len(self.query), -np.inf)
        self.den = np.zeros(len(self.query))
        self.num: Optional[np.ndarray] = None
        self.count = 0

    def append(self, eta: np.ndarray, phi: np.ndarray, q: np.ndarray) -> None:
        eta, phi, q = _as_float(eta), _as_float(phi), _as_float(q)
        q_scaled = q * self.scale[None, :]
        z_support = (eta - self.center) / self.scale
        intercept = phi - np.sum(q_scaled * z_support, axis=1)
        logits = (self.z_query @ q_scaled.T + intercept[None, :]) / self.tau
        new_m = np.maximum(self.m, np.max(logits, axis=1))
        old_scale = np.exp(self.m - new_m)
        new_scale = np.exp(logits - new_m[:, None])
        if self.num is None:
            self.num = np.zeros((len(self.query), q.shape[1]), dtype=float)
        self.den = old_scale * self.den + np.sum(new_scale, axis=1)
        self.num = old_scale[:, None] * self.num + new_scale @ q_scaled
        self.m = new_m
        self.count += len(q)

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.num is None or self.count <= 0:
            raise ValueError("incremental state is empty")
        phi = self.tau * (self.m + np.log(self.den) - np.log(self.count))
        q = (self.num / self.den[:, None]) / self.scale[None, :]
        return phi, q


def _initial_coverage(eta: np.ndarray, center: np.ndarray, scale: np.ndarray,
                      coverage_count: int) -> list[int]:
    """Create the deterministic common coverage prefix."""
    z = (_as_float(eta) - center) / scale
    ids: list[int] = []
    ids.append(int(np.argmin(np.sum(z * z, axis=1))))
    for dim in range(z.shape[1]):
        ids.extend((int(np.argmin(z[:, dim])), int(np.argmax(z[:, dim]))))
    ids = list(dict.fromkeys(ids))
    if coverage_count <= len(ids):
        return ids[:coverage_count]
    chosen = set(ids)
    nearest = np.full(len(z), np.inf)
    for idx in ids:
        nearest = np.minimum(nearest, np.sum((z - z[idx]) ** 2, axis=1))
    while len(ids) < min(int(coverage_count), len(z)):
        nearest[list(chosen)] = -np.inf
        idx = int(np.argmax(nearest))
        ids.append(idx)
        chosen.add(idx)
        nearest = np.minimum(nearest, np.sum((z - z[idx]) ** 2, axis=1))
    return ids


def _tau_choice(order: Sequence[int], candidate: Dict[str, np.ndarray],
                validation: Dict[str, np.ndarray], taus: Sequence[float],
                center: np.ndarray, scale: np.ndarray,
                weights: Sequence[float], composition_weights: Optional[Sequence[float]],
                chunk_size: int) -> Tuple[float, float, float]:
    best: Optional[Tuple[Tuple[float, float], float, float]] = None
    ids = np.asarray(order, dtype=int)
    for tau in taus:
        phi, q = evaluate_support(candidate["eta"][ids], candidate["phi"][ids],
                                  candidate["q"][ids], validation["eta"], float(tau),
                                  center, scale, chunk_size)
        scores = _score_arrays(phi, q, validation, weights, composition_weights)
        key = (float(np.mean(scores)), float(np.max(scores)))
        # Mean is primary; max breaks ties.  A larger tau wins exact ties for
        # deterministic reproducibility without using any non-validation data.
        tie = -float(tau)
        if best is None or (key, tie) < (best[0], -best[1]):
            best = (key, float(tau), float(np.max(scores)))
    assert best is not None
    return best[1], best[0][0], best[2]


@dataclass
class SelectionResult:
    support_order: np.ndarray
    selected_m: int
    selected_tau: float
    target_met: bool
    history: list[dict]
    center: np.ndarray
    scale: np.ndarray
    thresholds: dict
    algorithm: dict

    def as_dict(self) -> dict:
        return {
            "support_order": self.support_order.astype(int).tolist(),
            "selected_M": int(self.selected_m),
            "selected_tau": float(self.selected_tau),
            "target_met": bool(self.target_met),
            "validation_history": self.history,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "validation_thresholds": self.thresholds,
            "algorithm": self.algorithm,
        }


def select_minimum_validation_greedy(
    candidate: Dict[str, np.ndarray], validation: Dict[str, np.ndarray],
    *, max_m: Optional[int], tau_grid: Sequence[float], thresholds: Dict[str, float],
    weights: Sequence[float] = (1.0, 1.0, 1.0),
    composition_weights: Optional[Sequence[float]] = None,
    batch_add: int = 8, coverage_count: int = 8, neighbourhood_size: int = 256,
    worst_state_count: int = 16, candidate_residual_weight: float = 1.0,
    seed: int = 0, chunk_size: int = 4096,
) -> SelectionResult:
    """Select the first validation-feasible nested prefix.

    ``max_m`` is only a hard safety bound for a finite candidate pool.  It is
    never interpreted as the reported M.  The returned M is the first history
    entry for which both declared validation gates pass.
    """
    del seed  # the construction is deterministic; retaining it in metadata is useful
    required = ("mean", "max")
    if any(k not in thresholds for k in required):
        raise ValueError("thresholds must contain mean and max")
    n = len(candidate["eta"])
    if n == 0:
        raise ValueError("candidate pool is empty")
    if len(validation["eta"]) == 0:
        raise ValueError("validation pool is empty")
    max_m = n if max_m is None else min(int(max_m), n)
    if max_m <= 0 or batch_add <= 0:
        raise ValueError("max_m and batch_add must be positive")
    center, scale = _scale_coordinates(candidate, validation)
    order = _initial_coverage(candidate["eta"], center, scale,
                              max(int(coverage_count), 1))[:max_m]
    selected = set(order)
    history: list[dict] = []
    tau = float(tau_grid[0])
    incremental = (_IncrementalValidationState(validation["eta"], tau, center, scale)
                   if len(tau_grid) == 1 else None)
    if incremental is not None:
        ids0 = np.asarray(order, dtype=int)
        incremental.append(candidate["eta"][ids0], candidate["phi"][ids0], candidate["q"][ids0])
    selected_m = len(order)
    target_met = False

    while True:
        if incremental is None:
            tau, mean_score, max_score = _tau_choice(
                order, candidate, validation, tau_grid, center, scale,
                weights, composition_weights, chunk_size)
            phi_v, q_v = evaluate_support(candidate["eta"][order], candidate["phi"][order],
                                           candidate["q"][order], validation["eta"], tau,
                                           center, scale, chunk_size)
        else:
            phi_v, q_v = incremental.predict()
        scores = _score_arrays(phi_v, q_v, validation, weights, composition_weights)
        mean_score, max_score = float(np.mean(scores)), float(np.max(scores))
        met = mean_score <= float(thresholds["mean"]) and max_score <= float(thresholds["max"])
        history.append({"M": int(len(order)), "tau": float(tau),
                        "validation_mean_score": mean_score,
                        "validation_max_score": max_score,
                        "worst_validation_index": int(np.argmax(scores)),
                        "target_met": bool(met)})
        if met:
            target_met = True
            selected_m = len(order)
            break
        if len(order) >= max_m or len(order) >= n:
            selected_m = len(order)
            break

        # The worst validation states drive the next neighbourhood.  Candidate
        # residuals are evaluated with the current prefix, never with test or
        # coexistence data.
        worst = np.argsort(scores)[-max(1, int(worst_state_count)):]
        zc = (candidate["eta"] - center) / scale
        zv = (validation["eta"][worst] - center) / scale
        distance = np.min(((zc[:, None, :] - zv[None, :, :]) ** 2).sum(axis=2), axis=1)
        distance[list(selected)] = np.inf
        finite = np.flatnonzero(np.isfinite(distance))
        if len(finite) == 0:
            selected_m = len(order)
            break
        nearest = finite[np.argsort(distance[finite])[:min(int(neighbourhood_size), len(finite))]]
        phi_c, q_c = evaluate_support(candidate["eta"][order], candidate["phi"][order],
                                       candidate["q"][order], candidate["eta"][nearest], tau,
                                       center, scale, chunk_size)
        own_score = _score_arrays(phi_c, q_c,
                                  {"phi": candidate["phi"][nearest], "q": candidate["q"][nearest]},
                                  weights, composition_weights)
        d_norm = distance[nearest] / max(float(np.max(distance[nearest])), 1.0e-30)
        r_norm = own_score / max(float(np.max(own_score)), 1.0e-30)
        # A candidate with a large current residual is thermodynamically
        # informative, so its residual term is a benefit (1-r_norm), not a
        # penalty.  This is the precise candidate-own-residual definition.
        rank = d_norm + float(candidate_residual_weight) * (1.0 - r_norm)
        ranked = nearest[np.argsort(rank, kind="mergesort")]
        add = [int(i) for i in ranked[:min(int(batch_add), max_m - len(order))]]
        if not add:
            selected_m = len(order)
            break
        order.extend(add)
        selected.update(add)
        if incremental is not None:
            incremental.append(candidate["eta"][add], candidate["phi"][add], candidate["q"][add])

    # The first-feasible-prefix invariant is checked before returning.  This
    # catches accidental post-processing of M or a non-nested support order.
    if target_met:
        feasible = [h["M"] for h in history if h["target_met"]]
        if selected_m != min(feasible):
            raise AssertionError("selected M is not the first feasible prefix")
    if len(order) != len(set(order)):
        raise AssertionError("support order contains duplicates")
    metadata = {
        "name": "validation_error_greedy_neighbourhood",
        "coordinate_scaling": "mean/std over candidate+validation eta",
        "initial_coverage_count": int(coverage_count),
        "neighbourhood_size": int(neighbourhood_size),
        "batch_add": int(batch_add),
        "worst_state_count": int(worst_state_count),
        "candidate_residual": "same score expression evaluated at candidate state using current prefix",
        "candidate_residual_weight": float(candidate_residual_weight),
        "tau_selection": "minimum validation mean score, max score tie-break",
        "selection_data": ["candidate", "validation"],
        "excluded_from_selection": ["independent test", "coexistence", "KKT", "flow trajectory"],
        "postprocessing_selected_M": False,
    }
    return SelectionResult(np.asarray(order, dtype=int), selected_m, tau,
                           target_met, history, center, scale,
                           {"mean": float(thresholds["mean"]), "max": float(thresholds["max"])},
                           metadata)


def evaluate_metrics(candidate: Dict[str, np.ndarray], query: Dict[str, np.ndarray],
                     order: Sequence[int], tau: float, center: np.ndarray,
                     scale: np.ndarray, weights: Sequence[float] = (1., 1., 1.),
                     composition_weights: Optional[Sequence[float]] = None) -> dict:
    ids = np.asarray(order, dtype=int)
    phi, q = evaluate_support(candidate["eta"][ids], candidate["phi"][ids],
                               candidate["q"][ids], query["eta"], tau, center, scale)
    score = _score_arrays(phi, q, query, weights, composition_weights)
    return {
        "M": int(len(ids)), "tau": float(tau),
        "score_mean": float(np.mean(score)), "score_max": float(np.max(score)),
        "phi_relative_mae": float(np.mean(_relative(phi, query["phi"]))),
        "gradient_relative_mae": float(np.mean(_relative(q, query["q"]))),
    }


def save_json(path, payload: dict) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
