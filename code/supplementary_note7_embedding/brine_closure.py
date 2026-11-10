"""Controlled *model* test of non-cubic, electrolyte-like dual closure.

IMPORTANT: This is not a calibrated Pitzer/CPA/HEOS model and is not the
unavailable H2-brine model underlying Supplementary Note 5.1 of the manuscript.
It isolates expensive nonlinear thermodynamic state inversion from finite-
support online phase-branch evaluation at exactly the same constitutive model.

The dimensionless aqueous negative-entropy/free-energy generator is strictly
convex: h(q) = sum q_i(log(q_i/q0_i)-1) + q^T A q/2 + gamma*(q_w+q_s)^3/3.
The cross terms A[H2,NaCl] and the cubic salt-water contribution generate a
salinity effect absent from plain PR with no electrolyte extension. The gas
branch is a distinct ideal gas H2/H2O potential with zero salt density.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import root

from mage_core import predict

Q0 = np.array([.06, 1.0, .13])
A = np.array([[2.4, .12, 1.25], [.12, .55, .40], [1.25, .40, 2.35]])
GAMMA = .28
GAS_Q0 = np.array([.32, .012])
GAS_SHIFT = np.array([.35, .15])
Q_MIN = np.array([.015, .75, .045])
Q_MAX = np.array([.145, 1.18, .295])


def h_liq(q):
    q = np.asarray(q)
    return (np.sum(q*(np.log(q/Q0)-1), axis=-1)
            + .5*np.einsum("...i,ij,...j->...", q, A, q)
            + GAMMA/3.*(q[..., 1]+q[..., 2])**3)


def eta_liq(q):
    q = np.asarray(q)
    eta = np.log(q/Q0) + q@A
    out = np.array(eta, copy=True)
    assoc = GAMMA*(q[..., 1]+q[..., 2])**2
    out[..., 1] += assoc
    out[..., 2] += assoc
    return out


def hessian_liq(q):
    h = np.diag(1./q) + A
    assoc = 2*GAMMA*(q[1]+q[2])
    h[np.ix_([1, 2], [1, 2])] += assoc
    return h


def phi_liq_from_q(q):
    e = eta_liq(q)
    return np.einsum("...i,...i->...", e, q)-h_liq(q)


def solve_liq(eta, guess=None):
    """Optimized Newton with exact Jacobian and positivity via log densities."""
    if guess is None:
        guess = np.clip(Q0*np.exp(np.clip(eta-np.array([.4, 1., .6]), -8, 8)),
                        1e-4, 4.)
    logq0 = np.log(guess)

    def residual(logq):
        return eta_liq(np.exp(logq))-eta

    def jac(logq):
        q = np.exp(logq)
        return hessian_liq(q)*q[None, :]

    result = root(residual, logq0, jac=jac, method="hybr", tol=2e-11)
    q = np.exp(result.x)
    if np.linalg.norm(residual(result.x), ord=np.inf)>1e-8:
        raise RuntimeError(f"Aqueous inversion did not converge: {result.message}")
    return float(eta@q-h_liq(q)), q, result.nfev


def gas_from_eta(eta, offset):
    gas_q = GAS_Q0*np.exp(np.asarray(eta)[..., :2]-GAS_SHIFT)
    phi = np.sum(gas_q, axis=-1)-offset
    grad = np.zeros(np.asarray(eta).shape)
    grad[..., :2] = gas_q
    return phi, grad


def create_data(seed=20261105):
    rng = np.random.default_rng(seed)
    def get(n):
        q = rng.uniform(Q_MIN, Q_MAX, size=(n, 3))
        eta = eta_liq(q)
        phi = phi_liq_from_q(q)
        return eta, phi, q
    cand, val, test = get(1800), get(500), get(600)
    e, phi, _ = cand
    gphi, _ = gas_from_eta(e, 0.)
    offset = float(np.median(gphi-phi))
    return cand, val, test, offset


def performance_metrics(f, g, fref, gref, scale):
    return {
        "mean_potential_absolute": float(np.mean(np.abs(f-fref))*scale),
        "max_potential_absolute": float(np.max(np.abs(f-fref))*scale),
        "mean_gradient_relative": float(np.mean(np.linalg.norm(g-gref, axis=1)/
            np.maximum(np.linalg.norm(gref, axis=1), .05))),
        "max_gradient_relative": float(np.max(np.linalg.norm(g-gref, axis=1)/
            np.maximum(np.linalg.norm(gref, axis=1), .05)))
    }


def time_queries(func, etas, repeats=5):
    # Run identical query order in each repeat with an independent warm-up.
    for e in etas[:25]:
        func(e)
    times = []
    for rep in range(repeats):
        start = time.perf_counter_ns()
        for e in etas:
            func(e)
        times.append((time.perf_counter_ns()-start)/len(etas)/1e3)
    return {"median_us_per_query": float(np.median(times)),
            "repeats_us_per_query": [float(t) for t in times]}


def run(outdir: Path, ml=384, mg=256, tau=.0015):
    outdir.mkdir(parents=True, exist_ok=True)
    start_offline = time.perf_counter()
    cand, val, test, offset = create_data()
    ec, fc, gc = cand
    ev, fv, gv = val
    et, ft, gt = test
    center, scale_eta = ec.mean(axis=0), ec.std(axis=0)
    # The same affine coordinate transform is used for both phases.
    def standardized(e, f, g):
        return (e-center)/scale_eta, f, g*scale_eta
    xc, flc, glc = standardized(ec, fc, gc)
    xv, flv, glv = standardized(ev, fv, gv)
    xt, flt, glt = standardized(et, ft, gt)
    fgc, ggc = gas_from_eta(ec, offset)
    fgv, ggv = gas_from_eta(ev, offset)
    fgt, ggt = gas_from_eta(et, offset)
    _, fgc, ggc = standardized(ec, fgc, ggc)
    _, fgv, ggv = standardized(ev, fgv, ggv)
    _, fgt, ggt = standardized(et, fgt, ggt)

    # Fixed, reproducible coverage design: greedy farthest-point in the same
    # dual-coordinate metric for each branch. Counts are declared a priori.
    def coverage_indices(coords, m):
        indices = [int(np.argmin(np.sum((coords-coords.mean(axis=0))**2, axis=1)))]
        dmin = np.sum((coords-coords[indices[0]])**2, axis=1)
        for _ in range(m-1):
            i = int(np.argmax(dmin))
            indices.append(i)
            dmin = np.minimum(dmin, np.sum((coords-coords[i])**2, axis=1))
            dmin[i] = 0
        return np.array(indices, int)
    il = coverage_indices(xc, ml)
    ig = coverage_indices(xc, mg)
    offline_s = time.perf_counter()-start_offline

    l_val = predict(xv, xc[il], flc[il], glc[il], tau)
    g_val = predict(xv, xc[ig], fgc[ig], ggc[ig], tau)
    l_test = predict(xt, xc[il], flc[il], glc[il], tau)
    g_test = predict(xt, xc[ig], fgc[ig], ggc[ig], tau)
    # Undo normalized coordinate gradient scaling.
    g_l_test = l_test[1]/scale_eta
    g_g_test = g_test[1]/scale_eta
    l_metrics = performance_metrics(l_test[0], g_l_test, ft, gt, 1.)
    g_metrics = performance_metrics(g_test[0], g_g_test, fgt, ggt/scale_eta, 1.)

    # The phase decision is made at equal dual chemical potentials, without
    # mixing physical models or defining a flash at imposed p,z.
    reference_phase = (ft >= fgt).astype(int)
    mage_phase = (l_test[0] >= g_test[0]).astype(int)
    phase_correct = float(np.mean(reference_phase == mage_phase))
    margin = np.abs(ft-fgt)
    robust = margin > .025
    robust_accuracy = float(np.mean(reference_phase[robust] == mage_phase[robust]))

    # Audit the independently implemented nonlinear reference inversion.
    audit_e = et[:60]
    inversion_abs = []
    nfev = []
    for e, q_truth in zip(audit_e, gt[:60]):
        ff, qq, it = solve_liq(e)
        inversion_abs.append(float(np.max(np.abs(qq-q_truth))))
        nfev.append(it)

    timing_e = et[60:560]
    def reference_closure(e):
        p_l, q_l, _ = solve_liq(e)
        p_g, q_g = gas_from_eta(e, offset)
        return (p_l, q_l) if p_l>=p_g else (p_g, q_g)

    def compiled_mage(indices, f, g):
        gg = np.ascontiguousarray(g[indices])
        b = np.ascontiguousarray(f[indices]-np.einsum(
            "ij,ij->i", gg, xc[indices]))
        normalization = np.log(len(indices))

        def evaluate(x):
            planes = gg@x+b
            max_plane = planes.max()
            weights = np.exp((planes-max_plane)/tau)
            den = np.sum(weights)
            value = max_plane+tau*(np.log(den)-normalization)
            grad = weights@gg/den
            return value, grad
        return evaluate

    eval_l = compiled_mage(il, flc, glc)
    eval_g = compiled_mage(ig, fgc, ggc)

    def mage_closure(e):
        xx = (e-center)/scale_eta
        p_l, q_l = eval_l(xx)
        p_g, q_g = eval_g(xx)
        return (p_l, q_l/scale_eta) if p_l>=p_g else (p_g, q_g/scale_eta)
    timed_reference = time_queries(reference_closure, timing_e)
    timed_mage = time_queries(mage_closure, timing_e)
    speedup = timed_reference["median_us_per_query"]/timed_mage["median_us_per_query"]
    delta = (timed_reference["median_us_per_query"]-
             timed_mage["median_us_per_query"])
    break_even = (offline_s*1e6/delta) if delta>0 else None

    results = {"test": "dimensionless_nonideal_aqueous_dual_closure",
               "not_full_PR_Pitzer_or_eCPA": True,
               "is_not_fixed_p_z_flash": True,
               "liquid_supports": ml, "gas_supports": mg,
               "tau": tau, "candidate": len(ec), "validation": len(ev),
               "heldout_test": len(et), "timed_queries": len(timing_e),
               "liquid_accuracy": l_metrics, "gas_accuracy": g_metrics,
               "phase_classification_fraction": phase_correct,
               "robust_margin_count": int(robust.sum()),
               "robust_margin_classification_fraction": robust_accuracy,
               "reference_inversion_max_density_residual": max(inversion_abs),
               "reference_inversion_mean_function_evaluations": float(np.mean(nfev)),
               "reference": timed_reference, "mage": timed_mage,
               "online_speedup_reference_over_mage": speedup,
               "offline_construction_seconds": offline_s,
               "estimated_amortization_queries": break_even,
               "python": platform.python_version(), "numpy": np.__version__,
               "scipy": scipy.__version__, "platform": platform.platform()}
    (outdir/"brine_results.json").write_text(json.dumps(results, indent=2))
    with (outdir/"brine_holdout.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["eta_H2", "eta_H2O", "eta_NaCl", "phi_l_ref",
                     "phi_l_mage", "phi_g_ref", "phi_g_mage", "label_ref", "label_mage"])
        for i in range(len(et)):
            wr.writerow([*et[i], ft[i], l_test[0][i], fgt[i],
                         g_test[0][i], reference_phase[i], mage_phase[i]])
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--liquid-supports", type=int, default=384)
    parser.add_argument("--gas-supports", type=int, default=256)
    parser.add_argument("--tau", type=float, default=.0015)
    args = parser.parse_args()
    run(Path(args.outdir), args.liquid_supports, args.gas_supports, args.tau)
