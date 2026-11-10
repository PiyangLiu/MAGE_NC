"""Isometric embedding of a physical CH4/n-C4 PR Massieu branch.

This is a *feature-space* embedding of an unchanged binary system, not a
ten-species EOS with eight species at zero mole fraction.  Isometries preserve
exact support planes and the validation objective, hence support counts should
be invariant up to roundoff and deterministic tie-breaking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mage_core import greedy_select, predict

R = 8.31446261815324
T = 300.0
P_REF = 1e5
TC = np.array([190.564, 425.125])
PC = np.array([4.5992e6, 3.796e6])
OMEGA = np.array([0.01142, 0.2002])


def pr_state(p: float, x_ch4: float, phase: str):
    x = np.array([x_ch4, 1.-x_ch4])
    kappa = 0.37464+1.54226*OMEGA-0.26992*OMEGA**2
    alpha = (1.+kappa*(1.-np.sqrt(T/TC)))**2
    ai = .45724*R**2*TC**2/PC*alpha
    bi = .07780*R*TC/PC
    aij = np.sqrt(np.outer(ai, ai))
    am = x @ aij @ x
    bm = x @ bi
    aa = am*p/(R*T)**2
    bb = bm*p/(R*T)
    roots = np.roots([1., -(1.-bb), aa-3*bb**2-2*bb,
                      -(aa*bb-bb**2-bb**3)])
    good = sorted(float(z.real) for z in roots
                  if abs(z.imag)<1e-9 and z.real>bb+1e-9)
    if not good:
        raise ValueError("No physically admissible PR compressibility root")
    z = good[0] if phase == "liquid" else good[-1]
    term = np.log((z+(1+np.sqrt(2))*bb)/(z+(1-np.sqrt(2))*bb))
    log_phi = (bi/bm)*(z-1.)-np.log(z-bb) - (
        aa/(2*np.sqrt(2)*bb) * (2*(aij @ x)/am-bi/bm) * term)
    eta = np.log(x*p/P_REF)+log_phi
    massieu = p/(R*T)
    gradient = x*p/(z*R*T)
    return eta, massieu, gradient, z


def sample_phase(phase: str, n_candidate: int, n_validation: int,
                 n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    # Noncritical, compact local domains of mechanically stable PR roots.
    # Both branches are near, but not exactly on, the binary coexistence path.
    if phase == "liquid":
        pmin, pmax, xmin, xmax = 1.2e6, 3.6e6, .055, .22
    else:
        pmin, pmax, xmin, xmax = 1.8e6, 3.2e6, .84, .94

    def generate(n):
        eta, phi, grad, zvals = [], [], [], []
        for p, x in zip(rng.uniform(pmin, pmax, n),
                        rng.uniform(xmin, xmax, n)):
            e, f, g, z = pr_state(p, x, phase)
            eta.append(e); phi.append(f); grad.append(g); zvals.append(z)
        return np.array(eta), np.array(phi), np.array(grad), np.array(zvals)

    return generate(n_candidate), generate(n_validation), generate(n_test)


def orthogonal_embedding(dim: int):
    if dim == 2:
        return np.eye(2)
    rng = np.random.default_rng(7321+dim)
    raw = rng.normal(size=(dim, 2))
    q, _ = np.linalg.qr(raw)
    return q[:, :2]


def derivative_audit(p, x, phase):
    """Check the Legendre identity d(p/RT)/d(chemical eta)=component density."""
    dp, dx = 10., 1e-6
    ep, fp, _, _ = pr_state(p+dp, x, phase)
    em, fm, _, _ = pr_state(p-dp, x, phase)
    xp, fp2, _, _ = pr_state(p, x+dx, phase)
    xm, fm2, _, _ = pr_state(p, x-dx, phase)
    J = np.column_stack(((ep-em)/(2*dp), (xp-xm)/(2*dx)))
    dphi = np.array([(fp-fm)/(2*dp), (fp2-fm2)/(2*dx)])
    numerical = np.linalg.solve(J.T, dphi)
    _, _, analytic, z = pr_state(p, x, phase)
    return {"Z": float(z), "max_gradient_relative_discrepancy": float(
        np.max(np.abs(numerical-analytic)/np.maximum(np.abs(analytic), 1e-9)))}


def run(outdir: Path, tolerance: float, tau: float):
    outdir.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    audits = {"liquid": derivative_audit(2.5e6, .13, "liquid"),
              "vapour": derivative_audit(2.5e6, .89, "vapour")}
    for phase, seed in [("liquid", 202610), ("vapour", 202611)]:
        cand, val, test = sample_phase(phase if phase=="liquid" else "vapour",
                                       460, 280, 400, seed)
        ec, pc, gc, _ = cand
        ev, pv, gv, _ = val
        et, pt, gt, _ = test
        eta_center = ec.mean(axis=0)
        eta_scale = ec.std(axis=0)
        phi_scale = 1000.0

        def standardize(e, p, g):
            return ((e-eta_center)/eta_scale,
                    p/phi_scale, g*eta_scale/phi_scale)

        xc, fc, gc = standardize(ec, pc, gc)
        xv, fv, gv = standardize(ev, pv, gv)
        xt, ft, gt = standardize(et, pt, gt)
        phase_detail = []
        for ambient in [2, 3, 5, 8, 10]:
            Q = orthogonal_embedding(ambient)
            # X'=QX and gradient'=Q gradient, exactly preserving dot products.
            xce, xve, xte = xc@Q.T, xv@Q.T, xt@Q.T
            gce, gve = gc@Q.T, gv@Q.T
            inds, trace = greedy_select(xce, fc, gce, xve, fv, gve,
                                        tau=tau, max_m=240, tolerance=tolerance)
            fp, gp = predict(xte, xce[inds], fc[inds], gce[inds], tau)
            # Pull the ambient derivative back to the two original species.
            gp2 = gp@Q
            grad_rel = np.linalg.norm(gp2-gt, axis=1)/np.maximum(
                np.linalg.norm(gt, axis=1), .01)
            abs_phi = np.abs(fp-ft)*phi_scale
            rows.append({"phase": phase, "ambient_Nc": ambient,
                         "intrinsic_dimension": 2, "M": len(inds),
                         "threshold": tolerance,
                         "tau_dimensionless": tau,
                         "test_phi_MAE_mol_m3": float(abs_phi.mean()),
                         "test_phi_max_mol_m3": float(abs_phi.max()),
                         "test_gradient_rel_mean": float(grad_rel.mean()),
                         "test_gradient_rel_max": float(grad_rel.max()),
                         "validation_max_error": trace[-1]["max_validation_error"],
                         "stopping_threshold_reached": bool(trace[-1]["max_validation_error"]<=tolerance)})
            phase_detail.append({"ambient_Nc": ambient, "indices": inds.tolist(),
                                 "trace": trace})
        details[phase] = phase_detail

    with (outdir/"embedding_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (outdir/"embedding_trace.json").write_text(json.dumps(
        {"PR_derivative_audits": audits, "selections": details}, indent=2))
    for phase in ["liquid", "vapour"]:
        rr = [r for r in rows if r["phase"]==phase]
        assert len(set(r["M"] for r in rr))==1, "Isometric count differs!"
        assert all(np.array_equal(np.array(details[phase][0]["indices"]),
                                  np.array(d["indices"]))
                   for d in details[phase][1:]), "Isometric indices differ!"
    print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tolerance", type=float, default=.05)
    ap.add_argument("--tau", type=float, default=.0006)
    args = ap.parse_args()
    run(Path(args.outdir), args.tolerance, args.tau)
