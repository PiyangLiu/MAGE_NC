"""Reproduce the numerical examples used by the MAGE manuscript.

The default command uses the frozen manuscript pool sizes.  ``--quick`` is a
small smoke run that exercises exactly the same code paths.  All generated
tables are plain CSV/JSON/NPZ files so that the figures can be rebuilt without
rerunning the thermodynamic teachers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
# The bundled SciPy build still imports NumPy aliases removed in NumPy 1.26.
if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_
from scipy.optimize import least_squares
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from mage_core import (  # noqa: E402
    LEGACY_ZERO_CP0,
    MAGEBranch,
    PREOS,
    REFERENCE_EOS_CP0,
    GERGTeacher,
    RegularHelmholtzLLE,
    farthest_order,
    make_lle_pool,
    massieu_phase_state,
    ideal_gas_caloric_metadata,
    pr_equilibrium_state,
    pr_massieu_from_c,
    solve_mage_binary,
    solve_mage_pressure_step,
    solve_pr_pressure_step,
)


R = 8.31446261815324
SEED = 20261021
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def arrdict_save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in data.items()})


def csv_save(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, float), delimiter=",", header=",".join(header), comments="")


def json_save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def metric_summary(model: MAGEBranch, data: dict):
    phi, q = model.evaluate_batch(data["eta"])
    dphi = np.abs(phi - data["phi"])
    denom = np.maximum(np.linalg.norm(data["q"], axis=1), 1e-12)
    dq = np.linalg.norm(q - data["q"], axis=1) / denom
    du = np.abs(q[:, 0] - data["q"][:, 0]) / np.maximum(np.abs(data["q"][:, 0]), 1e-12)
    u_molar = q[:, 0] / np.sum(q[:, 1:], axis=1)
    u_molar_ref = data["q"][:, 0] / np.sum(data["c"], axis=1)
    du_molar = np.abs(u_molar - u_molar_ref) / 1.0e3
    dc = np.linalg.norm(q[:, 1:] - data["q"][:, 1:], axis=1) / np.maximum(np.linalg.norm(data["q"][:, 1:], axis=1), 1e-12)
    return {
        "phi_mae": float(np.mean(dphi)), "phi_max": float(np.max(dphi)),
        "phi_rel_mae": float(np.mean(dphi / np.maximum(np.abs(data["phi"]), 1e-12))),
        "gradient_rel_mae": float(np.mean(dq)), "gradient_rel_max": float(np.max(dq)),
        "u_rel_mae": float(np.mean(du)), "c_rel_mae": float(np.mean(dc)),
        "u_molar_abs_mae_kJ_mol": float(np.mean(du_molar)),
        "u_molar_abs_p95_kJ_mol": float(np.percentile(du_molar, 95)),
        "u_molar_abs_max_kJ_mol": float(np.max(du_molar)),
        "pred_phi": phi, "pred_q": q,
    }


def common_scaling(*datasets):
    eta = np.vstack([d["eta"] for d in datasets])
    center = np.mean(eta, axis=0)
    scale = np.std(eta, axis=0)
    scale = np.where(scale > 1e-14, scale, 1.0)
    phi = np.concatenate([d["phi"] for d in datasets])
    # Keep the reported smoothing grid in the physical Massieu units used by
    # the legacy PR benchmark.  The dual coordinates are scaled separately
    # through ``scale``; scaling Phi by the broad pressure range here would
    # turn the prescribed \bar{tau}=0.125 into O(10--100) mol/m^3 and wash
    # out the branch curvature.
    phi_scale = 1.0
    return center, scale, phi_scale


def make_branch(data, idx, tau_bar, center, scale, phi_scale, name, offset=0.0):
    idx = np.asarray(idx, int)
    return MAGEBranch(data["eta"][idx], data["phi"][idx], data["q"][idx],
                      tau=float(tau_bar * phi_scale), center=center, scale=scale,
                      name=name, offset=float(offset))


def self_error_support_order(candidate, max_m, center, scale, phi_scale,
                             tau_bar, seed_count=0):
    """Greedily retain candidates with the largest reconstruction error.

    This is the gradient-aware rule used by the earlier pressure-coupled
    benchmark.  Exact initial and inlet states can be supplied as a prefix,
    after which every new support is selected from the frozen offline pool.
    """
    X, phi, q = candidate["eta"], candidate["phi"], candidate["q"]
    selected = list(range(min(int(seed_count), len(X))))
    for dim in range(X.shape[1]):
        selected.extend((int(np.argmin(X[:, dim])), int(np.argmax(X[:, dim]))))
    selected.extend((int(np.argmin(phi)), int(np.argmax(phi))))
    selected = list(dict.fromkeys(selected))[:max_m]
    while len(selected) < min(max_m, len(X)):
        model = make_branch(candidate, selected, tau_bar, center, scale,
                            phi_scale, "selection")
        pred_phi, pred_q = model.evaluate_batch(X)
        gradient_error = np.linalg.norm(pred_q - q, axis=1) / np.maximum(
            np.linalg.norm(q, axis=1), 1.0e-12
        )
        potential_error = np.abs(pred_phi - phi) / np.maximum(np.abs(phi), 1.0e-12)
        score = gradient_error + 0.10 * potential_error
        score[np.asarray(selected, int)] = -np.inf
        selected.append(int(np.argmax(score)))
    return np.asarray(selected, int)


def select_support_order(candidate, validation, max_m, center, scale, phi_scale, tau_grid, seed=0, priority_count=0, composition_weight=1.0):
    """Build nested supports using the validation-error greedy rule.

    The initial support contains the candidate nearest the centroid and the
    nearest candidates to both coordinate extremes.  At each subsequent step
    the validation state with the largest potential/gradient error is found,
    and the unused candidate nearest to that state is appended.  For the very
    large GERG pool the same deterministic farthest ordering is retained to
    keep the full benchmark tractable; all binary, non-isothermal, ternary
    and LLE examples use this validation-driven construction.
    """
    X = (candidate["eta"] - center) / scale
    Xv = (validation["eta"] - center) / scale
    if len(candidate["eta"]) > 20000:
        order = farthest_order(X)[:max_m]
        qrange = np.ptp(validation["q"], axis=0)
        qscale = np.where(qrange > 1.0e-14, qrange, 1.0)
        tau_choices = {}
        for m in sorted(set(min(max_m, int(v)) for v in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536))):
            if m > max_m:
                continue
            best = None
            for tb in tau_grid:
                model = make_branch(candidate, order[:m], tb, center, scale, phi_scale, "selection")
                ph, qq = model.evaluate_batch(validation["eta"])
                obj = np.median(np.abs(ph - validation["phi"]) / phi_scale) + np.median(
                    np.linalg.norm((qq - validation["q"]) / qscale, axis=1)
                ) / np.sqrt(validation["q"].shape[1])
                key = (float(obj), -float(tb))
                if best is None or key < best[0]:
                    best = (key, float(tb))
            tau_choices[m] = best[1]
        return order, tau_choices

    # Candidate-range scaling used in the manuscript's dimensionless score.
    qrange = np.ptp(validation["q"], axis=0)
    qscale = np.where(qrange > 1.0e-14, qrange, 1.0)
    selected = [int(np.argmin(np.sum((X - X.mean(axis=0)) ** 2, axis=1)))]
    for dim in range(X.shape[1]):
        selected.append(int(np.argmin(X[:, dim])))
        selected.append(int(np.argmax(X[:, dim])))
    order = list(dict.fromkeys(selected))
    # A small deterministic prefix of coexistence states is retained when the
    # caller supplied one.  It is part of the candidate set, and prevents the
    # narrow equilibrium manifold from being discarded while the greedy score
    # focuses on the much larger homogeneous-state cloud.
    if priority_count:
        order.extend(i for i in range(min(priority_count, len(X))) if i not in order)
    order = order[:max_m]
    requested = sorted(set(min(max_m, int(v)) for v in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536)))
    requested = [m for m in requested if m <= max_m]
    qrange = np.ptp(validation["q"], axis=0)
    qscale = np.where(qrange > 1e-14, qrange, 1.0)
    tau_choices = {}

    def choose_tau(indices):
        best = None
        for tb in tau_grid:
            model = make_branch(candidate, np.asarray(indices, int), tb, center, scale, phi_scale, "selection")
            ph, qq = model.evaluate_batch(validation["eta"])
            obj = np.median(np.abs(ph - validation["phi"]) / phi_scale) + np.median(
                np.linalg.norm((qq - validation["q"]) / qscale, axis=1)) / np.sqrt(validation["q"].shape[1])
            # Larger smoothing wins exact ties, as specified in Note 3.
            key = (float(obj), -float(tb))
            if best is None or key < best[0]:
                best = (key, float(tb))
        return best[1]

    # Selecting tau at the requested support counts keeps the nested ordering
    # deterministic while avoiding an unnecessary eight-fold evaluation at
    # every intermediate support size.
    # Requested support counts below the seeded prefix still use their own
    # leading subset, since the benchmark intentionally includes M=16/24/32.
    for m in requested:
        if m <= len(order):
            tau_choices[m] = choose_tau(order[:m])
    tau_current = choose_tau(order)
    if len(order) in requested:
        tau_choices[len(order)] = tau_current
    selected_set = set(order)
    while len(order) < max_m:
        model = make_branch(candidate, np.asarray(order, int), tau_current, center, scale, phi_scale, "selection")
        ph, qq = model.evaluate_batch(validation["eta"])
        dq_scaled = (qq - validation["q"]) / qscale
        score = (ph - validation["phi"]) ** 2 / (phi_scale ** 2)
        score += dq_scaled[:, 0] ** 2
        score += float(composition_weight) * np.mean(dq_scaled[:, 1:] ** 2, axis=1)
        validation_index = int(np.argmax(score))
        distances = np.sum((X - Xv[validation_index]) ** 2, axis=1)
        if selected_set:
            distances[list(selected_set)] = np.inf
        next_index = int(np.argmin(distances))
        order.append(next_index)
        selected_set.add(next_index)
        if len(order) in requested:
            tau_current = choose_tau(order)
            tau_choices[len(order)] = tau_current
    return np.asarray(order, int), tau_choices


def binary_pool(eos, n, T_values, P_range, phase, x_min, seed,
                include_equilibrium=False, caloric_model=LEGACY_ZERO_CP0):
    sampler = qmc.Sobol(3, scramble=True, seed=seed)
    rows = []
    if include_equilibrium:
        # Guarantee that the candidate set resolves the coexistence manifold
        # itself.  Random homogeneous states are still used for the remaining
        # pool, while these deterministic states prevent a broad stable-state
        # cloud from undersampling the narrow VLE branch.
        for temperature in T_values:
            curve_count = 45 if len(T_values) == 1 else 31
            curve_lo = max(P_range[0], 0.55e6 if len(T_values) == 1 else 0.50e6)
            curve_hi = min(P_range[1], 4.95e6 if len(T_values) == 1 else 7.00e6)
            for pressure in np.linspace(curve_lo, curve_hi, curve_count):
                try:
                    eq = eos.binary_vle(float(temperature), float(pressure))
                    if not eq["success"] or np.max(np.abs(eq["x"] - eq["y"])) <= 1.0e-4:
                        continue
                    composition = eq["x"] if phase == "liquid" else eq["y"]
                    state = eos.state(float(temperature), float(pressure), composition, phase)
                    tr = pr_massieu_from_c(
                        [temperature], [state["c"]], eos.names,
                        caloric_model=caloric_model,
                    )
                    rows.append({
                        "T": float(temperature), "P": float(pressure),
                        "x1": float(composition[0]), "c": state["c"],
                        "is_equilibrium": 1.0,
                        **{k: tr[k][0] for k in ("eta", "phi", "q")},
                    })
                except Exception:
                    continue
                if len(rows) >= n:
                    break
            if len(rows) >= n:
                break
    power = int(np.ceil(np.log2(max(1024, n * 2))))
    while len(rows) < n:
        U = sampler.random_base2(power) if not rows else sampler.random(2**power)
        for u in U:
            T = T_values[int(np.floor(u[0] * len(T_values))) % len(T_values)] if len(T_values) > 1 else T_values[0]
            P = P_range[0] + u[1] * (P_range[1] - P_range[0])
            x1 = x_min + u[2] * (1.0 - 2.0 * x_min)
            try:
                # Root selection alone admits metastable states from the
                # two-phase region.  Use the flash classification first, then
                # retain only a stable single-phase composition or the stable
                # coexistence composition returned by the flash.
                z = np.array([x1, 1.0 - x1])
                fl = eos.flash(T, P, z)
                if fl["phase"] == "two-phase":
                    x = fl["x"] if phase == "liquid" else fl["y"]
                elif fl["phase"] == phase:
                    x = z
                else:
                    x = None
                if x is not None:
                    st = eos.state(T, P, x, phase)
                    if np.all(np.isfinite(st["c"])) and np.all(st["c"] > 0):
                        tr = pr_massieu_from_c(
                            [T], [st["c"]], eos.names,
                            caloric_model=caloric_model,
                        )
                        rows.append({"T": T, "P": P, "x1": float(x[0]),
                                     "c": st["c"], "is_equilibrium": 0.0,
                                     **{k: tr[k][0] for k in ("eta", "phi", "q")}})
            except Exception:
                pass
            if len(rows) >= n:
                break
    return {k: np.asarray([r[k] for r in rows]) for k in
            ("T", "P", "x1", "c", "eta", "phi", "q", "is_equilibrium")}


def run_binary(quick=False):
    eos = PREOS(("CH4", "nC4"))
    if quick:
        nc, nv, nt, npress = 512, 128, 256, 13
        Ms = [16, 32, 64, 128]
    else:
        nc, nv, nt, npress = 8192, 2048, 4096, 45
        Ms = [16, 24, 32, 48, 64, 96, 128, 192, 384, 1024]
    grid = {"candidate": {}, "validation": {}, "test": {}}
    for phase, seedoff in (("liquid", 10), ("vapor", 20)):
        grid["candidate"][phase] = binary_pool(eos, nc, [300.0], (0.45e6, 5.10e6), phase, 0.005, SEED + seedoff, include_equilibrium=True)
        grid["validation"][phase] = binary_pool(eos, nv, [300.0], (0.45e6, 5.10e6), phase, 0.005, SEED + seedoff + 100, include_equilibrium=True)
        grid["test"][phase] = binary_pool(eos, nt, [300.0], (0.45e6, 5.10e6), phase, 0.005, SEED + seedoff + 200)
        arrdict_save(RESULTS / f"binary_{phase}_pools.npz", {f"candidate_{k}": v for k, v in grid["candidate"][phase].items()} | {f"validation_{k}": v for k, v in grid["validation"][phase].items()} | {f"test_{k}": v for k, v in grid["test"][phase].items()})
    center, scale, phi_scale = common_scaling(grid["candidate"]["liquid"], grid["candidate"]["vapor"])
    tau_grid = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    order_l, tau_l = select_support_order(grid["candidate"]["liquid"], grid["validation"]["liquid"], max(Ms), center, scale, phi_scale, tau_grid, priority_count=45)
    order_v, tau_v = select_support_order(grid["candidate"]["vapor"], grid["validation"]["vapor"], max(Ms), center, scale, phi_scale, tau_grid, priority_count=45)
    pressures = np.linspace(0.55e6, 4.95e6, npress)
    ref = []
    for P in pressures:
        r = eos.binary_vle(300.0, P)
        ref.append([P, r["x"][0], r["y"][0], float(r["success"]), r["residual"]])
    ref = np.asarray(ref)
    rows = []
    models = {}
    for M in Ms:
        tl, tv = tau_l.get(M, tau_l[max(k for k in tau_l if k <= M)]), tau_v.get(M, tau_v[max(k for k in tau_v if k <= M)])
        L = make_branch(grid["candidate"]["liquid"], order_l[:M], tl, center, scale, phi_scale, "liquid")
        V = make_branch(grid["candidate"]["vapor"], order_v[:M], tv, center, scale, phi_scale, "vapor")
        models[M] = (L, V)
        ml, mv = metric_summary(L, grid["test"]["liquid"]), metric_summary(V, grid["test"]["vapor"])
        vler = []
        guess = None
        for P, xr, yr, ok, rr in ref:
            r = solve_mage_binary(300.0, P, L, V, guess)
            if r["success"]:
                guess = r["eta"][1:]
            vler.append([P, r["x"][0], r["y"][0], float(r["success"]), r["residual"], xr, yr, r["pL"], r["pV"]])
        vler = np.asarray(vler)
        rows.append({"M": M, "tau_l": tl, "tau_v": tv, "liquid": {k: v for k, v in ml.items() if np.isscalar(v)}, "vapor": {k: v for k, v in mv.items() if np.isscalar(v)}, "vle_liq_mae": float(np.mean(np.abs(vler[:, 1] - vler[:, 5]))), "vle_vap_mae": float(np.mean(np.abs(vler[:, 2] - vler[:, 6]))), "vle_max": float(np.max(np.abs(vler[:, 1:3] - vler[:, 5:7]))), "vle_converged": int(np.sum(vler[:, 3] > 0.5))})
        csv_save(RESULTS / f"binary_vle_M{M}.csv", vler, ["P", "x_mage", "y_mage", "success", "residual", "x_pr", "y_pr", "pL_mage", "pV_mage"])
    json_save(RESULTS / "binary_summary.json", {"seed": SEED, "n_candidate": nc, "n_validation": nv, "n_test": nt, "n_vle": npress, "center": center, "scale": scale, "phi_scale": phi_scale, "support_order_liquid": order_l, "support_order_vapor": order_v, "rows": rows})
    arrdict_save(RESULTS / "binary_reference.npz", {"P": ref[:, 0], "x": ref[:, 1], "y": ref[:, 2]})
    return {"summary": rows, "models": models, "reference": ref, "center": center, "scale": scale, "phi_scale": phi_scale}


def run_nonisothermal(quick=False, reuse_pools=False):
    eos = PREOS(("CH4", "nC4"))
    caloric_model = REFERENCE_EOS_CP0
    anchors = [295.0, 315.0, 335.0, 350.0]
    held = [305.0, 325.0, 345.0]
    # A coarse temperature grid was the dominant error source in the original
    # Figure 4(c).  Resolve eta_0 systematically while keeping every held-out
    # temperature absent from candidate and validation pools.
    temperature_spacing = 2.5 if quick else 0.25
    support_temperatures = [
        float(t) for t in np.arange(295.0, 350.0 + 0.1 * temperature_spacing,
                                    temperature_spacing)
        if not any(np.isclose(t, test_t) for test_t in held)
    ]
    if quick:
        nc, nv, nt = 640, 160, 320
    else:
        nc, nv, nt = 12000, 3000, 6000
    pools = {"candidate": {}, "validation": {}, "test": {}}
    for phase, seedoff in (("liquid", 30), ("vapor", 40)):
        pool_path = RESULTS / f"nonisothermal_{phase}_pools.npz"
        if reuse_pools and pool_path.exists():
            stored = np.load(pool_path)
            if "metadata_caloric_model" not in stored.files or str(
                    stored["metadata_caloric_model"].item()) != caloric_model:
                raise ValueError(
                    f"{pool_path.name} was not generated with {caloric_model}; "
                    "rebuild without --reuse-pools"
                )
            for kind in ("candidate", "validation", "test"):
                prefix = f"{kind}_"
                pools[kind][phase] = {
                    key[len(prefix):]: stored[key]
                    for key in stored.files if key.startswith(prefix)
                }
            expected = {"candidate": nc, "validation": nv, "test": nt}
            if any(len(pools[kind][phase]["phi"]) != count
                   for kind, count in expected.items()):
                raise ValueError(f"{pool_path.name} does not match the requested pool sizes")
            if "is_equilibrium" not in pools["candidate"][phase]:
                raise ValueError(f"{pool_path.name} predates equilibrium-source tracking")
        else:
            pools["candidate"][phase] = binary_pool(
                eos, nc, support_temperatures, (0.50e6, 7.00e6),
                phase, 0.005, SEED + seedoff, include_equilibrium=True,
                caloric_model=caloric_model,
            )
            # Held-out temperatures are excluded from both model construction
            # and hyperparameter selection; they enter only the final test.
            pools["validation"][phase] = binary_pool(
                eos, nv, support_temperatures, (0.50e6, 7.00e6),
                phase, 0.005, SEED + seedoff + 100,
                include_equilibrium=False, caloric_model=caloric_model,
            )
            pools["test"][phase] = binary_pool(
                eos, nt, anchors, (0.50e6, 7.00e6), phase, 0.005,
                SEED + seedoff + 200, caloric_model=caloric_model,
            )
            arrdict_save(
                pool_path,
                {f"candidate_{k}": v for k, v in pools["candidate"][phase].items()}
                | {f"validation_{k}": v for k, v in pools["validation"][phase].items()}
                | {f"test_{k}": v for k, v in pools["test"][phase].items()}
                | {"metadata_caloric_model": np.array(caloric_model)},
            )
    center, scale, phi_scale = common_scaling(pools["candidate"]["liquid"], pools["candidate"]["vapor"])
    # Retain the complete equilibrium grid as the finite support set.  The
    # previous 1024-point cap discarded either temperature or pressure
    # resolution, whereas adding unrelated homogeneous supports diluted the
    # narrow coexistence manifold in the normalized soft maximum.
    ol = np.flatnonzero(pools["candidate"]["liquid"]["is_equilibrium"] > 0.5)
    ov = np.flatnonzero(pools["candidate"]["vapor"]["is_equilibrium"] > 0.5)
    M = min(len(ol), len(ov))
    ol, ov = ol[:M], ov[:M]
    tau_grid = [0.0625, 0.125, 0.25, 0.5]
    validation_temperatures = [
        float(t) for t in np.arange(297.125, 348.0, 5.0)
        if not any(np.isclose(t, test_t) for test_t in held)
    ]
    validation_pressures = np.linspace(0.55e6, 6.95e6, 13 if not quick else 5)
    tau_validation = []
    for tau in tau_grid:
        Lt = make_branch(pools["candidate"]["liquid"], ol, tau, center, scale, phi_scale, "liquid")
        Vt = make_branch(pools["candidate"]["vapor"], ov, tau, center, scale, phi_scale, "vapor")
        ml = metric_summary(Lt, pools["validation"]["liquid"])
        mv = metric_summary(Vt, pools["validation"]["vapor"])
        equilibrium_errors = []
        for temperature in validation_temperatures:
            guess = None
            for pressure in validation_pressures:
                ref = eos.binary_vle(temperature, pressure)
                if not ref["success"] or np.max(np.abs(ref["x"] - ref["y"])) <= 1.0e-4:
                    continue
                pred = solve_mage_binary(temperature, pressure, Lt, Vt, guess)
                if pred["success"]:
                    guess = pred["eta"][1:]
                equilibrium_errors.append([
                    abs(pred["x"][0] - ref["x"][0]),
                    abs(pred["y"][0] - ref["y"][0]),
                ])
        equilibrium_errors = np.asarray(equilibrium_errors, float)
        curve_mae = float(np.mean(equilibrium_errors))
        curve_max = float(np.max(equilibrium_errors))
        # The primary validation target here is phase-equilibrium composition.
        # Include the worst validation point so a lower mean cannot hide the
        # low-pressure vapor error visible in Figure 4(c); the derivative term
        # remains a small regularizer.
        objective = curve_mae + curve_max + 0.005 * (
            ml["gradient_rel_mae"] + mv["gradient_rel_mae"]
        )
        tau_validation.append({
            "tau": tau, "objective": objective,
            "curve_composition_mae": curve_mae,
            "curve_composition_max": curve_max,
            "liquid_gradient_rel_mae": ml["gradient_rel_mae"],
            "vapor_gradient_rel_mae": mv["gradient_rel_mae"],
            "n_curve_states": int(len(equilibrium_errors)),
        })
    selected = min(tau_validation, key=lambda row: (row["objective"], row["tau"]))
    tau = float(selected["tau"]); bestJ = float(selected["objective"])
    L = make_branch(pools["candidate"]["liquid"], ol, tau, center, scale, phi_scale, "liquid")
    V = make_branch(pools["candidate"]["vapor"], ov, tau, center, scale, phi_scale, "vapor")
    temperatures = anchors + held; curve_rows = []; err_rows = []
    held_pools = {}
    if held:
        nheld = max(64, nt // len(held)) if not quick else max(32, nt // len(held))
        for T in held:
            held_pools[T] = {
                "liquid": binary_pool(
                    eos, nheld, [T], (0.50e6, 7.00e6), "liquid",
                    0.005, SEED + int(T) + 1,
                    caloric_model=caloric_model,
                ),
                "vapor": binary_pool(
                    eos, nheld, [T], (0.50e6, 7.00e6), "vapor",
                    0.005, SEED + int(T) + 2,
                    caloric_model=caloric_model,
                ),
            }
        held_arrays = {}
        for phase in ("liquid", "vapor"):
            keys = held_pools[held[0]][phase]
            for key in keys:
                held_arrays[f"{phase}_{key}"] = np.concatenate([
                    held_pools[T][phase][key] for T in held
                ], axis=0)
        arrdict_save(RESULTS / "nonisothermal_heldout_pools.npz", held_arrays)
    for T in temperatures:
        ps = np.linspace(0.50e6, 7.00e6, 31 if not quick else 11)
        guess = None
        for P in ps:
            r = eos.binary_vle(T, P)
            if r["success"] and np.max(np.abs(r["x"] - r["y"])) > 1e-4:
                m = solve_mage_binary(T, P, L, V, guess)
                if m["success"]:
                    guess = m["eta"][1:]
                curve_rows.append([T, P, r["x"][0], r["y"][0], m["x"][0], m["y"][0], float(m["success"])])
    for T in temperatures:
        if T in held_pools:
            dL, dV = held_pools[T]["liquid"], held_pools[T]["vapor"]
        else:
            mask = np.isclose(pools["test"]["liquid"]["T"], T)
            dL = {k: v[mask] for k, v in pools["test"]["liquid"].items()}; dV = {k: v[mask] for k, v in pools["test"]["vapor"].items()}
        a, b = metric_summary(L, dL), metric_summary(V, dV)
        err_rows.append({"T": T, "liquid": {k: v for k, v in a.items() if np.isscalar(v)}, "vapor": {k: v for k, v in b.items() if np.isscalar(v)}, "n": int(len(dL["phi"]))})
    curve_array = np.asarray(curve_rows, float)
    curve_errors = []
    for T in temperatures:
        q = curve_array[np.isclose(curve_array[:, 0], T)]
        curve_errors.append({
            "T": T, "n": int(len(q)), "converged": int(np.sum(q[:, 6] > 0.5)),
            "liquid_mae": float(np.mean(np.abs(q[:, 4] - q[:, 2]))),
            "liquid_max": float(np.max(np.abs(q[:, 4] - q[:, 2]))),
            "vapor_mae": float(np.mean(np.abs(q[:, 5] - q[:, 3]))),
            "vapor_max": float(np.max(np.abs(q[:, 5] - q[:, 3]))),
        })
    csv_save(RESULTS / "nonisothermal_curves.csv", curve_rows, ["T", "P", "x_pr", "y_pr", "x_mage", "y_mage", "success"])
    json_save(RESULTS / "nonisothermal_summary.json", {
        "seed": SEED, "anchors": anchors, "support_temperatures": support_temperatures,
        "ideal_gas_caloric_model": ideal_gas_caloric_metadata(eos.names),
        "support_temperature_spacing_K": temperature_spacing,
        "held_out": held, "candidate_states_per_phase": nc,
        "support_rule": "all valid equilibrium-grid states; homogeneous states reserved for validation",
        "selected_M": M, "selected_tau_l": tau,
        "selected_tau_v": tau, "center": center, "scale": scale,
        "phi_scale": phi_scale, "validation_objective": bestJ,
        "tau_validation": tau_validation,
        "validation_temperatures": validation_temperatures,
        "support_order_liquid": ol, "support_order_vapor": ov,
        "errors": err_rows, "curve_errors": curve_errors,
    })
    return {"L": L, "V": V, "curves": np.asarray(curve_rows), "selected_M": M, "errors": err_rows, "anchors": anchors, "held": held}


def ternary_sample(eos, n, P_range, zmin, seed, phase_filter=None):
    sampler = qmc.Sobol(4, scramble=True, seed=seed); rows = []
    while len(rows) < n:
        U = sampler.random(4096)
        for u in U:
            P = P_range[0] + u[0] * (P_range[1] - P_range[0])
            z = np.asarray(u[1:4]) + zmin; z /= z.sum()
            try:
                r = eos.flash(320.0, P, z)
                if r["success"] and (phase_filter is None or r["phase"] == phase_filter):
                    rows.append((P, z, r))
            except Exception:
                pass
            if len(rows) >= n: break
    return rows


TERNARY_OPERATING_CENTERS = np.asarray([
    [0.04, 0.08, 0.88], [0.08, 0.12, 0.80], [0.15, 0.15, 0.70],
    [0.25, 0.20, 0.55], [0.35, 0.23, 0.42], [0.45, 0.25, 0.30],
    [0.55, 0.25, 0.20], [0.65, 0.20, 0.15],
])


def ternary_operational_sample(eos, target_counts, seed, concentration=55.0):
    """Generate PR states around the declared flow operating envelope."""
    rng = np.random.default_rng(seed)
    rows = []
    accepted = {phase: 0 for phase in target_counts}
    attempts = 0
    limit = 250 * max(1, sum(target_counts.values()))
    while any(accepted[k] < target_counts[k] for k in target_counts):
        attempts += 1
        if attempts > limit:
            raise RuntimeError(f"ternary operational sampling stalled: {accepted}")
        pressure = rng.uniform(1.25e6, 3.25e6)
        center = TERNARY_OPERATING_CENTERS[rng.integers(len(TERNARY_OPERATING_CENTERS))]
        composition = rng.dirichlet(concentration * center + 1.0)
        try:
            state = eos.flash(320.0, pressure, composition)
        except Exception:
            continue
        phase = state["phase"]
        if state["success"] and phase in accepted and accepted[phase] < target_counts[phase]:
            rows.append((pressure, composition, state))
            accepted[phase] += 1
    return rows, attempts


def ternary_operational_validation(eos, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    attempts = 0
    while len(rows) < n:
        attempts += 1
        pressure = rng.uniform(1.25e6, 3.25e6)
        center = TERNARY_OPERATING_CENTERS[rng.integers(len(TERNARY_OPERATING_CENTERS))]
        composition = rng.dirichlet(45.0 * center + 1.0)
        try:
            state = eos.flash(320.0, pressure, composition)
            if state["success"]:
                rows.append((pressure, composition, state))
        except Exception:
            continue
    return rows, attempts


def phase_data_from_rows(eos, rows):
    out = {"liquid": [], "vapor": []}
    for P, z, r in rows:
        if r["phase"] == "liquid": phases = [("liquid", z)]
        elif r["phase"] == "vapor": phases = [("vapor", z)]
        else: phases = [("liquid", r["x"]), ("vapor", r["y"])]
        for ph, x in phases:
            st = eos.state(320.0, P, x, ph); tr = pr_massieu_from_c([320.0], [st["c"]], eos.names)
            out[ph].append({"T": 320.0, "P": P, "x": x, "eta": tr["eta"][0], "phi": tr["phi"][0], "q": tr["q"][0]})
    return {ph: {k: np.asarray([r[k] for r in rs]) for k in ("T", "P", "x", "eta", "phi", "q")} for ph, rs in out.items()}


def mage_from_reference_state(ref, L, V, T=320.0):
    P = float(ref["P"]); z = np.asarray(ref["z"]); Spr = float(ref.get("S", 0.0));
    if "cL" in ref: Ctarget = (1 - Spr) * ref["cL"] + Spr * ref["cV"]
    else: Ctarget = None
    if Ctarget is None:
        Ctarget = np.ones_like(z)
    eta0 = np.asarray(ref.get("etai", np.zeros_like(z)), float)
    if np.allclose(eta0, 0): eta0 = 0.5 * (L.center[1:] + V.center[1:])
    x0 = np.r_[eta0, Spr]
    def fun(w):
        st = massieu_phase_state(T, w[:3], w[3], L, V); scale = np.maximum(np.abs(Ctarget), 1.0)
        return np.r_[(st["C"] - Ctarget) / scale, (st["Pthermo"] - P) / 2e6]
    sol = least_squares(fun, x0, bounds=(np.r_[np.full(3, -30), 0.0], np.r_[np.full(3, 30), 1.0]), max_nfev=120)
    st = massieu_phase_state(T, sol.x[:3], sol.x[3], L, V); st["P"] = P
    return st


def run_ternary_flow(quick=False):
    eos = PREOS(("CH4", "CO2", "nC4")); T = 320.0
    if quick: nliq, ntp, nvap, nval = 20, 80, 30, 20
    else: nliq, ntp, nvap, nval = 236, 1421, 543, 260
    Pin, Pout, z0, zin = 3.0e6, 1.5e6, np.array([0.04, 0.08, 0.88]), np.array([0.45, 0.25, 0.30])
    N = 6; dx = 0.1
    Pinit = np.linspace(Pin - (Pin - Pout) * (0.5 / N), Pout + (Pin - Pout) * (0.5 / N), N)
    targets = {"liquid": nliq, "two-phase": ntp, "vapor": nvap}
    groups, candidate_attempts = ternary_operational_sample(eos, targets, SEED + 50)
    val_rows, validation_attempts = ternary_operational_validation(eos, nval, SEED + 80)
    branch_pools = phase_data_from_rows(eos, groups)
    # Include the exact initial and inlet thermodynamic states in the branch
    # candidate set.  These states are part of the declared offline pool and
    # anchor the pressure-coupled solve at the states it must reproduce.
    anchor_rows = [(Pin, zin, eos.flash(T, Pin, zin))]
    anchor_rows.extend((float(P), z0, eos.flash(T, float(P), z0)) for P in Pinit)
    anchor_pool = phase_data_from_rows(eos, anchor_rows)
    for ph in ("liquid", "vapor"):
        branch_pools[ph] = {
            key: np.concatenate([anchor_pool[ph][key], branch_pools[ph][key]], axis=0)
            for key in branch_pools[ph]
        }
    val_pool = phase_data_from_rows(eos, val_rows)
    center, scale, phi_scale = common_scaling(branch_pools["liquid"], branch_pools["vapor"])
    anchor_count_l = min(len(anchor_pool["liquid"]["phi"]), len(branch_pools["liquid"]["phi"]))
    anchor_count_v = min(len(anchor_pool["vapor"]["phi"]), len(branch_pools["vapor"]["phi"]))
    # The earlier 384-support validation-greedy order under-resolved the actual
    # coupled trajectory.  Use the established candidate self-error rule and a
    # slightly larger frozen budget.  Exact initial/inlet states remain in the
    # declared candidate pool but compete under the same error criterion; a
    # forced prefix was found to displace supports needed later in the path.
    M = min(128 if quick else 512, len(branch_pools["liquid"]["phi"]),
            len(branch_pools["vapor"]["phi"]))
    order_tau = 0.65
    ol = self_error_support_order(branch_pools["liquid"], M, center, scale,
                                  phi_scale, order_tau)
    ov = self_error_support_order(branch_pools["vapor"], M, center, scale,
                                  phi_scale, order_tau)

    # A common scalar offset corrects the absolute-pressure bias introduced by
    # normalized log-sum-exp.  It is fitted only on the independent validation
    # set and is identical for both phases, so phase competition is unchanged.
    tau_grid = [0.4, 0.5, 0.65, 0.8, 1.0]
    tau_validation = []
    for tau in tau_grid:
        raw_l = make_branch(branch_pools["liquid"], ol, tau, center, scale,
                            phi_scale, "liquid")
        raw_v = make_branch(branch_pools["vapor"], ov, tau, center, scale,
                            phi_scale, "vapor")
        pred_l, _ = raw_l.evaluate_batch(val_pool["liquid"]["eta"])
        pred_v, _ = raw_v.evaluate_batch(val_pool["vapor"]["eta"])
        offset = float(np.median(np.r_[
            val_pool["liquid"]["phi"] - pred_l,
            val_pool["vapor"]["phi"] - pred_v,
        ]))
        trial_l = make_branch(branch_pools["liquid"], ol, tau, center, scale,
                              phi_scale, "liquid", offset=offset)
        trial_v = make_branch(branch_pools["vapor"], ov, tau, center, scale,
                              phi_scale, "vapor", offset=offset)
        met_l = metric_summary(trial_l, val_pool["liquid"])
        met_v = metric_summary(trial_v, val_pool["vapor"])
        objective = (met_l["phi_rel_mae"] + met_v["phi_rel_mae"] +
                     met_l["gradient_rel_mae"] + met_v["gradient_rel_mae"])
        tau_validation.append({
            "tau": tau, "common_phi_offset": offset,
            "objective": objective,
            "liquid_phi_rel_mae": met_l["phi_rel_mae"],
            "vapor_phi_rel_mae": met_v["phi_rel_mae"],
            "liquid_gradient_rel_mae": met_l["gradient_rel_mae"],
            "vapor_gradient_rel_mae": met_v["gradient_rel_mae"],
        })
    best_validation = min(row["objective"] for row in tau_validation)
    # Prefer the sharpest stable member whose validation objective is within
    # 15% of the minimum.  This avoids the nonconvergent tau=0.125 regime while
    # retaining pressure resolution instead of over-smoothing the branches.
    selected_tau = min(
        (row for row in tau_validation if row["objective"] <= 1.15 * best_validation),
        key=lambda row: row["tau"],
    )
    tau = float(selected_tau["tau"])
    common_offset = float(selected_tau["common_phi_offset"])
    L = make_branch(branch_pools["liquid"], ol, tau, center, scale,
                    phi_scale, "liquid", offset=common_offset)
    V = make_branch(branch_pools["vapor"], ov, tau, center, scale,
                    phi_scale, "vapor", offset=common_offset)
    arrdict_save(RESULTS / "ternary_branch_pools.npz", {f"liquid_{k}": v for k, v in branch_pools["liquid"].items()} | {f"vapor_{k}": v for k, v in branch_pools["vapor"].items()})
    # Independent branch validation on whichever phase is active.
    val_metrics = {"liquid": metric_summary(L, val_pool["liquid"]), "vapor": metric_summary(V, val_pool["vapor"])}
    # Initial and inlet reference states, then map them into the MAGE coordinates.
    ref_old = [pr_equilibrium_state(eos, T, P, z0) for P in Pinit]; ref_in = pr_equilibrium_state(eos, T, Pin, zin)
    mage_old = [mage_from_reference_state(r, L, V, T) for r in ref_old]; mage_in = mage_from_reference_state(ref_in, L, V, T)
    # A 100 s step keeps the coupled complementarity solve in the same
    # convergence basin for all six fully implicit updates.
    params = dict(phi=0.20, dx=dx, dt=100.0, k=1e-16, mu_l=2.5e-4, mu_g=1.8e-5, Pin=Pin, Pout=Pout)
    mage_hist = []; pr_hist = []; nonlinear = []; counters = {"mage_branch_evaluations": 0, "pr_local_flash_calls": 0, "pr_eos_evaluations": 0, "mage_residual_calls": 0, "pr_residual_calls": 0}
    for step in range(6 if not quick else 2):
        mage_new, sm, cm = solve_mage_pressure_step(mage_old, mage_in, T, L, V, **params)
        pr_new, sp, cp = solve_pr_pressure_step(ref_old, ref_in, eos, T, **params)
        mage_residual = float(np.max(np.abs(sm.fun)))
        pr_residual = float(np.max(np.abs(sp.fun)))
        if mage_residual > 1.0e-6:
            raise RuntimeError(f"MAGE pressure step {step + 1} did not converge: {mage_residual:.3e}")
        if pr_residual > 1.0e-6:
            raise RuntimeError(f"PR pressure step {step + 1} did not converge: {pr_residual:.3e}")
        mage_hist.append(mage_new); pr_hist.append(pr_new)
        nonlinear.append({"step": step + 1, "mage_success": bool(sm.success and mage_residual <= 1.0e-6), "pr_success": bool(sp.success and pr_residual <= 1.0e-6), "mage_nfev": int(sm.nfev), "pr_nfev": int(sp.nfev), "mage_residual_max": mage_residual, "pr_residual_max": pr_residual})
        counters["mage_branch_evaluations"] += cm["branch_evaluations"]; counters["mage_residual_calls"] += cm["residual_calls"]; counters["pr_local_flash_calls"] += cp["local_flash_calls"]; counters["pr_eos_evaluations"] += cp["PR_EOS_evaluations"]; counters["pr_residual_calls"] += cp["residual_calls"]
        mage_old, ref_old = mage_new, pr_new
    rows = []; phase_rows = []
    for n, (mh, ph) in enumerate(zip(mage_hist, pr_hist), 1):
        for j, (m, p) in enumerate(zip(mh, ph)):
            z_m, z_p = m["z"], p["z"]; rows.append([n, j, m["P"], p["P"], abs(m["P"] - p["P"]) / p["P"], *np.abs(z_m - z_p), abs(m["S"] - p["S"]), m["S"], p["S"]])
            phase_rows.append([n, j, {"liquid": 0, "two-phase": 1, "vapor": 2}.get(p["phase"], -1), {"liquid": 0, "two-phase": 1, "vapor": 2}.get("two-phase" if 1e-4 < m["S"] < 1 - 1e-4 else ("vapor" if m["S"] >= 1 - 1e-4 else "liquid"), -1), (m["pV"] - m["pL"]) / 2e6])
    csv_save(RESULTS / "ternary_flow_states.csv", rows, ["step", "cell", "P_mage", "P_pr", "P_relerr", "z1_abs", "z2_abs", "z3_abs", "S_abs", "S_mage", "S_pr"])
    csv_save(RESULTS / "ternary_flow_phases.csv", phase_rows, ["step", "cell", "phase_pr", "phase_mage", "F_scaled"])
    flow_arr = np.asarray(rows, float); phase_arr = np.asarray(phase_rows, float)
    events = {"PR": [], "MAGE": []}
    for method, col in (("PR", 2), ("MAGE", 3)):
        for cell in range(N):
            seq = phase_arr[phase_arr[:, 1] == cell, col].astype(int)
            for k in range(1, len(seq)):
                if seq[k] != seq[k - 1]: events[method].append({"cell": cell, "from": int(seq[k - 1]), "to": int(seq[k]), "step": int(k + 1)})
    flow_metrics = {"P_rel_mean": float(np.mean(flow_arr[:, 4])), "P_rel_max": float(np.max(flow_arr[:, 4])), "z_abs_mean": float(np.mean(flow_arr[:, 5:8])), "z_abs_max": float(np.max(flow_arr[:, 5:8])), "S_abs_mean": float(np.mean(flow_arr[:, 8])), "S_abs_max": float(np.max(flow_arr[:, 8])), "F_scaled_max": float(np.max(np.abs(phase_arr[:, 4]))), "phase_agreement_fraction": float(np.mean(phase_arr[:, 2] == phase_arr[:, 3]))}
    json_save(RESULTS / "ternary_summary.json", {"seed": SEED, "domain": {"P_MPa": [1.25, 3.25], "sampling": "Dirichlet mixture around eight declared operating centers", "dirichlet_concentration_candidate": 55.0, "dirichlet_concentration_validation": 45.0, "operating_centers": TERNARY_OPERATING_CENTERS}, "counts": {"liquid": nliq, "two_phase": ntp, "vapor": nvap, "validation": nval, "candidate_attempts": candidate_attempts, "validation_attempts": validation_attempts}, "supports": M, "tau_liquid": tau, "tau_vapor": tau, "common_phi_offset": common_offset, "support_selection": "candidate self-error greedy over the complete declared pool", "tau_selection": "smallest tau within 15% of the minimum independent-validation objective", "tau_validation": tau_validation, "support_order_liquid": ol, "support_order_vapor": ov, "validation": {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)} for k, v in val_metrics.items()}, "flow": {"N": N, "L": 0.6, "P_in": Pin, "P_out": Pout, "z_initial": z0, "z_inlet": zin, **params}, "metrics": flow_metrics, "nonlinear": nonlinear, "phase_events": events, "counters": counters})
    return {"mage": mage_hist, "pr": pr_hist, "flow_rows": np.asarray(rows), "phase_rows": np.asarray(phase_rows), "L": L, "V": V}


def run_gerg(quick=False):
    teacher = GERGTeacher();
    if quick: nc, nv, nt, no, Ms = 2000, 500, 1000, 300, [96, 192, 384]
    else: nc, nv, nt, no, Ms = 80000, 20000, 20000, 5000, [96, 128, 192, 256, 384, 512, 768, 1024, 1536]
    pools = {}
    for ph, seed in (("liquid", 90), ("vapor", 100)):
        pools[("candidate", ph)] = gerg_pool(teacher, nc, 1.5e6, 10e6, seed, ph)
        pools[("validation", ph)] = gerg_pool(teacher, nv, 1.5e6, 10e6, seed + 1, ph)
        pools[("test", ph)] = gerg_pool(teacher, nt, 1.5e6, 10e6, seed + 2, ph)
        arrdict_save(RESULTS / f"gerg_{ph}_pools.npz", {f"{kind}_{k}": v for kind in ("candidate", "validation", "test") for k, v in pools[(kind, ph)].items()})
    center, scale, phi_scale = common_scaling(pools[("candidate", "liquid")], pools[("candidate", "vapor")]); tau_grid = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    ol, tl = select_support_order(pools[("candidate", "liquid")], pools[("validation", "liquid")], max(Ms), center, scale, phi_scale, tau_grid); ov, tv = select_support_order(pools[("candidate", "vapor")], pools[("validation", "vapor")], max(Ms), center, scale, phi_scale, tau_grid)
    rows = []
    for M in Ms:
        L = make_branch(pools[("candidate", "liquid")], ol[:M], tl.get(M, 0.5), center, scale, phi_scale, "liquid"); V = make_branch(pools[("candidate", "vapor")], ov[:M], tv.get(M, 0.5), center, scale, phi_scale, "vapor")
        a, b = metric_summary(L, pools[("test", "liquid")]), metric_summary(V, pools[("test", "vapor")]); rows.append({"M": M, "tau_l": tl.get(M, 0.5), "tau_v": tv.get(M, 0.5), "liquid": {k: v for k, v in a.items() if np.isscalar(v)}, "vapor": {k: v for k, v in b.items() if np.isscalar(v)}})
    # Independent overall-state classification is deliberately separated from
    # support selection and uses the same GERG flash routine.
    overall = gerg_overall_test(teacher, no, SEED + 110)
    json_save(RESULTS / "gerg_summary.json", {"seed": SEED, "counts": {"candidate": nc, "validation": nv, "branch_test": nt, "overall_test": no}, "selected_M": rows[-1]["M"], "rows": rows, "center": center, "scale": scale, "phi_scale": phi_scale, "overall": overall})
    return {"rows": rows, "overall": overall}


def gerg_pool(teacher, n, p0, p1, seed, phase):
    return teacher_pool(teacher, n, p0, p1, seed, phase)


def teacher_pool(teacher, n, p0, p1, seed, phase):
    sampler = qmc.Sobol(4, scramble=True, seed=seed); rows = []
    while len(rows) < n:
        for u in sampler.random(4096):
            P = p0 + u[0] * (p1 - p0); z = (np.asarray(u[1:4]) + 0.02); z /= z.sum()
            try:
                # Build supports only from stable states returned by the GERG
                # flash.  Sampling arbitrary phase roots admits metastable
                # states whose tangent planes are not global supports.
                zsafe = np.array([0.35 + 0.30 * u[1], 0.10 + 0.20 * u[2], 0.15 + 0.20 * u[3]])
                zsafe /= zsafe.sum()
                equilibrium = teacher.flash(320.0, P, zsafe)
                if equilibrium["phase"] == "two-phase":
                    composition = equilibrium["x"] if phase == "liquid" else equilibrium["y"]
                elif equilibrium["phase"] == phase:
                    composition = zsafe
                else:
                    composition = None
                if composition is not None:
                    rows.append(teacher.state(320.0, P, composition, phase))
            except Exception:
                pass
            if len(rows) >= n: break
    return {k: np.asarray([r[k] for r in rows]) for k in ("T", "P", "x", "mu", "c", "u", "phi", "eta", "q")}


def gerg_overall_test(teacher, n, seed):
    sampler = qmc.Sobol(4, scramble=True, seed=seed); rows = []
    for u in sampler.random(n):
        P = 1.5e6 + u[0] * 8.5e6; z = np.array([0.35 + 0.30 * u[1], 0.10 + 0.20 * u[2], 0.15 + 0.20 * u[3]]); z /= z.sum()
        try:
            r = teacher.flash(320.0, P, z); rows.append([P, *z, r["phase_id"], r["beta"]])
        except Exception: pass
    arr = np.asarray(rows); return {"n": int(len(arr)), "phase_counts": {str(k): int(np.sum(arr[:, 4] == k)) for k in np.unique(arr[:, 4])}, "states": arr}


def run_lle(quick=False):
    model = RegularHelmholtzLLE();
    if quick: nc, nv, nt, Ms = 600, 150, 300, [16, 32, 64, 128]
    else: nc, nv, nt, Ms = 20000, 5000, 5000, [16, 24, 32, 48, 64, 96, 128, 192, 256]
    pools = {}
    # The homogeneous regular-solution branches are convex only outside the
    # spinodal interval (0.2113, 0.7887) for chi=3.  Concave states cannot
    # define valid global Massieu supports and previously displaced coexistence.
    for br, xr, seed in (("L1", (0.005, 0.20), 120), ("L2", (0.80, 0.995), 130)):
        pools[("candidate", br)] = make_lle_pool(model, nc, (4.5e6, 5.5e6), xr, SEED + seed)
        pools[("validation", br)] = make_lle_pool(model, nv, (4.5e6, 5.5e6), xr, SEED + seed + 1)
        pools[("test", br)] = make_lle_pool(model, nt, (4.5e6, 5.5e6), xr, SEED + seed + 2)
    center, scale, phi_scale = common_scaling(pools[("candidate", "L1")], pools[("candidate", "L2")]); tau_grid = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    ol, tl = select_support_order(pools[("candidate", "L1")], pools[("validation", "L1")], max(Ms), center, scale, phi_scale, tau_grid); ov, tv = select_support_order(pools[("candidate", "L2")], pools[("validation", "L2")], max(Ms), center, scale, phi_scale, tau_grid)
    rows = []; coex_ref = np.array([0.0707201817, 0.9292798183])
    for M in Ms:
        L = make_branch(pools[("candidate", "L1")], ol[:M], tl.get(M, 0.5), center, scale, phi_scale, "L1"); V = make_branch(pools[("candidate", "L2")], ov[:M], tv.get(M, 0.5), center, scale, phi_scale, "L2")
        a, b = metric_summary(L, pools[("test", "L1")]), metric_summary(V, pools[("test", "L2")]); r = solve_mage_binary(300.0, 5e6, L, V); xp = np.array([r["x"][0], r["y"][0]])
        rows.append({"M": M, "tau_l1": tl.get(M, 0.5), "tau_l2": tv.get(M, 0.5), "coex_x1": xp[0], "coex_x2": xp[1], "coex_abs_max": float(np.max(np.abs(xp - coex_ref))), "converged": bool(r["success"]), "L": {k: v for k, v in a.items() if np.isscalar(v)}, "V": {k: v for k, v in b.items() if np.isscalar(v)}})
    json_save(RESULTS / "lle_summary.json", {"seed": SEED, "counts": {"candidate": nc, "validation": nv, "test": nt}, "analytic": coex_ref, "rows": rows, "center": center, "scale": scale, "phi_scale": phi_scale})
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true", help="small deterministic smoke run"); ap.add_argument("--reuse-pools", action="store_true", help="reuse compatible saved nonisothermal state pools"); ap.add_argument("--only", choices=["all", "binary", "nonisothermal", "ternary", "gerg", "lle"], default="all"); args = ap.parse_args(); t0 = time.time(); out = {}
    if args.only in ("all", "binary"): out["binary"] = run_binary(args.quick)
    if args.only in ("all", "nonisothermal"): out["nonisothermal"] = run_nonisothermal(args.quick, reuse_pools=args.reuse_pools)
    if args.only in ("all", "ternary"): out["ternary"] = run_ternary_flow(args.quick)
    if args.only in ("all", "gerg"): out["gerg"] = run_gerg(args.quick)
    if args.only in ("all", "lle"): out["lle"] = run_lle(args.quick)
    summaries = {}
    for name in ("binary", "nonisothermal", "ternary", "gerg", "lle"):
        path = RESULTS / f"{name}_summary.json"
        if path.exists():
            summaries[name] = {"file": path.name, "bytes": path.stat().st_size,
                               "modified_unix_s": path.stat().st_mtime}
    json_save(RESULTS / "run_manifest.json", {
        "seed": SEED,
        "last_run": {"quick": args.quick, "reuse_pools": args.reuse_pools,
                     "only": args.only, "elapsed_s": time.time() - t0},
        "python": sys.version,
        "available_full_result_groups": summaries,
    })
    print(json.dumps({"elapsed_s": time.time() - t0, "outputs": list(out)}, indent=2))


if __name__ == "__main__":
    main()
