#!/usr/bin/env python3
"""Audit final support domains, scaling, and Figure 5/Case B online states.

The audit is deliberately descriptive: it does not alter any retained result.
It reports componentwise support envelopes and, where the fixed-temperature
ternary dual space is full-dimensional after removing eta_0, convex-hull tests.
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

# The bundled SciPy build still references removed NumPy 1.x aliases.
for _name, _dtype in (("long", np.int64), ("ulong", np.uint64)):
    if _name not in np.__dict__:
        setattr(np, _name, _dtype)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE.parent / "data"
FIG5 = BASE / "figure5_ternary_fully_implicit"
UNIFIED = BASE / "unified_selection"
CASEB_DATA = BASE.parent / "source_data" / "Supplementary_Note_5_2_second_flow"
OUT = BASE / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(summary: dict, name: str):
    x = summary["datasets"][name]
    cols = x["columns"]
    rows = x["data"]
    if isinstance(rows, dict):
        rows = zip(*(rows[c] for c in cols))
    return [dict(zip(cols, row)) for row in rows]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def finite(a):
    a = np.asarray(a, float)
    return bool(np.isfinite(a).all())


def add_bounds(rows, case, source, quantity, values, labels=None):
    arr = np.asarray(values, float)
    if arr.ndim == 1:
        rows.append({"case": case, "source": source, "quantity": quantity,
                     "component": "", "minimum": float(np.min(arr)),
                     "maximum": float(np.max(arr)), "count": int(arr.size)})
        return
    for j in range(arr.shape[1]):
        comp = labels[j] if labels else str(j)
        rows.append({"case": case, "source": source, "quantity": quantity,
                     "component": comp, "minimum": float(np.min(arr[:, j])),
                     "maximum": float(np.max(arr[:, j])), "count": int(arr.shape[0])})


def pool_bounds(rows, case, source, pool, prefix="candidate_"):
    add_bounds(rows, case, source, "temperature_K", pool[prefix + "T"])
    add_bounds(rows, case, source, "pressure_Pa", pool[prefix + "P"])
    if prefix + "x" in pool:
        x = pool[prefix + "x"]
    else:
        x1 = pool[prefix + "x1"]
        x = np.c_[x1, 1.0 - x1]
    add_bounds(rows, case, source, "branch_composition", x, ["component_1", "component_2"] if x.shape[1] == 2 else ["component_1", "component_2", "component_3"])
    add_bounds(rows, case, source, "dual_eta", pool[prefix + "eta"], ["eta_0", "eta_1", "eta_2", "eta_3"][:pool[prefix + "eta"].shape[1]])
    add_bounds(rows, case, source, "gradient_q", pool[prefix + "q"], ["u_density", "c_1", "c_2", "c_3"][:pool[prefix + "q"].shape[1]])


def read_case_states(summary_path: Path, name: str):
    s = load_json(summary_path)
    return table(s, name)


def online_arrays(rows, prefix=""):
    return np.array([r[f"P_MPa{prefix}"] for r in rows], float), np.array(
        [[r[f"z_CH4{prefix}"], r[f"z_CO2{prefix}"], r[f"z_nC4{prefix}"]] for r in rows], float
    )


def ternary_dual_audit(case_name, states, config_path, model_dir, support_eta):
    """Map online P,z to MAGE coordinates and audit the fixed support domain."""
    sys.path.insert(0, str(FIG5 / "src"))
    from solvers import MAGEGlobalSolver  # type: ignore
    solver = MAGEGlobalSolver(config_path, model_dir)
    rows = []
    mapped = []
    for r in states:
        p = float(r["P_MPa_mage"] if "P_MPa_mage" in r else r["P_MPa"])
        z = np.array([float(r["z_CH4_mage"] if "z_CH4_mage" in r else r["z_CH4"]),
                      float(r["z_CO2_mage"] if "z_CO2_mage" in r else r["z_CO2"]),
                      float(r["z_nC4_mage"] if "z_nC4_mage" in r else r["z_nC4"])])
        state, residual, _ = solver.map_state(p * 1e6, z)
        mapped.append(state)
        rows.append({"case": case_name, "accepted_step": int(r["accepted_step"]),
                     "cell": int(r["cell"]), "map_residual": float(residual),
                     "P_MPa": p, "S": float(state[3]),
                     "eta_1": float(state[0]), "eta_2": float(state[1]), "eta_3": float(state[2])})
    mapped = np.asarray(mapped)
    nu = mapped[:, :3]
    bounds = {"P_MPa": [1.25, 3.25], "S": [0.0, 1.0],
              "eta_solver_lower": [-0.45, -0.2, -0.45],
              "eta_solver_upper": [3.35, 3.0, 1.55]}
    online_min = np.r_[np.min(nu, axis=0), np.min(mapped[:, 3]), np.min([x["P_MPa"] for x in rows])]
    online_max = np.r_[np.max(nu, axis=0), np.max(mapped[:, 3]), np.max([x["P_MPa"] for x in rows])]
    support_min = np.min(support_eta[:, 1:], axis=0)
    support_max = np.max(support_eta[:, 1:], axis=0)
    in_box = bool(np.all(nu >= support_min - 1e-10) and np.all(nu <= support_max + 1e-10))
    margins = np.column_stack([nu - np.asarray(bounds["eta_solver_lower"]), np.asarray(bounds["eta_solver_upper"]) - nu])
    s_margin = np.minimum(mapped[:, 3], 1.0 - mapped[:, 3])
    p_margin = np.minimum(np.array([x["P_MPa"] for x in rows]) - 1.25, 3.25 - np.array([x["P_MPa"] for x in rows]))
    hull = {"available": False, "inside_fraction": None, "max_inequality": None}
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(support_eta[:, 1:])
        vals = np.c_[nu, np.ones(len(nu))] @ h.equations.T
        hull = {"available": True, "inside_fraction": float(np.mean(np.max(vals, axis=1) <= 1e-9)),
                "max_inequality": float(np.max(vals))}
    except Exception as exc:
        hull = {"available": False, "inside_fraction": None, "max_inequality": None, "reason": str(exc)}
    audit = {
        "case": case_name, "state_count": len(rows),
        "all_map_residuals_finite": bool(np.isfinite([x["map_residual"] for x in rows]).all()),
        "max_map_residual": float(max(x["map_residual"] for x in rows)),
        "online_minimums_eta_1_eta_2_eta_3_S_P_MPa": online_min.tolist(),
        "online_maximums_eta_1_eta_2_eta_3_S_P_MPa": online_max.tolist(),
        "selected_support_eta_min": support_min.tolist(),
        "selected_support_eta_max": support_max.tolist(),
        "all_online_dual_states_inside_selected_componentwise_support_box": in_box,
        "all_online_pressure_inside_declared_1p25_to_3p25_MPa": bool(np.all(p_margin >= -1e-10)),
        "all_online_S_inside_closed_unit_interval": bool(np.all(s_margin >= -1e-10)),
        "minimum_distance_to_solver_eta_bounds": float(np.min(margins)),
        "minimum_distance_to_S_bounds": float(np.min(s_margin)),
        "minimum_distance_to_pressure_bounds_MPa": float(np.min(p_margin)),
        "selected_support_convex_hull_test_on_eta_1_to_eta_3": hull,
        "interpretation": "The box and hull tests are coverage audits. The Massieu log-sum-exp remains mathematically evaluable outside the support hull; no online state was accepted by an explicit extrapolative branch or a root-selection step.",
    }
    return rows, audit


def main():
    bounds_rows = []
    scaling_rows = []
    for fig in (2, 3):
        s = load_json(DATA / f"figure{fig}_summary.json")
        c = s["calculation_summary"]
        if fig == 6:
            # Figure 6 contains two topology cases; retain separate records below.
            for case, info in c.items():
                if not isinstance(info, dict):
                    continue
                rr = info.get("rows", [])
                selected = info.get("selected_M", max((r.get("M", 0) for r in rr), default=None))
                sr = next((r for r in rr if r.get("M") == selected), {})
                phi_s = info.get("phi_scale", c.get("phi_scale", 1.0))
                tau_l = sr.get("tau_l", sr.get("tau_l1", info.get("selected_tau", info.get("tau", "not_reported"))))
                tau_v = sr.get("tau_v", sr.get("tau_l2", tau_l))
                scaling_rows.append({
                    "figure": fig, "case": case, "phi_s": phi_s,
                    "tau_liquid": tau_l, "tau_vapor": tau_v,
                    "tau_bar_liquid": tau_l / phi_s if isinstance(tau_l, (int, float)) else tau_l,
                    "tau_bar_vapor": tau_v / phi_s if isinstance(tau_v, (int, float)) else tau_v,
                    "eta_center": json.dumps(info.get("center", c.get("center", "not_reported"))),
                    "D_eta": json.dumps(info.get("scale", c.get("scale", "not_reported"))),
                    "common_phi_offset": 0.0,
                    "branch_reference_policy": "same reference potential and zero added offset for both branches",
                })
        else:
            rows = c.get("rows", [])
            selected = c.get("selected_M", max((r.get("M", 0) for r in rows), default=None))
            selected_row = next((r for r in rows if r.get("M") == selected), {})
            phi_s = c.get("phi_scale", 1.0)
            tau_l = selected_row.get("tau_l", c.get("selected_tau_l", "not_reported"))
            tau_v = selected_row.get("tau_v", c.get("selected_tau_v", tau_l))
            scaling_rows.append({
                "figure": fig, "case": s.get("scientific_case", ""),
                "phi_s": phi_s, "tau_liquid": tau_l, "tau_vapor": tau_v,
                "tau_bar_liquid": tau_l / phi_s if isinstance(tau_l, (int, float)) else tau_l,
                "tau_bar_vapor": tau_v / phi_s if isinstance(tau_v, (int, float)) else tau_v,
                "eta_center": json.dumps(c.get("center", "not_reported")),
                "D_eta": json.dumps(c.get("scale", "not_reported")),
                "common_phi_offset": 0.0,
                "branch_reference_policy": "same reference potential and zero added offset for both branches",
            })
    c5 = load_json(DATA / "coupled_ternary_flow_plot_data.json")
    m5 = c5["calculation_summary"]["support_selection"]
    scaling_rows.append({
        "figure": 6, "case": "ternary fully implicit", "phi_s": 1.0,
        "tau_liquid": m5["tau"], "tau_vapor": m5["tau"],
        "tau_bar_liquid": m5["tau"], "tau_bar_vapor": m5["tau"],
        "eta_center": json.dumps([0.0, 0.0, 0.0, 0.0]),
        "D_eta": json.dumps([1.0, 1.0, 1.0, 1.0]),
        "common_phi_offset": m5["shared_phi_offset"],
        "branch_reference_policy": "one validation-calibrated offset applied identically to both branches",
    })
    scaling_rows.extend([
        {
            "figure": 5, "case": "GERG-2008", "phi_s": 1.0,
            "tau_liquid": 0.125, "tau_vapor": 0.125,
            "tau_bar_liquid": 0.125, "tau_bar_vapor": 0.125,
            "eta_center": json.dumps([-3.7585111e-4, 3.1566995, 2.5837546, 1.1041729]),
            "D_eta": json.dumps([1.0, 0.3500855647, 0.5801767858, 0.1126064030]),
            "common_phi_offset": 0.0,
            "branch_reference_policy": "same reference potential and zero added offset for both branches",
        },
        {
            "figure": 5, "case": "liquid-liquid equilibrium", "phi_s": 1.0,
            "tau_liquid": 0.125, "tau_vapor": 0.125,
            "tau_bar_liquid": 0.125, "tau_bar_vapor": 0.125,
            "eta_center": json.dumps([-4.0090785e-4, -0.0789596610, -0.0789591662]),
            "D_eta": json.dumps([1.0, 0.3747343810, 0.3747101886]),
            "common_phi_offset": 0.0,
            "branch_reference_policy": "same reference potential and zero added offset for both branches",
        },
    ])
    strengthened_dir = BASE / "figure4_nonisothermal" / "strengthened_results"
    strengthened_summary = load_json(strengthened_dir / "summary.json")
    strengthened_sel = dict(np.load(strengthened_dir / "support_selection.npz"))
    scaling_rows.append({
        "figure": "4-strengthened", "case": "three withheld 6 K temperature bands",
        "phi_s": 1.0,
        "tau_liquid": strengthened_summary["support_selection"]["selected_tau"]["liquid"],
        "tau_vapor": strengthened_summary["support_selection"]["selected_tau"]["vapor"],
        "tau_bar_liquid": strengthened_summary["support_selection"]["selected_tau"]["liquid"],
        "tau_bar_vapor": strengthened_summary["support_selection"]["selected_tau"]["vapor"],
        "eta_center": json.dumps(strengthened_sel["center"].tolist()),
        "D_eta": json.dumps(strengthened_sel["scale"].tolist()),
        "common_phi_offset": 0.0,
        "branch_reference_policy": "same nonisothermal PR potential and zero added offset for both branches",
    })

    for fig, folder, case in ((2, "figure2_binary_equilibrium", "binary"), (3, "figure3_support_compression", "binary"), (4, "figure4_nonisothermal", "nonisothermal"), (6, "figure6_gerg2008", "GERG-2008")):
        if fig in (2, 3):
            for ph in ("liquid", "vapor"):
                pool = dict(np.load(BASE / folder / f"binary_{ph}_pools.npz"))
                pool_bounds(bounds_rows, f"Figure {fig} {ph}", "candidate pool", pool)
        elif fig == 4:
            for ph in ("liquid", "vapor"):
                pool = dict(np.load(BASE / folder / f"nonisothermal_{ph}_pools.npz"))
                pool_bounds(bounds_rows, f"Figure 4 {ph}", "candidate pool", pool)
        else:
            for ph in ("liquid", "vapor"):
                pool = dict(np.load(BASE / folder / f"gerg_{ph}_pools.npz"))
                pool_bounds(bounds_rows, f"Figure 6 GERG {ph}", "candidate pool", pool)
    for ph in ("liquid", "vapor"):
        pool = dict(np.load(strengthened_dir / f"candidate_{ph}.npz"))
        flat = {f"candidate_{k}": v for k, v in pool.items()}
        pool_bounds(bounds_rows, f"Figure 4 strengthened {ph}", "candidate pool excluding three 6 K bands", flat)
        order = strengthened_sel[f"{ph}_order"][:strengthened_summary["support_selection"]["selected_support_count"]]
        add_bounds(bounds_rows, f"Figure 4 strengthened {ph}", "selected 1024 supports", "temperature_K", pool["T"][order])
        add_bounds(bounds_rows, f"Figure 4 strengthened {ph}", "selected 1024 supports", "pressure_Pa", pool["P"][order])
        add_bounds(bounds_rows, f"Figure 4 strengthened {ph}", "selected 1024 supports", "dual_eta", pool["eta"][order], ["eta_0", "eta_1", "eta_2"])

    # Figure 6 physical candidate/support envelopes.
    teacher = dict(np.load(FIG5 / "teacher" / "ternary_branch_pools.npz"))
    for ph in ("liquid", "vapor"):
        add_bounds(bounds_rows, f"Figure 5 {ph}", "declared candidate pool", "temperature_K", teacher[f"{ph}_T"])
        add_bounds(bounds_rows, f"Figure 5 {ph}", "declared candidate pool", "pressure_Pa", teacher[f"{ph}_P"])
        add_bounds(bounds_rows, f"Figure 5 {ph}", "declared candidate pool", "branch_composition", teacher[f"{ph}_x"], ["component_1", "component_2", "component_3"])
        model = dict(np.load(UNIFIED / "ternary_model" / "mage_ternary_1280.npz"))
        add_bounds(bounds_rows, f"Figure 5 {ph}", "selected 1280 supports", "dual_eta", model[f"{ph}_eta"], ["eta_0", "eta_1", "eta_2", "eta_3"])
        add_bounds(bounds_rows, f"Figure 5 {ph}", "selected 1280 supports", "gradient_q", model[f"{ph}_q"], ["u_density", "c_1", "c_2", "c_3"])

    with (OUT / "potential_scaling_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in scaling_rows))))
        w.writeheader(); w.writerows(scaling_rows)
    with (OUT / "support_domain_bounds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "source", "quantity", "component", "minimum", "maximum", "count"])
        w.writeheader(); w.writerows(bounds_rows)

    # Case A is the authoritative unified flow plot data; Case B is the
    # independently rerun second-condition history in Supplementary Note 5.2.
    rows5 = table(c5, "combined_global_states")
    rowsB = []
    import pandas as pd
    bdf = pd.read_csv(CASEB_DATA / "mage_global_states.csv")
    rowsB = bdf.to_dict("records")
    model = dict(np.load(UNIFIED / "ternary_model" / "mage_ternary_1280.npz"))
    _, auditA = ternary_dual_audit("Figure 6 Case A", rows5, FIG5 / "benchmark_config.json", UNIFIED / "ternary_model", np.vstack([model["liquid_eta"], model["vapor_eta"]]))
    _, auditB = ternary_dual_audit("Figure 6 Case B", rowsB, UNIFIED / "caseB_config.json", UNIFIED / "ternary_model", np.vstack([model["liquid_eta"], model["vapor_eta"]]))
    audits = {"case_A": auditA, "case_B": auditB, "model_sha256": sha256(UNIFIED / "ternary_model" / "mage_ternary_1280.npz")}
    (OUT / "flow_domain_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")

    filtering = {
        "PR_binary_and_ternary": {"flash_first": True, "tp_flash_or_stability_result_required": True, "single_phase_rule": "retain only when stable TP-flash phase matches requested branch", "two_phase_rule": "use TP-flash liquid x or vapour y composition for corresponding branch", "root_rule": "low cubic root for liquid and high cubic root for vapour only after stability classification", "metastable_states_in_support_pools": False, "reason": "cubic-root selection by itself admits metastable roots and is therefore not a support acceptance rule"},
        "GERG2008": {"flash_first": True, "interface": "Thermopack two_phase_tpflash", "single_phase_rule": "retain only when returned phase flag matches requested branch", "two_phase_rule": "use returned x/y compositions", "arbitrary_density_roots": False, "metastable_states_in_support_pools": False},
        "Figure6_flow": {"online_phase_switching": "complementarity residual Sg - projection(Sg + (pV-pL)/Pscale)", "online_local_flash_calls": 0, "phase_threshold": 1e-4, "support_domain": "fixed 1280-support branch model"}
    }
    (OUT / "support_state_filtering.json").write_text(json.dumps(filtering, indent=2), encoding="utf-8")

    (OUT / "offline_pool_conventions.md").write_text("""# Offline pool and acceptance conventions

## PR binary and nonisothermal pools

Each candidate draw is sent through the PR flash before a branch state is built. A stable single-phase state is retained only if the returned phase equals the requested branch. For a two-phase flash, the returned liquid or vapour composition is used for the corresponding branch. Cubic roots are evaluated only after this classification: the smallest valid root is the liquid root and the largest valid root is the vapour root. A root selected without a stable flash result is rejected, so metastable homogeneous roots are not included in the support pools. Candidate pools contain deterministic coexistence-grid states plus accepted stable states; validation pools are independent accepted states and test pools are held out from support ordering.

## Support ordering

Figures 2 and 3 use nested validation-driven ordering. The state with the largest combined potential/gradient/composition score is identified on the validation pool, and the nearest unused candidate is added. A deterministic coexistence prefix is retained. The strengthened Figure 4 calculation uses the same validation-error greedy rule on candidate pools containing stable homogeneous and coexistence states; all states in the three withheld temperature bands are excluded from candidate and validation pools. The earlier retained Figure 4 result is packaged separately and uses its equilibrium-grid support manifold. Figure 5 uses the frozen ternary candidate pool: the first 512 supports are retained and the order is extended to 1,280 by the maximum convex-support-plane gap over the declared candidate pool, without using flow-trajectory states. GERG-2008 uses deterministic farthest-point ordering for the large candidate pool after stable TP-flash filtering. The LLE example uses validation-driven nested construction on the two stable regular-solution branches.

## Validation and acceptance

Thermodynamic validation compares potential, gradient, concentration and phase-equilibrium outputs on pools not used to select the support order. Flow steps are accepted only when the scaled nonlinear residual is below the configured MAGE or PR threshold and the component mass-balance audit is finite. Case B additionally requires 552/552 phase-state agreement and identical phase-event transitions against the independent fully implicit PR calculation.
""", encoding="utf-8")

    (OUT / "si_parameter_table.md").write_text("""# Parameters for independent reconstruction

| Case | Reference model | T (K) | P domain | Supports per branch | Smoothing |
|---|---|---:|---|---:|---:|
| Figures 2-3 | PR, CH4/nC4 | 300 | 0.45-5.10 MPa candidate range | selected values in Figure 3 | selected from validation grid |
| Strengthened Figure 4 | nonisothermal PR, CH4/nC4 | 295-350 | 0.50-7.00 MPa | 1,592 liquid / 1,392 vapour | 0.125 |
| Figure 5 / Case B | PR, CH4/CO2/nC4 | 320 | 1.25-3.25 MPa declared flow domain | 1,280 | 0.5 |
| Figure 6 GERG | GERG-2008 through Thermopack | 320 | 1.5-10.0 MPa | selected M values in JSON | selected in JSON |
| Figure 6 LLE | analytic regular-Helmholtz model | fixed model temperature | 4.5-5.5 MPa sampled branch domain | selected M values in JSON | selected in JSON |

The machine-readable values, including dual-coordinate centers, `D_eta`, `Phi_s`, physical and scaled smoothing parameters, common branch offsets, and support bounds, are in `potential_scaling_parameters.csv` and `support_domain_bounds.csv`. The two phases always use a common thermodynamic reference. Figure 5 uses one validation-calibrated additive correction applied identically to both branches; all other packaged examples use zero added branch offset.
""", encoding="utf-8")

    print(json.dumps({"scaling_rows": len(scaling_rows), "bounds_rows": len(bounds_rows), "case_A": auditA, "case_B": auditB}, indent=2))


if __name__ == "__main__":
    main()
