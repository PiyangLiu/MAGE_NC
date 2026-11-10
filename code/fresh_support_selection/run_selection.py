"""Run the fresh common selector for all cases except Figure 2.

Usage examples (from this directory)::

    G:\developer\anconda\python.exe run_selection.py --case nonisothermal
    G:\developer\anconda\python.exe run_selection.py --case all

The command writes JSON/NPZ/CSV audit files below ``results``.  It never
touches manuscript files or any legacy selection directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cases import case_loaders
from greedy_selector import (evaluate_metrics, save_json,
                             select_minimum_validation_greedy)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


CONFIG = {
    # Thresholds are declared before the independent test calculation.  They
    # are the same validation score units used in the previous engineering
    # reports; no test/coexistence/flow metric enters these gates.
    "nonisothermal": {
        # The final nonisothermal selection was run with the full finite
        # candidate bound; its first feasible prefixes are 1592 and 1392.
        "branches": ("liquid", "vapor"), "max_m": 4096,
        "tau_grid": (0.125,), "weights": (1.0, 1.0, 1.0),
        "thresholds": {"liquid": {"mean": 7.980196842313327, "max": 222.36922633791635},
                       "vapor": {"mean": 6.475571123578879, "max": 40.7579192793293}},
        "coverage_count": 8, "neighbourhood_size": 256,
        "candidate_residual_weight": 1.0,
    },
    "gerg": {
        "branches": ("liquid", "vapor"), "max_m": 4096,
        "tau_grid": (0.03125,), "weights": (10.0, 1.0, 1.0),
        "thresholds": {"liquid": {"mean": 15.643776551296027, "max": 72.60773844090605},
                       "vapor": {"mean": 12.480594537861508, "max": 50.2414568553282}},
        "coverage_count": 8, "neighbourhood_size": 256,
        "candidate_residual_weight": 5.0, "worst_state_count": 16,
    },
    "liquid_liquid": {
        "branches": ("L1", "L2"), "max_m": 1024,
        "tau_grid": (0.125,), "weights": (1.0, 1.0, 1.0),
        "thresholds": {"L1": {"mean": 0.6082026660524847, "max": 1.5589441032006428},
                       "L2": {"mean": 0.5951493228469784, "max": 1.5526804061155276}},
        "coverage_count": 8, "neighbourhood_size": 256,
        "candidate_residual_weight": 1.0,
    },
    "gas_brine": {
        "branches": ("liquid", "gas"), "max_m": 2048,
        "tau_grid": (0.25, 0.5, 1.0, 2.0), "weights": (1.0, 1.0, 1.0),
        "composition_weights": (1000.0, 1.0, 1.0),
        "thresholds": {"liquid": {"mean": 800.0, "max": 19000.0},
                       "gas": {"mean": 30.0, "max": 900.0}},
        "coverage_count": 8, "neighbourhood_size": 256,
        "candidate_residual_weight": 2.0,
    },
    "ternary_flow": {
        "branches": ("liquid", "vapor"), "max_m": 4096,
        # The lower-temperature candidates are deliberately excluded: the
        # validation set prefers them, but the resulting finite-support
        # Jacobian is too sharp for the downstream global flow solve.  The
        # remaining grid still selects tau from validation data only.
        "tau_grid": (0.65, 0.8, 1.0), "weights": (1.0, 1.0, 1.0),
        "thresholds": {"liquid": {"mean": 22.49, "max": 1900.0},
                       "vapor": {"mean": 18.05, "max": 250.0}},
        "coverage_count": 8, "neighbourhood_size": 256,
        "candidate_residual_weight": 1.0,
    },
}


def _write_curve(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = ["M", "tau", "validation_mean_score", "validation_max_score",
            "worst_validation_index", "target_met"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in keys} for row in history)


def run_case(name: str) -> dict:
    cfg = CONFIG[name]
    pools = case_loaders()[name]()
    result = {
        "case": name,
        "algorithm": "fresh validation-error greedy neighbourhood selector",
        "figure2_modified": False,
        "manuscript_modified": False,
        "selection_excludes": ["independent test", "coexistence composition", "KKT", "flow trajectory"],
        "branches": {},
        "configuration": {k: v for k, v in cfg.items() if k != "thresholds"},
        "validation_thresholds": cfg["thresholds"],
    }
    support_arrays = {}
    for branch in cfg["branches"]:
        p = pools[branch]
        cw = cfg.get("composition_weights")
        if cw is not None:
            # Gas-brine stores two composition derivatives while the liquid
            # branch stores three; retain the same H2 emphasis and trim only
            # the branch-inapplicable trailing entries.
            cw = tuple(cw[:max(0, p["candidate"]["q"].shape[1] - 1)])
        selection = select_minimum_validation_greedy(
            p["candidate"], p["validation"], max_m=cfg["max_m"],
            tau_grid=cfg["tau_grid"], thresholds=cfg["thresholds"][branch],
            weights=cfg["weights"], composition_weights=cw,
            batch_add=8, coverage_count=cfg["coverage_count"],
            neighbourhood_size=cfg["neighbourhood_size"],
            worst_state_count=cfg.get("worst_state_count", 16),
            candidate_residual_weight=cfg.get("candidate_residual_weight_by_branch", {}).get(
                branch, cfg["candidate_residual_weight"]),
            seed=20261105 + sum(map(ord, branch)), chunk_size=2048)
        selected = selection.selected_m
        tau = selection.selected_tau
        test_metrics = evaluate_metrics(p["candidate"], p["test"],
                                         selection.support_order[:selected], tau,
                                         selection.center, selection.scale,
                                         cfg["weights"], cw)
        branch_result = {
            "candidate_count": int(len(p["candidate"]["eta"])),
            "validation_count": int(len(p["validation"]["eta"])),
            "test_count": int(len(p["test"]["eta"])),
            **selection.as_dict(), "test_metrics": test_metrics,
        }
        if name == "ternary_flow":
            branch_result["test_metrics_basis"] = (
                "independent centered Dirichlet PR test pool; flow trajectory excluded from selection"
            )
        result["branches"][branch] = branch_result
        support_arrays[f"{branch}_support_order"] = selection.support_order
        _write_curve(OUT / f"{name}_{branch}_validation_curve.csv", selection.history)
    np.savez_compressed(OUT / f"{name}_support_orders.npz", **support_arrays)
    save_json(OUT / f"{name}_selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", *CONFIG], default="all")
    args = parser.parse_args()
    names = list(CONFIG) if args.case == "all" else [args.case]
    reports = [run_case(name) for name in names]
    manifest = {
        "program": "greedy_support_selection_fresh",
        "cases": names,
        "algorithm": "one common validation-error greedy algorithm; case parameters are explicit configuration",
        "figure2": "unchanged and intentionally excluded",
        "minimum_M_rule": "first nested prefix satisfying both validation mean and max thresholds",
        "postprocessing_selected_M": False,
        "reports": [str((OUT / f"{name}_selection.json").relative_to(HERE))
                    for name in names],
    }
    save_json(OUT / "manifest.json", manifest)
    print(json.dumps({"cases": names,
                      "selected_M": {r["case"]: {b: r["branches"][b]["selected_M"]
                                                    for b in r["branches"]} for r in reports}},
                     indent=2))


if __name__ == "__main__":
    main()
