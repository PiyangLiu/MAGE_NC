"""Independent checks for the strengthened nonisothermal benchmark outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nonisothermal_bandgap"
BANDS = ((302.0, 308.0), (322.0, 328.0), (342.0, 348.0))
PHASES = ("liquid", "vapor")


def in_bands(values):
    values = np.asarray(values, float)
    mask = np.zeros(values.shape, dtype=bool)
    for lower, upper in BANDS:
        mask |= (values >= lower) & (values <= upper)
    return mask


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    convergence = read_csv(RESULTS / "support_convergence.csv")
    state_errors = read_csv(RESULTS / "state_errors.csv")
    curves = read_csv(RESULTS / "vle_curves.csv")
    metrics = read_csv(RESULTS / "vle_metrics.csv")
    selection = np.load(RESULTS / "support_selection.npz")
    selected = int(summary["support_selection"]["selected_support_count"])

    checks = {}
    for phase in PHASES:
        candidate = np.load(RESULTS / f"candidate_{phase}.npz")
        validation = np.load(RESULTS / f"validation_{phase}.npz")
        test = np.load(RESULTS / f"test_exact_{phase}.npz")
        order = selection[f"{phase}_order"]
        checks[f"{phase}_candidate_count_8192"] = len(candidate["T"]) == 8192
        checks[f"{phase}_validation_count_1536"] = len(validation["T"]) == 1536
        checks[f"{phase}_candidate_excludes_bands"] = not np.any(in_bands(candidate["T"]))
        checks[f"{phase}_validation_excludes_bands"] = not np.any(in_bands(validation["T"]))
        checks[f"{phase}_test_contains_bands"] = set(np.unique(test["T"][in_bands(test["T"])])) == set(
            np.r_[np.arange(302, 309), np.arange(322, 329), np.arange(342, 349)]
        )
        checks[f"{phase}_candidate_has_both_sources"] = (
            np.any(candidate["is_equilibrium"] > 0.5)
            and np.any(candidate["is_equilibrium"] < 0.5)
        )
        chosen_sources = candidate["is_equilibrium"][order[:selected]]
        checks[f"{phase}_selected_has_both_sources"] = (
            np.any(chosen_sources > 0.5) and np.any(chosen_sources < 0.5)
        )
        checks[f"{phase}_support_order_unique"] = len(np.unique(order)) == len(order)
        checks[f"{phase}_support_order_in_range"] = bool(
            np.all((order >= 0) & (order < len(candidate["T"])))
        )

    objective = {phase: {} for phase in PHASES}
    for row in convergence:
        phase = PHASES[int(float(row["phase_id"]))]
        objective[phase][int(float(row["M"]))] = float(row["tau_objective"])
    counts = sorted(set(objective["liquid"]) & set(objective["vapor"]))
    minima = {phase: min(objective[phase].values()) for phase in PHASES}
    selected_by_rule = next(
        count for count in counts
        if all(objective[phase][count] <= 1.10 * minima[phase] for phase in PHASES)
    )
    checks["selected_support_count_matches_declared_rule"] = selected == selected_by_rule

    numeric_state_columns = [
        name for name in state_errors[0]
        if name not in {"phase_id", "heldout"}
    ]
    checks["all_state_errors_finite"] = all(
        np.all(np.isfinite([float(row[name]) for row in state_errors]))
        for name in numeric_state_columns
    )
    selected_curves = [row for row in curves if int(float(row["M"])) == selected]
    checks["selected_model_vle_all_converged"] = all(
        float(row["success"]) > 0.5 for row in selected_curves
    )
    metric_keys = [(int(float(row["M"])), float(row["T"])) for row in metrics]
    checks["vle_metric_rows_unique"] = len(metric_keys) == len(set(metric_keys))

    required_products = [
        "summary.json", "quality_report.json", "configuration.json",
        "support_convergence.csv", "state_errors.csv", "vle_curves.csv",
        "vle_metrics.csv", "state_error_by_temperature.csv",
        "holdout_band_state_summary.csv", "selected_model_vle_by_temperature.csv",
        "holdout_band_vle_summary.csv", "old_vs_bandgap_key_metrics.csv", "README.md",
    ]
    checks["all_delivery_products_present"] = all(
        (RESULTS / name).is_file() and (RESULTS / name).stat().st_size > 0
        for name in required_products
    )
    checks = {key: bool(value) for key, value in checks.items()}
    checks["all_checks_pass"] = all(checks.values())

    output = RESULTS / "independent_validation_report.json"
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not checks["all_checks_pass"]:
        raise SystemExit("one or more validation checks failed")


if __name__ == "__main__":
    main()
