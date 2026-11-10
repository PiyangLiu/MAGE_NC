"""Run the nonisothermal CH4/n-C4 vapour--liquid-equilibrium case.

The driver creates the candidate, validation and independent band-test pools;
selects the finite supports; evaluates state and coexistence errors; writes the
machine-readable products; and renders manuscript Figure 4 in Python.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
CODE = HERE.parents[1]
REPO = CODE.parent
sys.path.insert(0, str(CODE / "figure4_nonisothermal"))
sys.path.insert(0, str(CODE / "figure_generation"))
import run_figure4 as calculation  # noqa: E402
from plot_figures_2_to_6 import plot_figure4  # noqa: E402


def _pad(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    length = max(len(v) for v in columns.values())
    return pd.DataFrame({name: np.pad(np.asarray(values, float),
                                     (0, length - len(values)),
                                     constant_values=np.nan)
                         for name, values in columns.items()})


def prepare_plot_data(results: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    quality = json.loads((results / "quality_report.json").read_text(encoding="utf-8"))
    selected_m = int(quality["selected_support_count"])
    selection = np.load(results / "support_selection.npz")
    candidates = {phase: np.load(results / f"candidate_{phase}.npz")
                  for phase in ("liquid", "vapor")}
    coverage = {}
    for phase in ("liquid", "vapor"):
        pool = candidates[phase]
        chosen = np.zeros(len(pool["T"]), dtype=int)
        chosen[selection[f"{phase}_order"][:selected_m].astype(int)] = 1
        coverage[f"temperature_{phase}_K"] = pool["T"]
        coverage[f"pressure_{phase}_MPa"] = pool["P"] / 1e6
        coverage[f"selected_{phase}"] = chosen
    pd.DataFrame(coverage).to_csv(destination / "figure4_coverage.csv", index=False)

    state = pd.read_csv(results / "state_errors.csv")
    energy = {}
    for phase_id, phase in ((0, "liquid"), (1, "vapor")):
        for heldout, label in ((0, "anchor"), (1, "heldout")):
            values = state.loc[(state.phase_id == phase_id) &
                               (state.heldout == heldout),
                               "u_molar_absolute_kJ_mol"].to_numpy()
            energy[f"{phase}_{label}_kJ_mol"] = values
    _pad(energy).to_csv(destination / "figure4_energy_errors.csv", index=False)

    curves = pd.read_csv(results / "vle_curves.csv")
    curves = curves[curves.M == selected_m].copy()
    curves.to_csv(destination / "figure4_curves.csv", index=False)
    metrics = pd.read_csv(results / "vle_metrics.csv")
    metrics = metrics[metrics.M == selected_m].rename(columns={"T": "temperature_K"})
    metrics[["temperature_K", "liquid_mae", "vapor_mae", "heldout"]].to_csv(
        destination / "figure4_vle_errors.csv", index=False)
    (destination / "plot_manifest.json").write_text(
        json.dumps({"selected_support_count": selected_m,
                    "source_result_directory": str(results)}, indent=2) + "\n",
        encoding="utf-8")


def write_si_tables(results: Path):
    """Write the SI temperature-band summaries from this case run."""
    state = pd.read_csv(results / "state_errors.csv")
    state = state.rename(columns={"T": "temperature_K", "phase_id": "phase"})
    state["phase"] = state["phase"].map({0: "liquid", 1: "vapor"}).fillna(state["phase"])
    rows = []
    for (phase, temperature), group in state.groupby(["phase", "temperature_K"], sort=True):
        row = {"phase": phase, "temperature_K": float(temperature), "state_count": int(len(group))}
        for column in ("phi_relative", "gradient_relative", "concentration_relative", "u_molar_absolute_kJ_mol", "u_molar_relative"):
            if column in group:
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_p95"] = float(group[column].quantile(.95))
                row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(results / "state_error_by_temperature.csv", index=False)

    bands = ((304.5, 305.5), (324.5, 325.5), (344.5, 345.5))
    band_rows = []
    for number, (lower, upper) in enumerate(bands, 1):
        mask = state.temperature_K.between(lower, upper)
        for phase, group in state.loc[mask].groupby("phase", sort=True):
            row = {"band": number, "lower_temperature_K": lower,
                   "upper_temperature_K": upper, "centre_temperature_K": (lower + upper) / 2,
                   "phase": phase, "temperature_layers": int(group.temperature_K.nunique())}
            for column in ("phi_relative", "gradient_relative", "concentration_relative", "u_molar_absolute_kJ_mol", "u_molar_relative"):
                if column in group:
                    row[f"{column}_mean"] = float(group[column].mean())
                    row[f"{column}_p95"] = float(group[column].quantile(.95))
                    row[f"{column}_max"] = float(group[column].max())
            band_rows.append(row)
    pd.DataFrame(band_rows).to_csv(results / "holdout_band_state_summary.csv", index=False)

    curves = pd.read_csv(results / "vle_curves.csv")
    curves["temperature_K"] = curves["T"]
    curves["success"] = curves["success"].astype(float)
    curve_rows = []
    for temperature, group in curves.groupby("temperature_K", sort=True):
        good = group[group.success > .5]
        curve_rows.append({"temperature_K": float(temperature), "state_count": int(len(good)),
                           "liquid_mae": float(np.mean(np.abs(good.x_mage - good.x_pr))) if len(good) else np.nan,
                           "vapor_mae": float(np.mean(np.abs(good.y_mage - good.y_pr))) if len(good) else np.nan})
    selected = pd.DataFrame(curve_rows)
    selected.to_csv(results / "selected_model_vle_by_temperature.csv", index=False)
    vle_rows = []
    for number, (lower, upper) in enumerate(bands, 1):
        group = selected[selected.temperature_K.between(lower, upper)]
        if len(group):
            vle_rows.append({"band": number, "lower_temperature_K": lower,
                             "upper_temperature_K": upper,
                             "centre_temperature_K": (lower + upper) / 2,
                             "temperature_layers": int(len(group)),
                             "liquid_mae_mean": float(group.liquid_mae.mean()),
                             "vapor_mae_mean": float(group.vapor_mae.mean()),
                             "liquid_mae_max": float(group.liquid_mae.max()),
                             "vapor_mae_max": float(group.vapor_mae.max())})
    pd.DataFrame(vle_rows).to_csv(results / "holdout_band_vle_summary.csv", index=False)


def run(quick: bool, reuse_pools: bool, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    calculation.run(quick=quick, reuse_pools=reuse_pools, output_dir=output)
    write_si_tables(output)
    plot_data = output / "plot_data"
    prepare_plot_data(output, plot_data)
    plot_figure4(plot_data, output / "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small deterministic smoke test")
    parser.add_argument("--reuse-pools", action="store_true", help="reuse matching NPZ pools in the output directory")
    parser.add_argument("--output-dir", type=Path, default=REPO / "case_results" / "nonisothermal_vle")
    args = parser.parse_args()
    run(args.quick, args.reuse_pools, args.output_dir)
    print(f"Nonisothermal VLE case complete: {args.output_dir}")


if __name__ == "__main__":
    main()
