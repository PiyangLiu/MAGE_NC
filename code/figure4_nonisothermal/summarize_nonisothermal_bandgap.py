"""Create concise tables and a scientific readme for the band-gap benchmark."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nonisothermal_bandgap"
OLD_SUMMARY = ROOT / "results" / "nonisothermal_summary.json"
BANDS = ((302.0, 308.0), (322.0, 328.0), (342.0, 348.0))
CENTRES = (305.0, 325.0, 345.0)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(np.asarray(list(values), float)))


def maximum(values):
    return float(np.max(np.asarray(list(values), float)))


def state_table(summary):
    rows = []
    for phase in ("liquid", "vapor"):
        phase_data = summary["state_error_summary_by_temperature"][phase]
        for temperature_text, metrics in phase_data.items():
            temperature = float(temperature_text)
            rows.append({
                "phase": phase,
                "temperature_K": temperature,
                "heldout": int(any(lo <= temperature <= hi for lo, hi in BANDS)),
                "phi_relative_mean": metrics["phi_relative"]["mean"],
                "phi_relative_p95": metrics["phi_relative"]["p95"],
                "phi_relative_max": metrics["phi_relative"]["max"],
                "gradient_relative_mean": metrics["gradient_relative"]["mean"],
                "gradient_relative_p95": metrics["gradient_relative"]["p95"],
                "concentration_relative_mean": metrics["concentration_relative"]["mean"],
                "u_molar_absolute_mean_kJ_mol": metrics["u_molar_absolute_kJ_mol"]["mean"],
                "u_molar_absolute_p95_kJ_mol": metrics["u_molar_absolute_kJ_mol"]["p95"],
                "u_molar_relative_mean": metrics["u_molar_relative"]["mean"],
                "u_molar_relative_p95": metrics["u_molar_relative"]["p95"],
            })
    return sorted(rows, key=lambda row: (row["phase"], row["temperature_K"]))


def band_state_table(state_rows):
    rows = []
    metric_names = [
        name for name in state_rows[0]
        if name not in {"phase", "temperature_K", "heldout"}
    ]
    for band_index, (lower, upper) in enumerate(BANDS, start=1):
        for phase in ("liquid", "vapor"):
            subset = [
                row for row in state_rows
                if row["phase"] == phase
                and lower <= row["temperature_K"] <= upper
            ]
            row = {
                "band": band_index,
                "lower_temperature_K": lower,
                "upper_temperature_K": upper,
                "centre_temperature_K": (lower + upper) / 2,
                "phase": phase,
                "temperature_layers": len(subset),
            }
            for name in metric_names:
                row[f"mean_over_temperature_layers__{name}"] = mean(
                    item[name] for item in subset
                )
                row[f"maximum_over_temperature_layers__{name}"] = maximum(
                    item[name] for item in subset
                )
            rows.append(row)
    return rows


def selected_vle_table(summary, all_rows):
    selected = int(summary["support_selection"]["selected_support_count"])
    rows = []
    for raw in all_rows:
        if int(float(raw["M"])) != selected:
            continue
        rows.append({
            "support_count_per_phase": selected,
            "temperature_K": float(raw["T"]),
            "heldout": int(float(raw["heldout"])),
            "reference_states": int(float(raw["n_reference"])),
            "converged_states": int(float(raw["n_converged"])),
            "liquid_composition_MAE": float(raw["liquid_mae"]),
            "liquid_composition_max_abs_error": float(raw["liquid_max"]),
            "vapor_composition_MAE": float(raw["vapor_mae"]),
            "vapor_composition_max_abs_error": float(raw["vapor_max"]),
        })
    return sorted(rows, key=lambda row: row["temperature_K"])


def band_vle_table(vle_rows):
    output = []
    for band_index, (lower, upper) in enumerate(BANDS, start=1):
        subset = [
            row for row in vle_rows
            if lower <= row["temperature_K"] <= upper
        ]
        output.append({
            "band": band_index,
            "lower_temperature_K": lower,
            "upper_temperature_K": upper,
            "centre_temperature_K": (lower + upper) / 2,
            "temperature_layers": len(subset),
            "reference_states": sum(row["reference_states"] for row in subset),
            "converged_states": sum(row["converged_states"] for row in subset),
            "mean_of_liquid_composition_MAE": mean(
                row["liquid_composition_MAE"] for row in subset
            ),
            "maximum_liquid_composition_abs_error": maximum(
                row["liquid_composition_max_abs_error"] for row in subset
            ),
            "mean_of_vapor_composition_MAE": mean(
                row["vapor_composition_MAE"] for row in subset
            ),
            "maximum_vapor_composition_abs_error": maximum(
                row["vapor_composition_max_abs_error"] for row in subset
            ),
        })
    return output


def old_new_table(old_summary, new_summary):
    old = {float(row["T"]): row for row in old_summary["errors"]}
    rows = []
    for temperature in CENTRES:
        for phase in ("liquid", "vapor"):
            before = old[temperature][phase]
            after = new_summary["state_error_summary_by_temperature"][phase][str(temperature)]
            rows.append({
                "temperature_K": temperature,
                "phase": phase,
                "old_phi_relative_mean": before["phi_rel_mae"],
                "bandgap_phi_relative_mean": after["phi_relative"]["mean"],
                "old_u_molar_absolute_mean_kJ_mol": before["u_molar_abs_mae_kJ_mol"],
                "bandgap_u_molar_absolute_mean_kJ_mol": after["u_molar_absolute_kJ_mol"]["mean"],
                "old_u_molar_relative_mean": before["u_rel_mae"],
                "bandgap_u_molar_relative_mean": after["u_molar_relative"]["mean"],
            })
    return rows


def make_readme(summary, band_states, band_vle):
    supports = summary["support_selection"]["selected_support_composition"]
    selected = summary["support_selection"]["selected_support_count"]
    lines = [
        "# Strengthened nonisothermal benchmark",
        "",
        "This directory contains the completed CH4/n-C4 nonisothermal PR/MAGE "
        "calculation with three withheld temperature bands. The retained Figure 4 "
        "files and the earlier nonisothermal calculation are not overwritten.",
        "",
        "## Frozen problem definition",
        "",
        "- Temperature domain: 295–350 K.",
        "- Pressure domain: 0.50–7.00 MPa.",
        "- Withheld closed bands: 302–308, 322–328, and 342–348 K.",
        "- Candidate equilibrium layers: 1 K spacing outside the withheld bands.",
        "- Independent validation equilibrium layers: the corresponding 0.5 K-offset "
        "  grid outside the withheld bands.",
        "- Each phase candidate pool: 8,192 states, including 858 coexistence-branch "
        "  states and 7,334 stable non-coexistence homogeneous states.",
        "- Each phase validation pool: 1,536 states, including 465 coexistence-branch "
        "  states and 1,071 stable non-coexistence homogeneous states.",
        "- Held-out homogeneous test states: 192 per phase and temperature at every "
        "  integer temperature in each withheld band, plus 295, 300, 315, 335, and 350 K.",
        "- Potential and all derivatives use the same PR Helmholtz teacher with the "
        "  Setzmann–Wagner methane and Bücker–Wagner n-butane ideal-gas caloric terms.",
        "",
        "## Support selection",
        "",
        "At each greedy iteration, the independent validation state with the largest "
        "scaled potential-plus-gradient error is identified, and the nearest unused "
        "candidate state is appended. Held-out-band data are not used for support or "
        "smoothing selection.",
        "",
        f"The evaluated counts were 64, 128, 256, 384, 512, 768, 1,024, and 1,536 "
        f"supports per phase. The selected count is {selected:,}: the smallest count "
        "for which both branch validation objectives are within 10% of their respective "
        "minimum over the evaluated counts.",
        "",
        f"- Liquid supports: {supports['liquid']['equilibrium_count']} coexistence and "
        f"{supports['liquid']['stable_homogeneous_count']} stable homogeneous states.",
        f"- Vapour supports: {supports['vapor']['equilibrium_count']} coexistence and "
        f"{supports['vapor']['stable_homogeneous_count']} stable homogeneous states.",
        "- Selected smoothing parameter: 0.0625 for each branch.",
        "",
        "## Principal held-out results",
        "",
        "Band-averaged metrics are available in `holdout_band_state_summary.csv` and "
        "`holdout_band_vle_summary.csv`. The relative internal-energy error can become "
        "large when the sensible-energy reference crosses or approaches zero; therefore "
        "the absolute molar error in kJ mol-1 is retained alongside the relative value.",
        "",
        "| Band (K) | Phase | mean relative Phi error | mean absolute u error (kJ mol-1) |",
        "|---|---:|---:|---:|",
    ]
    for row in band_states:
        lines.append(
            f"| {row['lower_temperature_K']:.0f}–{row['upper_temperature_K']:.0f} "
            f"| {row['phase']} "
            f"| {row['mean_over_temperature_layers__phi_relative_mean']:.4g} "
            f"| {row['mean_over_temperature_layers__u_molar_absolute_mean_kJ_mol']:.4g} |"
        )
    lines += [
        "",
        "| Band (K) | liquid composition MAE | vapour composition MAE | converged/reference |",
        "|---|---:|---:|---:|",
    ]
    for row in band_vle:
        lines.append(
            f"| {row['lower_temperature_K']:.0f}–{row['upper_temperature_K']:.0f} "
            f"| {row['mean_of_liquid_composition_MAE']:.4g} "
            f"| {row['mean_of_vapor_composition_MAE']:.4g} "
            f"| {row['converged_states']}/{row['reference_states']} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `summary.json`: complete configuration, support composition, per-temperature "
        "  state-error summaries, VLE summaries, and quality checks.",
        "- `support_convergence.csv`: validation errors versus support count.",
        "- `state_errors.csv`: every independent homogeneous test-state error.",
        "- `vle_curves.csv`: every PR and MAGE coexistence state used in the convergence "
        "  and final band evaluations.",
        "- `state_error_by_temperature.csv`: compact per-temperature homogeneous-state table.",
        "- `holdout_band_state_summary.csv`: band-level homogeneous-state summary.",
        "- `selected_model_vle_by_temperature.csv`: final-model VLE errors by temperature.",
        "- `holdout_band_vle_summary.csv`: band-level VLE summary.",
        "- `old_vs_bandgap_key_metrics.csv`: same-definition comparison with the retained "
        "  earlier calculation at 305, 325, and 345 K.",
        "- Candidate, validation, and exact-temperature test pools are stored as NPZ files.",
        "",
        "Run `run_nonisothermal_bandgap.py` to recompute the benchmark and "
        "`validate_nonisothermal_bandgap.py` to repeat the independent checks.",
    ]
    return "\n".join(lines) + "\n"


def main():
    summary = read_json(RESULTS / "summary.json")
    state_rows = state_table(summary)
    band_states = band_state_table(state_rows)
    vle_rows = selected_vle_table(summary, read_csv(RESULTS / "vle_metrics.csv"))
    band_vle = band_vle_table(vle_rows)
    old_new = (
        old_new_table(read_json(OLD_SUMMARY), summary)
        if OLD_SUMMARY.is_file() else None
    )

    write_csv(RESULTS / "state_error_by_temperature.csv", state_rows)
    write_csv(RESULTS / "holdout_band_state_summary.csv", band_states)
    write_csv(RESULTS / "selected_model_vle_by_temperature.csv", vle_rows)
    write_csv(RESULTS / "holdout_band_vle_summary.csv", band_vle)
    if old_new is not None:
        write_csv(RESULTS / "old_vs_bandgap_key_metrics.csv", old_new)
    (RESULTS / "README.md").write_text(
        make_readme(summary, band_states, band_vle), encoding="utf-8"
    )
    print(f"Wrote concise result products to {RESULTS}")


if __name__ == "__main__":
    main()
