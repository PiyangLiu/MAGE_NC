"""Build plot-ready data for the Python Figure 2--6 renderer.

The script combines the accepted unified recomputations with the unchanged
reference curves and phase-space samples used by the current manuscript.  It
does not fit, smooth, or interpolate any scientific result.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parent / "unified_selection"
PACKAGE = HERE.parent.parent
OUT = HERE / "data"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def table_payload(frame: pd.DataFrame) -> dict:
    data = {}
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            data[name] = [bool(value) for value in series]
        elif pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy()
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite values in column {name}")
            data[name] = values.tolist()
        else:
            data[name] = series.fillna("").astype(str).tolist()
    return {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "data": data,
    }


def metric_at(branch: dict, count: int) -> dict:
    return next(row for row in branch["test_metrics"] if int(row["M"]) == count)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_static_plot_data() -> list[Path]:
    source = PACKAGE / "source_data"
    sources = {
        "figure2_vle.csv": source / "Figure2" / "figure2_vle.csv",
        "figure2_ecdf.csv": source / "Figure2" / "figure2_ecdf.csv",
        "figure3_metrics.csv": source / "Figure3" / "figure3_metrics.csv",
        "figure3_vle_M16.csv": source / "Figure3" / "figure3_vle_M16.csv",
        "figure3_vle_M64.csv": source / "Figure3" / "figure3_vle_M64.csv",
        "figure3_vle_M1024.csv": source / "Figure3" / "figure3_vle_M1024.csv",
    }
    written = []
    for name, source_path in sources.items():
        target = OUT / name
        shutil.copyfile(source_path, target)
        written.append(target)

    # Figure 2(d) reads the independently stored coexistence pressures from
    # this summary.  The accepted unified 1024-support prefix is identical.
    summary = PACKAGE / "data" / "figure2_summary.json"
    target = OUT / "figure2_summary.json"
    shutil.copyfile(summary, target)
    written.append(target)

    # The current nonisothermal support counts are phase-specific; the
    # plot-ready tables in source_data/Figure4 are authoritative here.
    noniso = source / "Figure4"
    for name in (
        "figure4_coverage.csv",
        "figure4_energy_errors.csv",
        "figure4_curves.csv",
        "figure4_vle_errors.csv",
    ):
        target = OUT / name
        shutil.copyfile(noniso / name, target)
        written.append(target)
    return written


def build_generality_data() -> Path:
    authoritative = PACKAGE / "source_data" / "Figure5" / "figure5_generality_plot_data.json"
    if authoritative.is_file():
        target = OUT / "figure5_generality_plot_data.json"
        shutil.copyfile(authoritative, target)
        return target

    template_path = PACKAGE / "data" / "gerg_lle_plot_data.json"
    payload = load_json(template_path)
    payload["figure"] = 5
    payload["purpose"] = (
        "Plot-ready Figure 5 data using the accepted unified GERG and LLE "
        "support-selection results."
    )

    gerg = load_json(UNIFIED / "results" / "gerg_selection.json")
    counts = [96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096]
    rows = []
    for count in counts:
        liquid = metric_at(gerg["branches"]["liquid"], count)
        vapor = metric_at(gerg["branches"]["vapor"], count)
        rows.append(
            {
                "M": count,
                "phi_liquid": liquid["mean_phi"],
                "phi_vapor": vapor["mean_phi"],
                # mean_u is the componentwise relative derivative diagnostic
                # used in the accepted comparison with the manuscript metric.
                "gradient_liquid": liquid["mean_u"],
                "gradient_vapor": vapor["mean_u"],
            }
        )
    gerg_frame = pd.DataFrame(rows)
    panel = payload["panels"]["b_c_GERG_convergence"]
    panel["selected_support_count"] = {"liquid": 3264, "vapor": 1592}
    panel["metrics"] = table_payload(gerg_frame)

    # Use the current manuscript source tables for LLE.  Older selector
    # archives contain a different support order and must not regenerate the
    # published coexistence curve.
    lle_errors_path = PACKAGE / "source_data" / "Figure5" / "Figure5e_LLE_errors.csv"
    lle_coex_path = PACKAGE / "source_data" / "Figure5" / "Figure5d_LLE_coexistence.json"
    lle_errors = pd.read_csv(lle_errors_path)
    lle_coex = load_json(lle_coex_path)
    coex = {
        int(row.M): {
            "M": int(row.M),
            "x1": float(row.x1),
            "x2": float(row.x2),
            "absolute_max": float(row.coexistence_error),
        }
        for row in lle_errors.itertuples(index=False)
    }
    selected_lle = lle_coex["MAGE"]
    coex[int(selected_lle["M_L1"])] = {
        "M": int(selected_lle["M_L1"]),
        "x1": float(selected_lle["x_L1"]),
        "x2": float(selected_lle["x_L2"]),
        "absolute_max": float(selected_lle["maximum_absolute_error"]),
    }
    lle_rows = []
    for count in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        value = coex.get(count, coex[max(coex)])
        lle_rows.append({
            "M": count,
            "x1": value["x1"],
            "x2": value["x2"],
            "coexistence_error": value["absolute_max"],
        })
    lle_frame = pd.DataFrame(lle_rows)
    payload["panels"]["e_LLE_convergence"]["results"] = table_payload(lle_frame)
    # Report the branchwise prefixes used by the current manuscript.
    selected_m = int(selected_lle["M_L1"])
    selected = coex[selected_m]
    payload["panels"]["d_LLE_free_energy_and_coexistence"]["MAGE_coexistence_shown"] = {
        "M": int(selected["M"]),
        "x_L1": float(selected["x1"]),
        "x_L2": float(selected["x2"]),
        "maximum_absolute_error": float(selected["absolute_max"]),
    }
    payload["integrity"].update(
        {
            "GERG_support_counts": len(gerg_frame),
            "LLE_support_counts": len(lle_frame),
            "unified_acceptance_status": "accepted",
            "selected_GERG_supports_per_phase": {"liquid": 3264, "vapor": 1592},
            "selected_LLE_supports_per_branch": {"L1": 192, "L2": 184},
        }
    )
    target = OUT / "figure5_generality_plot_data.json"
    write_json(target, payload)
    return target


def build_flow_data() -> Path:
    source = UNIFIED / "results" / "ternary_flow_authoritative"
    summary = load_json(source / "summary.json")
    states = pd.read_csv(source / "combined_global_states.csv")
    events = pd.read_csv(source / "combined_phase_events.csv")
    if len(states) != 552 or len(events) != 24:
        raise ValueError("unexpected authoritative flow result dimensions")

    payload = {
        "schema_version": "1.0",
        "figure": 6,
        "scientific_case": "Pressure-coupled ternary fully implicit flow",
        "configuration": summary["benchmark"],
        "calculation_summary": {
            "MAGE_solver": summary["MAGE_solver"],
            "PR_solver": summary["PR_solver"],
            "comparison": summary["comparison"],
            "support_selection": summary["support_selection"],
        },
        "datasets": {
            "combined_global_states": table_payload(states),
            "combined_phase_events": table_payload(events),
        },
        "integrity": {
            "all_numeric_values_finite": True,
            "space_time_states": int(len(states)),
            "phase_events": int(len(events)),
            "phase_state_mismatches": int(summary["comparison"]["phase_state_mismatches"]),
            "phase_event_identity": bool(
                summary["comparison"]["phase_event_step_cell_transition_identity"]
            ),
            "unified_acceptance_status": "accepted",
        },
    }
    target = OUT / "figure6_flow_plot_data.json"
    write_json(target, payload)
    return target


def validate_data() -> dict:
    csv_files = sorted(OUT.glob("*.csv"))
    for path in csv_files:
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if np.any(np.isinf(numeric)):
            raise ValueError(f"infinite plotting data in {path.name}")
        if path.name == "figure4_energy_errors.csv":
            # The four distributions have unequal sample counts and are
            # stored as padded columns.  NaN is therefore a table-layout
            # sentinel here, not a calculated thermodynamic value.
            finite_counts = frame.notna().sum().to_dict()
            expected_counts = {
                "liquid_anchor_kJ_mol": 960,
                "liquid_heldout_kJ_mol": 1728,
                "vapor_anchor_kJ_mol": 960,
                "vapor_heldout_kJ_mol": 1728,
            }
            if finite_counts != expected_counts:
                raise ValueError(
                    "unexpected Figure 4 energy distribution lengths: "
                    f"{finite_counts}"
                )
        elif not np.all(np.isfinite(numeric)):
            raise ValueError(f"non-finite plotting data in {path.name}")

    generality = load_json(OUT / "figure5_generality_plot_data.json")
    flow = load_json(OUT / "figure6_flow_plot_data.json")
    checks = {
        "binary_selected_supports_per_phase": 1024,
        "nonisothermal_selected_supports_per_phase": {"liquid": 1592, "vapor": 1392},
        "gerg_selected_supports_per_phase": generality["panels"]["b_c_GERG_convergence"]["selected_support_count"],
        "lle_selected_supports_per_branch": {"L1": 192, "L2": 184},
        "flow_selected_supports_per_phase": int(
            flow["calculation_summary"]["support_selection"]["supports_per_phase"]
        ),
        "flow_space_time_states": int(
            flow["datasets"]["combined_global_states"]["row_count"]
        ),
        "flow_phase_events": int(
            flow["datasets"]["combined_phase_events"]["row_count"]
        ),
    }
    expected = {
        "binary_selected_supports_per_phase": 1024,
        "nonisothermal_selected_supports_per_phase": {"liquid": 1592, "vapor": 1392},
        "gerg_selected_supports_per_phase": {"liquid": 3264, "vapor": 1592},
        "lle_selected_supports_per_branch": {"L1": 192, "L2": 184},
        "flow_selected_supports_per_phase": 1280,
        "flow_space_time_states": 552,
        "flow_phase_events": 24,
    }
    if checks != expected:
        raise ValueError(f"plot-data validation failed: {checks}")
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = copy_static_plot_data()
    files.extend([build_generality_data(), build_flow_data()])
    checks = validate_data()
    manifest = {
        "description": (
            "Python plot-ready products for accepted unified support-selection "
            "recomputations; no scientific interpolation or refitting."
        ),
        "checks": checks,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        },
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": "accepted", **checks}, indent=2))


if __name__ == "__main__":
    main()
