#!/usr/bin/env python3
"""Validate the packaged calculations and write machine-readable delivery reports."""
from __future__ import annotations

import csv
import json
import py_compile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    entry_points = {
        "figure2_binary_equilibrium": "figure2_binary_equilibrium/run_figure2.py",
        "figure3_support_compression": "figure3_support_compression/run_figure3.py",
        "figure4_nonisothermal_strengthened": "figure4_nonisothermal/run_figure4.py",
        "figure4_nonisothermal_retained": "figure4_nonisothermal/run_retained_figure4.py",
        "figure5_ternary_fully_implicit": "unified_selection/run_ternary_unified.py",
        "figure6_gerg2008": "figure6_gerg2008/run_figure6_gerg.py",
        "figure6_lle": "figure6_lle/run_figure6_lle.py",
        "caseB_same_support_flow": "unified_selection/run_ternary_caseB_unified.py",
        "non_pr_gas_brine": "non_pr_storage_or_brine_validation/run_brine_validation.py",
        "non_pr_cpa_companion": "non_pr_storage_or_brine_validation/cpa_three_phase_reference/run_cpa_three_phase.py",
        "tangent_plane_support_audit": "reports/audit_tangent_plane_supports.py",
    }
    key_results = {
        "figure4_strengthened": "figure4_nonisothermal/strengthened_results/summary.json",
        "figure5_caseA": "../data/coupled_ternary_flow_plot_data.json",
        "primary_flow": "../source_data/Figure6_and_SupplementaryFigure1/calculation_summary.json",
        "caseB": "../source_data/Supplementary_Note_5_2_second_flow/caseB_summary.json",
        "flow_domain_audit": "reports/flow_domain_audit.json",
        "tangent_plane_support_validity": "reports/tangent_plane_support_validity.json",
        "gas_brine": "non_pr_storage_or_brine_validation/brine_validation_summary.json",
    }
    required = [BASE / p for p in entry_points.values()]
    required += [(BASE / p).resolve() for p in key_results.values()]
    required += [
        REPORTS / "support_state_filtering.json",
        REPORTS / "potential_scaling_parameters.csv",
        REPORTS / "support_domain_bounds.csv",
        REPORTS / "offline_pool_conventions.md",
        REPORTS / "si_parameter_table.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Missing delivery files: " + ", ".join(missing))

    compiled = []
    for rel in entry_points.values():
        path = BASE / rel
        py_compile.compile(str(path), doraise=True)
        compiled.append(rel)
    for path in (BASE / "common").glob("*.py"):
        py_compile.compile(str(path), doraise=True)
        compiled.append(path.relative_to(BASE).as_posix())

    primary = load_json(BASE / key_results["primary_flow"])
    primary_comparison = primary["comparison"]
    require(primary_comparison["space_time_states"] == 552, "Primary flow does not contain 552 space-time states")
    require(primary_comparison["phase_state_match_fraction"] == 1.0, "Primary flow phase-state agreement is below 100%")
    require(primary_comparison["phase_state_mismatches"] == 0, "Primary flow contains phase-state mismatches")
    require(
        primary_comparison["phase_event_step_cell_transition_identity"] is True,
        "Primary flow phase-event transitions are not identical",
    )
    require(
        primary_comparison["phase_events_MAGE"] == primary_comparison["phase_events_PR"],
        "Primary flow phase-event counts differ",
    )
    combined = read_csv(BASE / "../source_data/Figure6_and_SupplementaryFigure1/combined_global_states.csv")
    require(len(combined) == 552, "Combined primary-flow history does not contain 552 rows")
    require(all(row["phase_match"].strip().lower() == "true" for row in combined), "A primary-flow CSV phase label differs")
    events = read_csv(BASE / "../source_data/Figure6_and_SupplementaryFigure1/combined_phase_events.csv")
    require(all(row["_merge"] == "both" for row in events), "A primary-flow phase event is not shared")

    case_b = load_json(BASE / key_results["caseB"])
    comparison = case_b["comparison"]
    require(comparison["space_time_states"] == 558, "Second flow does not contain 558 space-time states")
    require(comparison["phase_state_match_fraction"] == 1.0, "Second-flow phase-state agreement is below 100%")
    require(comparison["phase_state_mismatches"] == 0, "Second flow contains phase-state mismatches")
    require(comparison["phase_event_step_cell_transition_identity"] is True, "Second-flow phase-event transitions are not identical")
    require(comparison["phase_events_MAGE"] == comparison["phase_events_PR"], "Second-flow phase-event counts differ")
    combined_second = read_csv(BASE / "../source_data/Supplementary_Note_5_2_second_flow/combined_global_states.csv")
    require(len(combined_second) == 558, "Combined second-flow history does not contain 558 rows")
    require(all(row["phase_match"].strip().lower() == "true" for row in combined_second), "A second-flow CSV phase label differs")
    events_second = read_csv(BASE / "../source_data/Supplementary_Note_5_2_second_flow/combined_phase_events.csv")
    require(all(row["_merge"] == "both" for row in events_second), "A second-flow phase event is not shared")

    domain = load_json(REPORTS / "flow_domain_audit.json")
    tangent = load_json(REPORTS / "tangent_plane_support_validity.json")
    require(tangent["status"] == "passed", "Tangent-plane support audit did not complete")
    require(tangent["all_selected_ternary_supports_accepted"], "A selected ternary support failed stable branch acceptance")
    for branch in ("liquid", "vapor"):
        check = tangent["figure5_ternary_selected_support_recheck"][branch]
        require(check["selected_support_count"] == 1280, f"Incomplete Figure 5 {branch} support recheck")
        require(check["unique_teacher_state_matches"] == 1280, f"Duplicate Figure 5 {branch} support mapping")
        require(check["stable_branch_invalid_count"] == 0, f"Unstable Figure 5 {branch} support detected")
        require(check["returned_branch_composition_mismatch_count"] == 0, f"Figure 5 {branch} branch composition mismatch")
    for name in ("case_A", "case_B"):
        audit = domain[name]
        require(audit["state_count"] == 552, f"{name} domain audit is incomplete")
        require(audit["all_online_dual_states_inside_selected_componentwise_support_box"], f"{name} leaves the support box")
        require(audit["all_online_pressure_inside_declared_1p25_to_3p25_MPa"], f"{name} leaves the pressure domain")
        require(audit["minimum_distance_to_solver_eta_bounds"] > 0.0, f"{name} touches a dual solver bound")
        hull = audit["selected_support_convex_hull_test_on_eta_1_to_eta_3"]
        require(hull["available"] and hull["inside_fraction"] == 1.0, f"{name} leaves the selected-support convex hull")

    f4_quality = load_json(BASE / "figure4_nonisothermal/strengthened_results/quality_report.json")
    require(f4_quality["all_required_checks_pass"], "Strengthened Figure 4 quality checks failed")
    require(
        f4_quality["selected_supports_per_phase"] == {"liquid": 1592, "vapor": 1392},
        "Strengthened Figure 4 support counts changed",
    )

    brine = load_json(BASE / key_results["gas_brine"])
    require(brine["status"] == "completed", "Gas-brine validation did not complete")
    require(brine["derivative_check"]["maximum_relative_error"] < 1.0e-8, "Gas-brine derivative audit failed")
    max_coexist = max(row["chemical_potential_residual"] for row in brine["coexistence_checks"])
    require(max_coexist < 1.0e-10, "Gas-brine coexistence residual audit failed")

    validation = {
        "status": "passed",
        "python_entry_points_compiled": sorted(compiled),
        "required_files_present": len(required),
        "caseB_phase_validation": {
            "space_time_states": primary_comparison["space_time_states"],
            "matched_states": 552 - primary_comparison["phase_state_mismatches"],
            "match_fraction": primary_comparison["phase_state_match_fraction"],
            "mismatches": primary_comparison["phase_state_mismatches"],
            "MAGE_phase_events": primary_comparison["phase_events_MAGE"],
            "PR_phase_events": primary_comparison["phase_events_PR"],
            "event_identity": primary_comparison["phase_event_step_cell_transition_identity"],
        },
        "second_flow_phase_validation": {
            "space_time_states": comparison["space_time_states"],
            "matched_states": 558 - comparison["phase_state_mismatches"],
            "match_fraction": comparison["phase_state_match_fraction"],
            "mismatches": comparison["phase_state_mismatches"],
            "MAGE_phase_events": comparison["phase_events_MAGE"],
            "PR_phase_events": comparison["phase_events_PR"],
            "event_identity": comparison["phase_event_step_cell_transition_identity"],
        },
        "caseB_continuous_state_errors": {
            key: comparison[key]
            for key in (
                "pressure_relative_error_mean",
                "pressure_relative_error_max",
                "composition_absolute_error_mean",
                "composition_absolute_error_max",
                "gas_saturation_absolute_error_mean",
                "gas_saturation_absolute_error_max",
            )
        },
        "support_domain": {
            "case_A_inside_box": domain["case_A"]["all_online_dual_states_inside_selected_componentwise_support_box"],
            "case_A_inside_hull_fraction": domain["case_A"]["selected_support_convex_hull_test_on_eta_1_to_eta_3"]["inside_fraction"],
            "case_B_inside_box": domain["case_B"]["all_online_dual_states_inside_selected_componentwise_support_box"],
            "case_B_inside_hull_fraction": domain["case_B"]["selected_support_convex_hull_test_on_eta_1_to_eta_3"]["inside_fraction"],
        },
        "strengthened_figure4_all_checks_pass": f4_quality["all_required_checks_pass"],
        "gas_brine_maximum_derivative_relative_error": brine["derivative_check"]["maximum_relative_error"],
        "gas_brine_maximum_coexistence_residual": max_coexist,
        "tangent_plane_support_audit": {
            "status": tangent["status"],
            "selected_ternary_supports_accepted": tangent["all_selected_ternary_supports_accepted"],
            "liquid_supports_rechecked": tangent["figure5_ternary_selected_support_recheck"]["liquid"]["selected_support_count"],
            "vapor_supports_rechecked": tangent["figure5_ternary_selected_support_recheck"]["vapor"]["selected_support_count"],
        },
    }
    (REPORTS / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "delivery_root": ".",
        "entry_points": entry_points,
        "key_results": key_results,
        "method_reports": {
            "support_filtering": "reports/support_state_filtering.json",
            "potential_scaling": "reports/potential_scaling_parameters.csv",
            "support_domains": "reports/support_domain_bounds.csv",
            "flow_domain_audit": "reports/flow_domain_audit.json",
            "offline_conventions": "reports/offline_pool_conventions.md",
            "SI_parameter_table": "reports/si_parameter_table.md",
            "validation": "reports/validation_report.json",
        },
        "hard_acceptance_rule": "The primary flow requires all 552 phase labels and the second flow all 558 phase labels to match the independent fully implicit PR calculation; every phase-event step/cell/transition must also match.",
    }
    (BASE / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
