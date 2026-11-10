"""Run Supplementary Note 5.2: the second pressure-coupled flow condition."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_


HERE = Path(__file__).resolve()
CODE = HERE.parents[1]
REPO = CODE.parent
SOURCE = CODE / "unified_selection"

sys.path.insert(0, str(CODE / "figure_generation"))
from plot_figures_2_to_6 import (build_flow_plot_data,
                                 plot_supplementary_flow_errors)  # noqa: E402


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["UNIFIED_CASEB_OUTPUT_DIR"] = str(output / "results")
    sys.path.insert(0, str(SOURCE))
    import run_ternary_caseB_unified as calculation  # noqa: E402

    calculation.main()
    result_dir = output / "results"
    model_summary = SOURCE / "ternary_model" / "mage_model_summary.json"
    support_selection = json.loads(model_summary.read_text(encoding="utf-8"))
    plot_data = output / "plot_data"
    build_flow_plot_data(result_dir, plot_data, {
        "algorithm": support_selection.get("support_selection", "frozen finite-support model"),
        "supports_per_phase": int(support_selection.get("supports_per_phase", 1280)),
        "tau": float(support_selection.get("tau", 0.5)),
    })
    plot_supplementary_flow_errors(result_dir, output / "figures",
                                   name="SupplementaryFigure1_CaseB")
    (output / "case_manifest.json").write_text(
        json.dumps({"case": "Supplementary Note 5.2 second flow condition",
                    "result_directory": str(result_dir),
                    "plot_data": str(plot_data)}, indent=2) + "\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "case_results" / "second_flow_condition")
    args = parser.parse_args()
    run(args.output_dir)
    print(f"Second flow condition complete: {args.output_dir}")


if __name__ == "__main__":
    main()
