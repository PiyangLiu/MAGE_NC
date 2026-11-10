"""Run the fixed-temperature CH4/n-C4 binary-equilibrium case.

One invocation performs the candidate/validation/test sampling, validation-
controlled support selection, MAGE coexistence solve, result-table export and
the Figure 2--3 rendering.  ``--quick`` uses the same algorithms on smaller
deterministic pools for a smoke test; omit it for the manuscript calculation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
CODE = HERE.parents[1]
REPO = CODE.parent
sys.path.insert(0, str(CODE / "common"))
sys.path.insert(0, str(CODE / "figure_generation"))
import run_mage_examples as examples  # noqa: E402
from plot_figures_2_to_6 import plot_figure2, plot_figure3  # noqa: E402


def _write_plot_tables(result: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    # Figure 2(b,c): independent test distributions for the selected branch.
    distributions = {}
    for phase, suffix in (("liquid", "liquid"), ("vapor", "vapor")):
        model = result["models"][max(result["models"])]
        model = model[0 if phase == "liquid" else 1]
        test = result["test_pools"][phase]
        metrics = examples.metric_summary(model, test)
        phi = np.abs(metrics["pred_phi"] - test["phi"]) / np.maximum(np.abs(test["phi"]), 1e-12)
        grad = np.linalg.norm(metrics["pred_q"] - test["q"], axis=1) / np.maximum(np.linalg.norm(test["q"], axis=1), 1e-12)
        distributions[f"phi_{suffix}"] = phi
        distributions[f"gradient_{suffix}"] = grad
    n = max(len(v) for v in distributions.values())
    frame = pd.DataFrame({k: np.pad(v, (0, n - len(v)), constant_values=np.nan) for k, v in distributions.items()})
    frame.to_csv(output / "figure2_ecdf.csv", index=False)

    summary = json.loads((output / "binary_summary.json").read_text(encoding="utf-8"))
    for count in (16, 64, 1024):
        source = output / f"binary_vle_M{count}.csv"
        if not source.exists():
            # Quick mode stops at M=128.  Preserve a complete plotting schema
            # by using the largest available support as the smoke-test alias.
            available = sorted(output.glob("binary_vle_M*.csv"))
            source = available[-1]
        table = pd.read_csv(source).rename(columns={"P": "pressure_MPa"})
        table["pressure_MPa"] /= 1e6
        table["pL_mage"] /= 1e3
        table["pV_mage"] /= 1e3
        table.to_csv(output / f"figure3_vle_M{count}.csv", index=False)
    target = max(result["models"])
    source = pd.read_csv(output / f"binary_vle_M{target}.csv").rename(columns={"P": "pressure_MPa"})
    source["pressure_MPa"] /= 1e6
    source["pL_mage"] /= 1e3
    source["pV_mage"] /= 1e3
    source.to_csv(output / "figure2_vle.csv", index=False)
    # Keep an auditable pointer to the support count used in a quick run.
    (output / "plot_manifest.json").write_text(json.dumps({"selected_support_for_plot": int(target), "full_manuscript_support": 1024}, indent=2) + "\n", encoding="utf-8")


def run(quick: bool, output: Path):
    examples.RESULTS = output
    examples.FIGURES = output / "intermediate_figures"
    examples.RESULTS.mkdir(parents=True, exist_ok=True)
    result = examples.run_binary(quick=quick)
    _write_plot_tables(result, output)
    figure_data = CODE / "figure_generation" / "data"
    figures = output / "figures"
    plot_figure2(figure_data, figures, output)
    plot_figure3(figure_data, figures, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small deterministic smoke test")
    parser.add_argument("--output-dir", type=Path, default=REPO / "case_results" / "binary_equilibrium")
    args = parser.parse_args()
    run(args.quick, args.output_dir)
    print(f"Binary equilibrium case complete: {args.output_dir}")


if __name__ == "__main__":
    main()
