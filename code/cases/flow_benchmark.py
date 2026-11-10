"""Run the fully implicit ternary flow benchmark and render Figure 6."""
from __future__ import annotations

import argparse
import shutil
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
SOURCE = CODE / "figure5_ternary_fully_implicit"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(CODE / "figure_generation"))
import run_benchmark as benchmark  # noqa: E402
from plot_figures_2_to_6 import (build_flow_plot_data, plot_figure6,
                                 plot_supplementary_flow_errors)  # noqa: E402


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("benchmark_config.json", "time_grid.csv"):
        shutil.copyfile(SOURCE / name, output / name)
    shutil.copytree(SOURCE / "model", output / "model", dirs_exist_ok=True)
    benchmark.ROOT = output
    benchmark.main()
    result_dir = output / "results"
    plot_data = output / "plot_data"
    build_flow_plot_data(result_dir, plot_data)
    plot_figure6(plot_data, output / "figures")
    plot_supplementary_flow_errors(result_dir, output / "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "case_results" / "flow_benchmark")
    args = parser.parse_args(); run(args.output_dir)
    print(f"Ternary flow benchmark complete: {args.output_dir}")


if __name__ == "__main__":
    main()
