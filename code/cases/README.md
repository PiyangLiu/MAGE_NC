# Case-level reproduction

Each entry point is organized by scientific calculation. It runs the numerical
model, writes CSV/JSON/NPZ result products, builds plot-ready data from those
products, and renders every figure associated with the case.

| Scientific case | Command | Figures and products |
| --- | --- | --- |
| Binary equilibrium (CH4/n-C4, 300 K) | `python code/cases/binary_equilibrium.py` | Figures 2 and 3 |
| Nonisothermal vapour-liquid equilibrium | `python code/cases/nonisothermal_vle.py` | Figure 4 and SI Note 3 tables |
| GERG-2008 and liquid-liquid benchmarks | `python code/cases/generality_benchmarks.py` | Figure 5 |
| Pressure-coupled ternary flow | `python code/cases/flow_benchmark.py` | Figure 6 and Supplementary Figure 1 |
| Non-PR gas-brine validation (SI Note 5.1) | `python code/cases/gas_brine_validation.py` | KKT, training and summary tables |
| Second pressure-coupled flow condition (SI Note 5.2) | `python code/cases/second_flow_condition.py` | Case-B tables and `SupplementaryFigure1_CaseB` |
| Six-component noncritical PR benchmark (SI Note 6) | `python code/supplementary_note6_sixcomponent/benchmark.py` | Accuracy, compiled timing, audit and benchmark figure |
| Isometric embedding control (SI Note 7) | `python code/supplementary_note7_embedding/pr_embedding.py --outdir code/supplementary_note7_embedding/results` | Supplementary Table 15 and embedding trace |

The first four commands are the manuscript cases. The next two commands cover
Supplementary Notes 5.1 and 5.2, and the final two cover Supplementary Notes 6
and 7. Add `--quick` to binary, nonisothermal,
generality or gas-brine commands for a deterministic smoke run. Full mode
uses the manuscript pool sizes and accepted support counts. `--output-dir`
keeps independent reruns separate.

Every figure-producing case directory contains a `figures/` folder with PDF,
SVG and 600-dpi PNG output. The numerical tables used by a plot are kept next
to the figure data, so a reader can inspect or post-process them without
proprietary tools. The Note 6 and Note 7 directories additionally include the
state splits, models and validation traces required for their tabulated
results.
