# Computational code

The public interface is in `cases/`. These scripts are grouped by scientific
case rather than by figure. They reuse the final EOS, support-selection and
fully implicit solver programs included in this repository; the plotting
backend is implemented in Python.

## Reproduction

```text
python code/cases/binary_equilibrium.py
python code/cases/nonisothermal_vle.py
python code/cases/generality_benchmarks.py
python code/cases/flow_benchmark.py
python code/cases/gas_brine_validation.py
python code/cases/second_flow_condition.py
python code/supplementary_note6_sixcomponent/benchmark.py
python code/supplementary_note7_embedding/pr_embedding.py --outdir code/supplementary_note7_embedding/results
```

Use `--quick` on the first, second, third and fifth commands for a short
deterministic check. The default output root is `case_results/`, which is
ignored by Git so repeated runs do not dirty the source tree.

`figure_generation/plot_figures_2_to_6.py` contains the shared Matplotlib
renderer and the Supplementary Figure 1 renderer. It consumes ordinary
CSV/JSON tables and writes journal-ready PDF, SVG and 600-dpi PNG files.

The gas-brine case needs PyTorch in addition to NumPy, SciPy, pandas and
Matplotlib. The six-component benchmark additionally requires a C++17 compiler
when its included Linux shared libraries are rebuilt. The embedding control
uses the NumPy/SciPy stack only. All numerical reference files are stored in
the matching `source_data/` directories, and no proprietary runtime is
required.

The six-component implementation and its source data are in
`supplementary_note6_sixcomponent/`; its `data/`, `model/` and `results/`
subdirectories contain the state splits, finite-support model and verified
benchmark products. The isometric control is in
`supplementary_note7_embedding/`; `results/embedding_results.csv` and
`results/embedding_trace.json` are the machine-readable products for
Supplementary Table 15.
