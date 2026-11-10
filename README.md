# MAGE-NC reproducibility repository

This repository contains the Python implementation and machine-readable data
for the manuscript *A finite-support convex Massieu approximation for global
phase equilibrium*. The public interface is organized by scientific case:
running one command performs the corresponding numerical calculation, writes
the result data, and renders its manuscript or supplementary figures.

## Install

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The optional gas-brine validation also requires PyTorch. The complete
gas-brine case has been verified in the Anaconda `PINN` environment with
PyTorch 2.11.0 and CUDA enabled. The reference GERG implementation and all
model data needed by the cases are bundled under `code/` and `source_data/`.

## Run the cases

```bash
python code/cases/binary_equilibrium.py
python code/cases/nonisothermal_vle.py
python code/cases/generality_benchmarks.py
python code/cases/flow_benchmark.py
python code/cases/gas_brine_validation.py
python code/cases/second_flow_condition.py
python code/supplementary_note6_sixcomponent/benchmark.py
python code/supplementary_note7_embedding/pr_embedding.py --outdir code/supplementary_note7_embedding/results
```

For a short deterministic smoke test, append `--quick` to binary equilibrium,
nonisothermal VLE, generality benchmarks or gas-brine validation. Full mode
uses the manuscript pool sizes and accepted support counts. Each run writes a
separate directory below `case_results/` unless `--output-dir` is supplied.

The first four commands reproduce Figures 2-6. The nonisothermal command also
writes the SI temperature-band tables; the flow command writes Supplementary
Figure 1; the next two commands reproduce SI Notes 5.1 and 5.2. The final two
commands reproduce the six-component PR benchmark (Supplementary Note 6) and
the isometric embedding control (Supplementary Note 7). The Note 6 shared
libraries are included for Linux x86-64 and can also be rebuilt with the C++17
commands in its local README.

## Shared figure renderer

To render the supplied plot-ready tables without rerunning the thermodynamic
teachers:

```bash
python code/figure_generation/prepare_updated_figure_data.py
python code/figure_generation/plot_figures_2_to_6.py \
  --figures 2 3 4 5 6 \
  --data-dir code/figure_generation/data \
  --output-dir figures
```

The renderer uses only Python/NumPy/SciPy/pandas/Matplotlib. It writes vector
PDF/SVG and 600-dpi PNG output and has no proprietary plotting dependency.

## Current support-selection implementation

The manuscript calculations use the common validation-error greedy selector in
`code/fresh_support_selection`. This directory is self-contained: its input
pools, ternary-flow solver source, configurations and archived outputs are
stored below the same directory, so it does not depend on an absolute path on
the original author's computer. Run the complete selection suite with:

```bash
cd code/fresh_support_selection
python run_selection.py --case all
python run_flow_evaluation.py
python run_flow_evaluation.py second
```

The first command reproduces the selected support orders and validation curves
for the nonisothermal VLE, GERG-2008, liquid-liquid, gas-brine and ternary-flow
benchmarks. The two flow commands perform the downstream primary and second
ternary-flow checks. Figure 2 is a conceptual binary-equilibrium product and
is intentionally unchanged by this selector update.

The updated selected support counts are: nonisothermal liquid/vapour 1592/1392;
GERG-2008 liquid/vapour 3264/1592; liquid-liquid L1/L2 192/184; gas-brine
liquid/gas 1376/368; and ternary flow liquid/vapour 1280/1280. The ternary
offline candidate pools contain 1331 liquid and 1572 vapor states; those are
candidate-database sizes, not selected support counts. The archived JSON, CSV
and NPZ outputs under `code/fresh_support_selection/results` provide the exact
audit trail and machine-readable support orders.

## Data and citation

`source_data/` contains the fixed manuscript tables. `data/` contains shared
archival summaries, while `code/figure_generation/data/` contains the
validated plot-ready products and its integrity manifest. Pool partitions and
held-out states remain separate. `source_data/Fresh_Greedy_Selection/` contains
the updated selection audits and downstream flow outputs corresponding to the
current manuscript calculations.
Please cite the manuscript and see `CITATION.cff`. Code is released under the
MIT license in `LICENSE`; bundled third-party components retain their notices.

The Note 6 and Note 7 numerical source files are also mirrored under
`source_data/Supplementary_Note_6_sixcomponent/` and
`source_data/Supplementary_Note_7_embedding/`. The original benchmark archives
are retained in `code/` for exact file-level provenance.
