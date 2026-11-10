# Figure 4 nonisothermal calculation

This directory contains the final CH4/n-C4 Peng-Robinson/MAGE calculation
used by the public case entry point. The final support-selection test uses
three genuine 1 K held-out bands (304.5-305.5, 324.5-325.5 and
344.5-345.5 K), stable homogeneous and coexistence candidate states, and
1,592 liquid plus 1,392 vapour selected supports with tau_bar=0.125 in the
manuscript run. The lower-level `strengthened_results` directory contains
retained numerical pools and diagnostics; the current plot-ready products are
in `code/figure_generation/data`.

Run the complete case, including the Python Figure 4 renderer, from the
repository root:

```text
python code/cases/nonisothermal_vle.py
```

Append `--quick` for a deterministic smoke test. The lower-level
`run_figure4.py` and archived validation scripts are retained for provenance;
they are not required for the public workflow.
