# Figure 2 binary equilibrium fidelity

`run_figure2.py` regenerates the CH4/n-butane PR candidate, validation, and independent test pools; applies stable-flash filtering; constructs the nested finite-support branches; and solves the PR and MAGE coexistence curves. The full retained sizes are 8,192 candidate, 2,048 validation, and 4,096 test states per branch, with 45 equilibrium pressures at 300 K.

Use `--quick` for a deterministic smoke test. Omit it for the retained calculation. The associated complete result data are in `../../data/figure2_summary.json`.
