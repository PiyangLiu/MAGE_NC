# Figure 3 finite-support compression

`run_figure3.py` regenerates the same frozen binary state pools as Figure 2 and evaluates the nested support counts 16, 24, 32, 48, 64, 96, 128, 192, 384, and 1,024. At each greedy step, the largest combined validation potential/gradient error identifies the target state and the nearest unused candidate is appended; deterministic coexistence candidates form the retained prefix.

Use `--quick` for a smoke test. The complete retained convergence products are in `../../data/figure3_summary.json`.
