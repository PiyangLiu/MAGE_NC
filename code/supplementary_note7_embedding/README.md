# Supplementary Note 7: isometric embedding control

This directory contains the reproducible isometric-embedding control for the
Peng-Robinson finite-support representation. The physical thermodynamic system
is unchanged: binary CH4/nC4 liquid and vapour states are embedded into
ambient dimensions 2, 3, 5, 8 and 10 by orthonormal maps.

## Method

The chemical-potential coordinates, Massieu potential and gradients are
standardised before applying an orthonormal embedding. The embedding maps
support points and gradients together, so inner products, tangent planes and
the normalized log-sum-exp representation are invariant. The deterministic
validation selector adds one support at a time and stops at the first prefix
whose maximum validation error is at most `0.05`, with `tau = 0.0006`.

The analytic PR gradient is independently checked by finite differences. The
script asserts that support indices and counts are identical in every ambient
dimension.

## Recorded results

The liquid branch uses 13 supports and the vapour branch uses 27 supports for
each ambient dimension. The liquid test potential MAE is
`3.167268548559e+0 mol m^-3`, with maximum `8.552191065265e+0`; the mean and
maximum relative gradient errors are `1.83705300853e-2` and
`4.90037900002e-2`. The vapour values are potential MAE
`2.664805724550e+0`, maximum `4.532231814847e+0`, and mean/maximum relative
gradient errors `2.24589566748e-2` and `4.75972100233e-2`. These values are
identical across the five ambient dimensions up to floating-point roundoff.

## Reproduction

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python pr_embedding.py --outdir results
python verify_results.py --outdir results
```

`embedding_results.csv` is the table used for Supplementary Table 15 and
`embedding_trace.json` contains every support index sequence, validation trace
and derivative audit. `mage_core.py` provides the normalized finite-support
evaluation and deterministic selector. The companion `brine_closure.py`
records a separate dimensionless closure benchmark bundled with the same
software distribution; it is not used for the embedding table.

This control isolates ambient-coordinate representation from intrinsic
thermodynamic dimension. It does not imply that independently varying real
five- or ten-component mixtures have the same support complexity.
