# Supplementary Note 7 source data

This directory contains the numerical source data for the isometric-embedding
control and Supplementary Table 15.

`results/embedding_results.csv` reports support counts, validation stopping
values and independent test errors for ambient dimensions 2, 3, 5, 8 and 10.
`results/embedding_trace.json` contains the selected support indices, complete
validation traces and analytic-gradient audits. The five ambient dimensions
use the same intrinsic two-dimensional PR state manifold; the identical
13-support liquid and 27-support vapour prefixes are therefore expected.

The control uses maximum validation error `0.05` and dimensionless smoothing
parameter `tau=0.0006`. Values in the CSV retain full decimal precision.
