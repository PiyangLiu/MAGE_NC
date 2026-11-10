# Nonisothermal PR/MAGE benchmark

This directory contains the reproducible nonisothermal methane/n-butane
Peng--Robinson benchmark used for the manuscript Figure 4 and Supplementary
Note 3.

## Problem and data split

The temperature and pressure domains are 295--350 K and 0.5--7 MPa. Three
closed temperature bands (304.5--305.5 K, 324.5--325.5 K and
344.5--345.5 K) are held out from support selection. Candidate and validation
pools contain 12,000 and 5,000 states per phase, respectively. The candidate
pool has 8,273 paired coexistence states and 3,727 homogeneous states; the
validation pool has 3,430 paired coexistence states and 1,570 homogeneous
states. The independent exact-temperature audit contains 2,688 states per
phase.

## Support selection

The common validation-error greedy selector is applied independently to the
liquid and vapour branches. It starts with the scaled-coordinate coverage
prefix, then adds eight candidates per step from the 256 nearest unused
candidates ranked by normalized distance plus the candidate's own residual.
Only candidate and validation data are used for selection; the independent
test, coexistence-composition evaluation and flow calculations are excluded.
The first nested prefix satisfying the declared validation gates is retained.

The selected prefixes contain 1,592 liquid supports and 1,392 vapour supports,
with dimensionless smoothing parameter `tau_bar=0.125` in both branches.
The complete support orders and validation histories are in
`code/fresh_support_selection/results/nonisothermal_selection.json`.

## Evaluation

The retained model is evaluated on the three held-out temperature bands. The
liquid composition MAE is 3.47e-3, 3.37e-3 and 3.21e-3, and the vapour
composition MAE is 9.45e-3, 8.97e-3 and 1.18e-2 for the three bands in the
order listed above. The corresponding MAGE continuation counts are 123/123,
120/120 and 114/114. State-error and energy-error tables are supplied as CSV
files in this directory and in `source_data/Supplementary_Note_3`.

## Reproduction

From the repository root, run `python code/cases/nonisothermal_vle.py` to
regenerate the benchmark products. Append `--quick` for a short smoke test.
