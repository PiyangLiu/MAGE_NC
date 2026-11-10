# Strengthened nonisothermal benchmark

This directory contains the final CH4/n-C4 PR/MAGE nonisothermal benchmark and the regenerated Figure 4 products.

## Problem definition

- Temperature domain: 295-350 K.
- Pressure domain: 0.5-7 MPa.
- Held-out temperatures: 304.5-305.5 K, 324.5-325.5 K, 344.5-345.5 K.
- Candidate coexistence layers: 0.25 K spacing outside the held-out temperatures.
- Validation coexistence layers: 0.25 K offset spacing outside the held-out temperatures.
- Candidate pool per phase: 12,000 states (8,273 coexistence, 3,727 homogeneous).
- Validation pool per phase: 5,000 states (3,430 coexistence, 1,570 homogeneous).
- Exact-temperature test: 192 homogeneous states per phase and temperature.

## Support selection

Supports are selected by a stratified residual-greedy rule using the independent validation pool, followed by farthest-point coverage. Complete pressure coverage is protected on the retained 0.25 K layers immediately adjacent to each held-out band.
No held-out thermodynamic values are used for support-count or smoothing selection; only the declared gap geometry is used for adjacent-layer coverage protection.
The smoothing scale is the largest validation trial whose non-held-out VLE MAE is within 5% of the minimum and whose homogeneous Phi mean relative error is below 1% in both branches.
- Selected support counts: 1,592 liquid and 1,392 vapour.
- Selected smoothing parameters: liquid 0.125, vapour 0.125.
- The current nested support orders, validation curves and thresholds are in
  `code/fresh_support_selection/results/nonisothermal_selection.json`.

## Held-out results

Band-averaged state and VLE metrics are available in the CSV products.

| Band (K) | Phase | mean relative Phi error | mean absolute u error (kJ mol-1) |
|---|---:|---:|---:|
| 304.5-305.5 | liquid | 0.00228 | 0.116 |
| 304.5-305.5 | vapor | 0.00351 | 0.154 |
| 324.5-325.5 | liquid | 0.00214 | 0.125 |
| 324.5-325.5 | vapor | 0.00395 | 0.198 |
| 344.5-345.5 | liquid | 0.00218 | 0.152 |
| 344.5-345.5 | vapor | 0.00408 | 0.192 |

| Band (K) | liquid composition MAE | vapour composition MAE | converged/reference |
|---|---:|---:|---:|
| 304.5-305.5 | 0.00347 | 0.00945 | 123/123 |
| 324.5-325.5 | 0.00337 | 0.00897 | 120/120 |
| 344.5-345.5 | 0.00321 | 0.0118 | 114/114 |

Run `python code/cases/nonisothermal_vle.py` from the repository root to
recompute the benchmark and render Figure 4. Append `--quick` for a short
smoke test. The lower-level files in the parent directory are retained as
calculation provenance; the case entry point is the authoritative public
workflow.
