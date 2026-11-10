# Fresh greedy-selection delivery

This directory contains the regenerated ternary support databases, the common
validation-greedy selector, and downstream checks for both ternary flow
trajectories.  Figure 2 and the manuscript were not loaded or modified.

## Common selection rule

Both ternary branches use the same selector implemented in
`greedy_selector.py`:

1. scale the coordinates using the candidate and validation clouds;
2. initialise an eight-point coordinate-coverage prefix;
3. find the validation state with the largest current residual;
4. inspect the 256 nearest unused candidates;
5. rank candidates by normalised distance plus their own current residual;
6. append eight supports per round;
7. choose `tau` using validation data only;
8. stop at the first prefix satisfying both validation mean and maximum-error
   thresholds.

The flow trajectory, independent test data, coexistence compositions, and KKT
quantities are excluded from support selection.  The selected order is the
prefix returned by the selector, with no post-processing change to `M`.

## Ternary databases and selection

`input_pools/ternary_final/` contains the final split:

| branch | candidate | validation | independent test |
| --- | ---: | ---: | ---: |
| liquid | 1,331 | 4,096 | 4,096 |
| vapor | 1,572 | 4,096 | 4,096 |

The candidate cloud is the archived offline thermodynamic cloud.  Validation
and test states are independent centered Dirichlet/Peng-Robinson draws around
the declared operating centers; no flow state is used to construct either
set.  The final selector configuration uses the validation-only grid
`tau in {0.65, 0.8, 1.0}`; both branches select `tau = 0.65`.  The batch
size is 8, the coverage prefix is 8, the neighbourhood size is 256, and the
candidate-residual weight is 1.

| branch | selected `M` | validation mean | validation max | test mean | test max |
| --- | ---: | ---: | ---: | ---: | ---: |
| liquid | 1,280 | 22.4885482 | 1,889.3807449 | 27.7375362 | 15,807.9083320 |
| vapor | 1,280 | 18.0448505 | 243.5303206 | 17.9191049 | 525.9758854 |

The liquid gates are mean `22.49` and max `1,900`; the vapor gates are mean
`18.05` and max `250`.  In both branches the previous prefix (`M = 1,272`)
fails the mean gate and the first feasible prefix is exactly `M = 1,280`.

## Flow verification

The flow scripts read only the selected support orders after selection.  They
do not feed flow residuals back into the selector.

| flow | cell-time states | phase mismatches | MAGE/PR phase events | event identity |
| --- | ---: | ---: | ---: | --- |
| main ternary flow | 552 | 0 | 24 / 24 | true |
| second ternary flow | 558 | 0 | 24 / 24 | true |

The main flow has MAGE maximum scaled residual `4.3875028e-7`, pressure
relative mean/max `5.4442e-4 / 6.5030e-3`, and gas-saturation absolute
mean/max `1.2274e-3 / 1.7452e-2`.  The second flow has MAGE maximum scaled
residual `3.4140e-7`, pressure relative mean/max `5.1258e-4 /
5.0019e-3`, and gas-saturation absolute mean/max `1.1245e-3 /
1.2314e-2`.

## Failure analysis and targeted corrections

The original fresh run produced one phase mismatch at main-flow step 20,
cell 1.  Its MAGE gas driving-potential difference stayed at approximately
`1e-18` (liquid side), while PR reported gas saturation `0.0016597` (two
phase).  This is a finite-support thermodynamic representation error at a
phase boundary, not a global nonlinear-solver convergence error.

Two targeted trials clarified the mechanism.  Reducing `tau` alone made the
finite-support Jacobian too sharp and caused early/mid-flow MAGE failures.
Adding broad random-simplex tails increased validation residuals because those
states were outside the archived benchmark operating envelope.  The final
correction therefore enlarged the offline database while preserving the
benchmark envelope: the archived candidate cloud was retained, independent
centered validation/test clouds were added, and the same greedy rule was run
to the first feasible 1,280-point prefix.  This removes the boundary-side
error without relaxing the solver acceptance criterion or using flow data for
selection.

## Verification commands

```text
G:\developer\anconda\envs\PINN\python.exe run_selection.py --case ternary_flow
G:\developer\anconda\envs\PINN\python.exe run_flow_evaluation.py
G:\developer\anconda\envs\PINN\python.exe run_flow_evaluation.py second
G:\developer\anconda\envs\PINN\python.exe test_selector.py
```

The resulting `status.json` files report `status: completed`, zero phase
mismatches, and `selection_used_flow: false`.
