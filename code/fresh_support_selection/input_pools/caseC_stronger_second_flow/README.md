# Case C: stronger second coupled-flow condition

This directory is self-contained. It reuses the frozen ternary MAGE support
representation used by the primary flow benchmark: 1,280 supports per phase
and `tau=0.65`. No support is rebuilt or retuned.

The online trajectory differs through the injection schedule:

- two-phase injection: `(0.15, 0.15, 0.70)` for 8,000 s;
- liquid wash: `(0.04, 0.08, 0.88)` for 22,000 s;
- vapour-rich injection: `(0.65, 0.20, 0.15)` for 30,000 s.

The initial composition remains `(0.04, 0.08, 0.88)` and the pressure
boundaries remain 3.2/2.0 MPa. The time grid contains 93 accepted steps and
uses exact stage boundaries at 8,000 and 30,000 s. The solver copy retains the
published acceptance thresholds (`2e-6` for MAGE and `2e-5` for PR) and only
raises the nonlinear iteration caps for this more strongly perturbed history.

Run from the parent `fresh_support_selection` directory:

```text
python run_flow_evaluation.py second
```

The driver runs MAGE and an independent fully implicit PR calculation, checks
all 558 cell-time states, requires 100% phase-label agreement and identical
phase-event step/cell/transition records, and writes the CSV/JSON files below
`results/`.

To package the same tables as the Zenodo raw-simulation file, run:

```text
python export_hdf5.py
```

The exporter writes `results/caseC_second_ternary_flow.h5` using the CSV/JSON
files produced by the driver.

The delivered rerun reports 558 cell-time states, zero phase-state
mismatches, 24 identical phase events, mean relative pressure error
`5.1258e-4`, maximum relative pressure error `5.0019e-3`, mean absolute
composition error `6.6110e-4`, maximum absolute composition error `0.05748`,
mean absolute gas-saturation error `1.1245e-3`, and maximum absolute
gas-saturation error `1.2314e-2`.
