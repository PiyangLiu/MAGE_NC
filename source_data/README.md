# Source Data

This directory provides the numerical data underlying Figures 2--6,
Supplementary Figure 1 and the numerical tables in Supplementary Notes 3--7.
Figure 1 is a conceptual schematic and has no numerical source data.

- `Figure2`: coexistence states and empirical error distributions.
- `Figure3`: support-count metrics and the three displayed coexistence curves.
- `Figure4`: support coverage, nonisothermal coexistence curves, caloric errors
  and temperature-resolved coexistence errors.
- `Figure5`: GERG phase states and convergence metrics, together with the LLE
  free-energy curve, coexistence states and support-count errors.
- `Figure6_and_SupplementaryFigure1`: paired MAGE/PR flow histories, phase
  events and the calculation summary used for both figures.
- `Supplementary_Note_3`: nonisothermal temperature-resolved tables and
  independent validation summaries.
- `Supplementary_Note_5_1_gas_brine`: support-selection, training and
  independent KKT results for the gas--brine calculation. The file
  `gas_brine_full_recalculation_manifest.json` records the verified full
  rerun in the Anaconda `PINN` environment.
- `Supplementary_Note_5_2_second_flow`: complete MAGE/PR histories, events,
  solver summaries and support-domain audit for the second flow condition.
- `Fresh_Greedy_Selection`: the current common validation-error greedy
  selection audits, support orders, validation curves and downstream checks
  used to update the manuscript calculations. The five selection reports are
  `nonisothermal`, `gerg`, `liquid_liquid`, `gas_brine` and `ternary_flow`;
  the two flow subdirectories contain the primary and second ternary-flow
  histories.
- `Supplementary_Note_6_sixcomponent`: six-component PR state splits, the
  finite-support model, the complete confirmatory state table, compiled audit
  and verified timing/accuracy summary for Supplementary Tables 12--14.
- `Supplementary_Note_7_embedding`: the embedding-result CSV and complete
  support-selection/derivative-audit trace for Supplementary Table 15.

CSV files use a header row and full-precision decimal values. JSON files retain
the calculation configuration, metric definitions and nested summaries.
