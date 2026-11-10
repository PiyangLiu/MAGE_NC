# Parameters for independent reconstruction

| Case | Reference model | T (K) | P domain | Supports per branch | Smoothing |
|---|---|---:|---|---:|---:|
| Figures 2-3 | PR, CH4/nC4 | 300 | 0.45-5.10 MPa candidate range | selected values in Figure 3 | selected from validation grid |
| Strengthened Figure 4 | nonisothermal PR, CH4/nC4 | 295-350 | 0.50-7.00 MPa | 1,592 liquid / 1,392 vapour | 0.125 |
| Figure 5 / ternary flows | PR, CH4/CO2/nC4 | 320 | 1.25-3.25 MPa declared flow domain | 1,280 | 0.65 |
| Figure 6 GERG | GERG-2008 through Thermopack | 320 | 1.5-10.0 MPa | selected M values in JSON | selected in JSON |
| Figure 6 LLE | analytic regular-Helmholtz model | fixed model temperature | 4.5-5.5 MPa sampled branch domain | selected M values in JSON | selected in JSON |

The machine-readable values, including dual-coordinate centers, `D_eta`, `Phi_s`, physical and scaled smoothing parameters, common branch offsets, and support bounds, are in `potential_scaling_parameters.csv` and `support_domain_bounds.csv`. The two phases always use a common thermodynamic reference. Figure 5 uses one validation-calibrated additive correction applied identically to both branches; all other packaged examples use zero added branch offset.
