# Figure 6 GERG-2008 reference transfer

`run_figure6_gerg.py` builds Thermopack GERG-2008 liquid and vapour pools for CO2/CH4/n-butane at 320 K and 1.5-10.0 MPa. Every candidate first passes `two_phase_tpflash`; only stable matching single-phase states or returned coexistence compositions are retained. The bundled Thermopack 2.2.3 library is resolved from `../vendor`.

The full calculation uses 80,000 candidate, 20,000 validation, 20,000 test, and 5,000 independent overall states. Use `--quick` to verify the runtime interface. The complete retained plot data are in `../../data/gerg_lle_plot_data.json` and the machine-readable source tables are under `../../source_data/Figure5`.
