# Figure 5 fully implicit ternary calculation

This folder contains the authoritative 92-step calculation for CH4/CO2/n-butane. The archived branch pool is filtered through the PR flash, and the fixed finite-support representation contains 1,280 supports per phase with `tau=0.5` and one shared additive potential gauge.

Run from this folder:

```text
python src/run_benchmark.py
```

`src/build_mage_model.py` reconstructs the 1,280-support model from the frozen teacher pool. It is not necessary for an online rerun. No flow-trajectory state is used during support selection. The MAGE global solve evaluates both branch potentials and the complementarity residual directly; its online residual contains no local flash or stability call. The PR comparison invokes a local PR flash inside the fully implicit global residual.
