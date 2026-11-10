# Supplementary Note 6: six-component PR benchmark

This directory provides the complete reproducible implementation for the
six-component noncritical Peng-Robinson benchmark. It compares a compiled
finite-support MAGE representation with a compiled PR inverse solver for
prescribed chemical potentials at 300 K.

## Problem definition

The mixture is CH4/C2H6/C3H8/nC4/nC5/nC6 with classical PR mixing and zero
binary interaction parameters. Each query supplies the six dimensionless
chemical coordinates

```text
eta_i = log(x_i p / p_ref) + log(phi_i),  p_ref = 1e5 Pa.
```

Both solvers return the liquid and vapour branch pressures, all six component
number densities on each branch, both compressibility factors, and the stable
branch. The stable branch is the branch with the higher pressure at fixed
chemical potentials.

## Sampling and finite-support model

The state manifold is generated from PR coexistence anchors between 1.95 and
2.75 MPa using the anchor composition `(0.46, 0.12, 0.08, 0.12, 0.11,
0.11)`, followed by positive or negative nC6 chemical-coordinate offsets.
Candidate, validation, development-held-out and confirmatory streams contain
1,200, 360, 800 and 1,000 states, respectively, with seeds 880101--880104.
Support
selection uses the candidate and validation streams only. The stored model has
128 exact PR tangent supports per phase and dimensionless smoothing parameter
`tau = 4e-5`; no confirmatory state is used during selection.

## Confirmatory accuracy

The independent confirmatory set contains 1,000 states, with 499 liquid-stable
and 501 vapour-stable PR references. MAGE matches all 1,000 stable-phase labels.
The pressure relative error has mean `1.81e-4` and maximum `5.14e-4`. The
component number-density relative error has mean `2.59e-3` and maximum
`5.41e-2`. The maximum occurs for dilute vapour nC6 (mole fraction about
0.0053). The maximum absolute compressibility-factor error is `5.02e-3`.

## Compiled timing

All methods use the same batched query contract and C++17 `-O3` compilation.
The five-repeat median times are 1.901 microseconds/query for direct MAGE,
7.246 microseconds/query for the fastest PR analytic-Jacobian variant,
13.513 microseconds/query for cold finite-difference PR, and 14.065
microseconds/query for warm finite-difference PR. The measured direct-MAGE
speed ratio against the fastest PR variant is 3.81x for this prescribed
chemical-potential benchmark.

Reference generation and validation-greedy support construction take 24.50 s
and 0.272 s, respectively, for the recorded hardware run. These values are
included for amortisation accounting and are not universal hardware timings.

## Reproduction

From this directory, install `requirements.txt`, then compile the three shared
libraries and run:

```bash
g++ -std=c++17 -O3 -fPIC -shared pr_six.cpp -o libpr_six.so
g++ -std=c++17 -O3 -DANALYTIC_JAC -fPIC -shared pr_six.cpp -o libpr_six_analytic.so
g++ -std=c++17 -O3 -fPIC -shared mage_six.cpp -o libmage_six.so
python generate.py
python fit.py --supports 128
python audit.py
python benchmark.py
python plot.py
```

The included shared libraries support Linux x86-64 readers who do not need to
recompile. `benchmark.py` performs the phase-label, PR-parity and continuous
error checks before writing `results/verified_benchmark.json`.

## Files

`data/` contains the independent state splits, `model/` contains the selected
finite-support representation, and `results/` contains the verified metrics,
state table, implementation audit and figure. `pr_six.py` and `pr_six.cpp`
implement the Python and compiled PR references; `mage_six.cpp` implements the
compiled finite-support evaluation.

The benchmark is intentionally limited to the stated two-parameter manifold
and prescribed fugacity specification. It is not a claim about every
six-dimensional composition, a near-critical flash, or an ordinary `(T,p,z)`
industrial flash.
