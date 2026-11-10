# Nonisothermal PR/MAGE benchmark

This file documents the nonisothermal methane/n-butane Peng--Robinson
calculation used for Figure 4 and Supplementary Note 3.

The temperature and pressure domains are 295--350 K and 0.5--7 MPa. The
reserved evaluation bands are 304.5--305.5 K, 324.5--325.5 K and
344.5--345.5 K. Candidate and validation pools contain 12,000 and 5,000
states per phase, including coexistence and stable-homogeneous states.

The common validation-error greedy selector starts with the scaled-coordinate
coverage prefix and adds eight points per step from the 256 nearest unused
candidates, ranked by normalized distance plus candidate residual. The first
nested prefix satisfying the validation gates is retained. Independent test,
coexistence-composition and flow results are excluded from selection. The
selected prefixes contain 1,592 liquid and 1,392 vapour supports, with
dimensionless `tau_bar=0.125` for both branches.

The publication redraw uses a 0.5 K temperature grid from 295 to 350 K and
41 pressure samples per temperature. The three reserved bands have liquid
composition MAEs of 3.47e-3, 3.37e-3 and 3.21e-3 and vapour composition MAEs
of 9.45e-3, 8.97e-3 and 1.18e-2, respectively. The complete support order,
validation history and state tables are supplied with the repository.
