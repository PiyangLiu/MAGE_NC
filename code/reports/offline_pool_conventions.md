# Offline pool and acceptance conventions

## PR binary and nonisothermal pools

Each candidate draw is sent through the PR flash before a branch state is built. A stable single-phase state is retained only if the returned phase equals the requested branch. For a two-phase flash, the returned liquid or vapour composition is used for the corresponding branch. Cubic roots are evaluated only after this classification: the smallest valid root is the liquid root and the largest valid root is the vapour root. A root selected without a stable flash result is rejected, so metastable homogeneous roots are not included in the support pools. Candidate pools contain deterministic coexistence-grid states plus accepted stable states; validation pools are independent accepted states and test pools are held out from support ordering.

## Support ordering

Figures 2 and 3 use nested validation-driven ordering. The state with the largest combined potential/gradient/composition score is identified on the validation pool, and the nearest unused candidate is added. A deterministic coexistence prefix is retained. The strengthened Figure 4 calculation uses the same validation-error greedy rule on candidate pools containing stable homogeneous and coexistence states; all states in the three withheld temperature bands are excluded from candidate and validation pools. The earlier retained Figure 4 result is packaged separately and uses its equilibrium-grid support manifold. Figure 5 uses the frozen ternary candidate pool: the first 512 supports are retained and the order is extended to 1,280 by the maximum convex-support-plane gap over the declared candidate pool, without using flow-trajectory states. GERG-2008 uses deterministic farthest-point ordering for the large candidate pool after stable TP-flash filtering. The LLE example uses validation-driven nested construction on the two stable regular-solution branches.

## Validation and acceptance

Thermodynamic validation compares potential, gradient, concentration and phase-equilibrium outputs on pools not used to select the support order. Flow steps are accepted only when the scaled nonlinear residual is below the configured MAGE or PR threshold and the component mass-balance audit is finite. The primary flow requires 552/552 phase-state agreement and the second flow 558/558; both also require identical phase-event transitions against the independent fully implicit PR calculation.
