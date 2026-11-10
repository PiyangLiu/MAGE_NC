"""Core numerical objects used by the MAGE paper examples.

The implementation is deliberately self-contained.  The thermodynamic teacher
for the PR examples is the same PR Helmholtz free-energy density for all
quantities, so that ``Phi``, ``u`` and ``c`` are derivatives of one potential.
The optional GERG teacher is loaded from the bundled Thermopack DLL when the
path is available.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
# The bundled SciPy build still references aliases removed by NumPy 1.26.
# Restoring the two aliases locally keeps the calculation scripts portable.
if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_
from scipy.optimize import brentq, least_squares

R = 8.31446261815324
P_REF = 1.0e5
T_REF = 298.15

COMPONENTS = {
    "CH4": {"Tc": 190.564, "Pc": 4.5992e6, "omega": 0.01142, "MW": 16.0425e-3},
    "CO2": {"Tc": 304.1282, "Pc": 7.3773e6, "omega": 0.22394, "MW": 44.0095e-3},
    "nC4": {"Tc": 425.125, "Pc": 3.7960e6, "omega": 0.20020, "MW": 58.1222e-3},
}

# Pure-fluid ideal-gas heat-capacity correlations from the ideal Helmholtz
# terms of the cited reference equations of state. In terms of theta/T,
# cp0/R = 1 + a + sum[n_k (theta_k/T)^2 exp(-theta_k/T) /
#                     (1 - exp(-theta_k/T))^2].
# The constants below are transcribed from the public CoolProp fluid files;
# no fitted coefficients or CoolProp runtime calls enter the benchmark.
IDEAL_GAS_CALORIC = {
    "CH4": {
        "log_tau": 3.0016,
        "n": np.array([0.008449, 4.6942, 3.4865, 1.6572, 1.4115]),
        "theta_K": np.array([648.0, 1957.0, 3895.0, 5705.0, 15080.0]),
        "valid_T_K": (90.6941, 625.0),
        "source": "Setzmann and Wagner, J. Phys. Chem. Ref. Data 20, 1061-1151 (1991)",
        "doi": "10.1063/1.555898",
    },
    "nC4": {
        "log_tau": 3.24680487,
        "n": np.array([5.54913289, 11.4648996, 7.59987584, 9.66033239]),
        "theta_K": 425.125 * np.array([
            0.774840445, 3.340602552, 4.970513096, 9.975553778
        ]),
        "valid_T_K": (134.895, 575.0),
        "source": "Buecker and Wagner, J. Phys. Chem. Ref. Data 35, 929-1019 (2006)",
        "doi": "10.1063/1.1901687",
    },
}

REFERENCE_EOS_CP0 = "reference_eos_cp0"
LEGACY_ZERO_CP0 = "zero_standard_state"


def ideal_gas_caloric_metadata(names):
    """Return a JSON-safe record of the caloric model and reference state."""
    return {
        "model": REFERENCE_EOS_CP0,
        "reference_state": {
            "T_ref_K": T_REF,
            "p_ref_Pa": P_REF,
            "h_ref_J_per_mol": 0.0,
            "s_ref_J_per_mol_K": 0.0,
            "convention": (
                "Sensible ideal-gas enthalpy and entropy are zero at T_ref; "
                "formation enthalpies are not included."
            ),
        },
        "components": {
            name: {
                "correlation": "Planck-Einstein ideal-gas Helmholtz",
                "cp0_equation": (
                    "cp0/R = 1 + a + sum[n*(theta/T)^2*exp(-theta/T)/"
                    "(1-exp(-theta/T))^2]"
                ),
                "log_tau": float(IDEAL_GAS_CALORIC[name]["log_tau"]),
                "n": IDEAL_GAS_CALORIC[name]["n"].tolist(),
                "theta_K": IDEAL_GAS_CALORIC[name]["theta_K"].tolist(),
                "valid_T_K": list(IDEAL_GAS_CALORIC[name]["valid_T_K"]),
                "source": IDEAL_GAS_CALORIC[name]["source"],
                "doi": IDEAL_GAS_CALORIC[name]["doi"],
            }
            for name in names
        },
    }


def ideal_gas_standard_state(T, names):
    """Return cp0, h0, s0 and g0 for the declared sensible-energy reference.

    The Planck-Einstein integrals are analytic. Keeping complex arithmetic
    intact is essential because ``pr_massieu_from_c`` differentiates the same
    total Helmholtz potential by complex step.
    """
    temperature = np.asarray(T)
    real_temperature = np.real(temperature)
    if np.any(real_temperature <= 0):
        raise ValueError("temperature must be positive")
    cp, h, s = [], [], []
    for name in names:
        if name not in IDEAL_GAS_CALORIC:
            raise ValueError(f"no ideal-gas caloric correlation for {name}")
        spec = IDEAL_GAS_CALORIC[name]
        lower, upper = spec["valid_T_K"]
        if np.any((real_temperature < lower) | (real_temperature > upper)):
            raise ValueError(
                f"{name} ideal-gas correlation is valid over {lower:g}-{upper:g} K"
            )
        baseline = 1.0 + spec["log_tau"]
        theta = spec["theta_K"]
        coefficients = spec["n"]
        x = theta / temperature
        x_ref = theta / T_REF
        exp_neg_x = np.exp(-x)
        exp_neg_ref = np.exp(-x_ref)
        cp_i = R * (
            baseline
            + np.sum(coefficients * x**2 * exp_neg_x / (1.0 - exp_neg_x) ** 2)
        )
        h_i = R * (
            baseline * (temperature - T_REF)
            + np.sum(coefficients * theta * (
                exp_neg_x / (1.0 - exp_neg_x)
                - exp_neg_ref / (1.0 - exp_neg_ref)
            ))
        )
        entropy_oscillator = (
            x * exp_neg_x / (1.0 - exp_neg_x)
            - np.log1p(-exp_neg_x)
        )
        entropy_oscillator_ref = (
            x_ref * exp_neg_ref / (1.0 - exp_neg_ref)
            - np.log1p(-exp_neg_ref)
        )
        s_i = R * (
            baseline * np.log(temperature / T_REF)
            + np.sum(coefficients * (
                entropy_oscillator - entropy_oscillator_ref
            ))
        )
        cp.append(cp_i)
        h.append(h_i)
        s.append(s_i)
    cp = np.asarray(cp)
    h = np.asarray(h)
    s = np.asarray(s)
    return {"cp": cp, "h": h, "s": s, "g": h - temperature * s}


def pr_ideal_helmholtz_density(T, c, names,
                               caloric_model=LEGACY_ZERO_CP0):
    """Ideal-gas Helmholtz density consistent with the PR fugacity scale."""
    concentration = np.asarray(c)
    value = R * T * np.sum(
        concentration * (np.log(concentration * R * T / P_REF) - 1.0)
    )
    if caloric_model == REFERENCE_EOS_CP0:
        value = value + np.dot(
            concentration, ideal_gas_standard_state(T, names)["g"]
        )
    elif caloric_model != LEGACY_ZERO_CP0:
        raise ValueError(f"unknown ideal-gas caloric model: {caloric_model}")
    return value


class MAGEBranch:
    """Finite-support log-sum-exp convex Massieu branch."""

    def __init__(self, eta, phi, q, tau=1.0, center=None, scale=None,
                 name="phase", offset=0.0):
        self.eta = np.asarray(eta, float)
        self.phi = np.asarray(phi, float)
        self.q = np.asarray(q, float)
        if self.eta.ndim != 2 or self.q.shape != self.eta.shape:
            raise ValueError("eta and q must have shape (M,D)")
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        self.center = self.eta.mean(0) if center is None else np.asarray(center, float)
        self.scale = self.eta.std(0) if scale is None else np.asarray(scale, float)
        self.scale = np.where(np.abs(self.scale) < 1.0e-14, 1.0, self.scale)
        self.x = (self.eta - self.center) / self.scale
        self.qx = self.q * self.scale[None, :]
        self.intercept = self.phi - np.einsum("ij,ij->i", self.qx, self.x)
        self.name = str(name)
        self.offset = float(offset)

    @property
    def n_atoms(self):
        return len(self.phi)

    @property
    def dim(self):
        return self.eta.shape[1]

    def evaluate(self, eta, hessian=False):
        z = (np.asarray(eta, float) - self.center) / self.scale
        logits = self.qx @ z + self.intercept
        s = logits / self.tau
        smax = np.max(s)
        w = np.exp(s - smax)
        w /= w.sum()
        # Eq. (MAGE): normalized log-sum-exp, i.e. 1/M inside the sum.
        phi = self.tau * (smax + np.log(np.exp(s - smax).sum()) - np.log(self.n_atoms)) + self.offset
        gx = w @ self.qx
        q = gx / self.scale
        out = {"phi": float(phi), "q": q, "u": float(q[0]), "c": q[1:], "weights": w}
        if hessian:
            second = (self.qx.T * w) @ self.qx
            Hx = (second - np.outer(gx, gx)) / self.tau
            H = Hx / (self.scale[:, None] * self.scale[None, :])
            out["H"] = 0.5 * (H + H.T)
        return out

    def evaluate_batch(self, eta):
        E = np.asarray(eta, float)
        if E.ndim != 2 or E.shape[1] != self.dim:
            raise ValueError("eta must have shape (N,D)")
        # Bound the temporary N x M matrices for the dense nonisothermal
        # representation. Chunking does not alter the numerical definition.
        chunk = max(1, int(5_000_000 // max(self.n_atoms, 1)))
        phi_parts, q_parts = [], []
        for start in range(0, len(E), chunk):
            z = (E[start:start + chunk] - self.center) / self.scale
            logits = z @ self.qx.T + self.intercept[None, :]
            s = logits / self.tau
            m = s.max(1, keepdims=True)
            e = np.exp(s - m)
            w = e / e.sum(1, keepdims=True)
            phi_parts.append(self.tau * (
                m[:, 0] + np.log(e.sum(1)) - np.log(self.n_atoms)
            ) + self.offset)
            q_parts.append((w @ self.qx) / self.scale[None, :])
        return np.concatenate(phi_parts), np.vstack(q_parts)


def farthest_order(X):
    """Deterministic farthest-point order used for nested support sets."""
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    Z = (X - lo) / np.maximum(hi - lo, 1.0e-14)
    first = int(np.argmax(np.sum((Z - Z.mean(0)) ** 2, axis=1)))
    order = [first]
    dist = np.sum((Z - Z[first]) ** 2, axis=1)
    for _ in range(1, len(X)):
        j = int(np.argmax(dist))
        order.append(j)
        dist = np.minimum(dist, np.sum((Z - Z[j]) ** 2, axis=1))
    return np.asarray(order, int)


def relative_errors(model, data):
    phi, q = model.evaluate_batch(data["eta"])
    ep = np.abs(phi - data["phi"]) / np.maximum(np.abs(data["phi"]), 1.0e-12)
    eq = np.linalg.norm(q - data["q"], axis=1) / np.maximum(np.linalg.norm(data["q"], axis=1), 1.0e-12)
    eu = np.abs(q[:, 0] - data["q"][:, 0]) / np.maximum(np.abs(data["q"][:, 0]), 1.0e-12)
    ec = np.linalg.norm(q[:, 1:] - data["q"][:, 1:], axis=1) / np.maximum(np.linalg.norm(data["q"][:, 1:], axis=1), 1.0e-12)
    return {"phi": ep, "gradient": eq, "u": eu, "c": ec, "pred_phi": phi, "pred_q": q}


class PREOS:
    """Classical Peng--Robinson EOS with zero binary interaction parameters."""

    def __init__(self, names=("CH4", "nC4"), kij=None):
        self.names = tuple(names)
        vals = [COMPONENTS[n] for n in self.names]
        self.Tc = np.array([v["Tc"] for v in vals])
        self.Pc = np.array([v["Pc"] for v in vals])
        self.omega = np.array([v["omega"] for v in vals])
        self.MW = np.array([v["MW"] for v in vals])
        n = len(vals)
        self.kij = np.zeros((n, n)) if kij is None else np.asarray(kij, float)

    def parameters(self, T):
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        alpha = (1.0 + kappa * (1.0 - np.sqrt(T / self.Tc))) ** 2
        ai = 0.45724 * R**2 * self.Tc**2 / self.Pc * alpha
        bi = 0.07780 * R * self.Tc / self.Pc
        aij = np.sqrt(ai[:, None] * ai[None, :]) * (1.0 - self.kij)
        return ai, bi, aij

    def z_roots(self, T, P, x):
        x = np.asarray(x, float)
        _, bi, aij = self.parameters(T)
        am = x @ aij @ x
        bm = x @ bi
        A = am * P / (R**2 * T**2)
        B = bm * P / (R * T)
        coeff = [1.0, -(1.0 - B), A - 3 * B * B - 2 * B, -(A * B - B * B - B**3)]
        roots = np.roots(coeff)
        rr = np.sort(roots.real[np.abs(roots.imag) < 1.0e-8])
        if len(rr) == 0:
            raise RuntimeError("no real PR root")
        return rr, A, B, am, bm, bi, aij

    def lnphi(self, T, P, x, phase):
        x = np.asarray(x, float)
        roots, A, B, am, bm, bi, aij = self.z_roots(T, P, x)
        Z = roots[-1] if phase == "vapor" else roots[0]
        if B <= 0 or Z <= B:
            raise RuntimeError("invalid PR logarithm")
        suma = aij @ x
        s2 = np.sqrt(2.0)
        logterm = np.log((Z + (1 + s2) * B) / (Z + (1 - s2) * B))
        ln = bi / bm * (Z - 1) - np.log(Z - B) - A / (2 * s2 * B) * (2 * suma / am - bi / bm) * logterm
        return ln, float(Z), len(roots)

    def state(self, T, P, x, phase):
        x = np.asarray(x, float)
        ln, Z, nroots = self.lnphi(T, P, x, phase)
        c = x * P / (Z * R * T)
        return {"T": float(T), "P": float(P), "x": x, "Z": Z, "c": c,
                "phi": P / (R * T), "mu": R * T * np.log(x * np.exp(ln) * P / P_REF),
                "nroots": nroots}

    def flash(self, T, P, z, tol=1.0e-9, maxiter=120):
        z = np.asarray(z, float)
        K = (self.Pc / P) * np.exp(5.373 * (1 + self.omega) * (1 - self.Tc / T))
        for it in range(1, maxiter + 1):
            f0 = np.sum(z * (K - 1) / (1 + 0 * (K - 1)))
            f1 = np.sum(z * (K - 1) / (1 + 1 * (K - 1)))
            if f0 < 0:
                beta, regime = 0.0, "liquid"
            elif f1 > 0:
                beta, regime = 1.0, "vapor"
            else:
                beta = brentq(lambda b: np.sum(z * (K - 1) / (1 + b * (K - 1))), 0.0, 1.0)
                regime = "two-phase"
            den = 1 + beta * (K - 1)
            x = z / den
            y = K * x
            x /= x.sum(); y /= y.sum()
            lnl, _, _ = self.lnphi(T, P, x, "liquid")
            lnv, _, _ = self.lnphi(T, P, y, "vapor")
            target = lnl - lnv
            err = float(np.max(np.abs(np.log(K) - target)))
            if err < tol:
                return {"success": True, "beta": float(beta), "phase": regime, "x": x, "y": y,
                        "iterations": it, "eos_evals": 2 * it, "residual": err}
            K = np.exp(0.45 * np.log(K) + 0.55 * target)
        return {"success": False, "beta": float(beta), "phase": regime, "x": x, "y": y,
                "iterations": maxiter, "eos_evals": 2 * maxiter, "residual": err}

    def binary_vle(self, T, P, guess=(0.05, 0.8)):
        if len(self.names) != 2:
            raise ValueError("binary_vle is for binary mixtures")
        u0 = np.log(np.clip(np.asarray(guess, float), 1e-7, 1 - 1e-7) / (1 - np.clip(np.asarray(guess, float), 1e-7, 1 - 1e-7)))

        def residual(u):
            a = 1 / (1 + np.exp(-np.clip(u, -30, 30)))
            x = np.array([a[0], 1 - a[0]])
            y = np.array([a[1], 1 - a[1]])
            try:
                l, _, _ = self.lnphi(T, P, x, "liquid")
                v, _, _ = self.lnphi(T, P, y, "vapor")
                return np.log(x) + l - np.log(y) - v
            except Exception:
                return np.ones(2) * 1.0e3

        sol = least_squares(residual, u0, max_nfev=200, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        a = 1 / (1 + np.exp(-np.clip(sol.x, -30, 30)))
        return {"success": bool(sol.success and np.max(np.abs(sol.fun)) < 1e-7),
                "x": np.array([a[0], 1 - a[0]]), "y": np.array([a[1], 1 - a[1]]),
                "nfev": int(sol.nfev), "residual": float(np.max(np.abs(sol.fun)))}


def pr_massieu_from_c(T, c, names, caloric_model=LEGACY_ZERO_CP0):
    """Massieu teacher from the common PR Helmholtz density."""
    # Complex-step differentiation avoids an additional automatic-diff
    # dependency while retaining machine-precision first derivatives.
    T = np.asarray(T, float).reshape(-1)
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = c[None, :]

    def f_density(temp, cc):
        Tc = np.array([COMPONENTS[n]["Tc"] for n in names])
        Pc = np.array([COMPONENTS[n]["Pc"] for n in names])
        om = np.array([COMPONENTS[n]["omega"] for n in names])
        kap = 0.37464 + 1.54226 * om - 0.26992 * om**2
        alpha = (1 + kap * (1 - np.sqrt(temp / Tc))) ** 2
        ai = 0.45724 * R**2 * Tc**2 / Pc * alpha
        # PR co-volume parameter: b_i = 0.07780 R T_c / P_c.
        bi = 0.07780 * R * Tc / Pc
        ctot = np.sum(cc)
        x = cc / ctot
        aij = np.sqrt(ai[:, None] * ai[None, :])
        am = x @ aij @ x
        bm = x @ bi
        v = 1 / ctot
        # The PR fugacity convention is nu_i = log(f_i / P_REF).  The ideal
        # Helmholtz term must therefore use the same concentration scale,
        # c_i R T / P_REF; omitting this factor shifts every chemical
        # potential by R T log(R T / P_REF) and breaks the non-isothermal
        # Massieu coordinate.
        fid = pr_ideal_helmholtz_density(
            temp, cc, names, caloric_model=caloric_model
        )
        s2 = np.sqrt(2.0)
        ar = -R * temp * np.log(1 - bm / v) - am / (2 * s2 * bm) * np.log((v + (1 + s2) * bm) / (v + (1 - s2) * bm))
        return fid + ctot * ar

    eta = np.empty((len(T), 1 + c.shape[1])); phi = np.empty(len(T)); q = np.empty_like(eta); mu_all = np.empty_like(c); p_all = np.empty(len(T))
    h = 1.0e-30
    for j, (temp, cc) in enumerate(zip(T, c)):
        f0 = float(np.real(f_density(temp, cc)))
        mu = np.array([np.imag(f_density(temp, cc.astype(complex) + 1j * h * np.eye(1, len(cc), i)[0])) / h for i in range(len(cc))])
        fT = np.imag(f_density(temp + 1j * h, cc)) / h
        p = float(np.dot(mu, cc) - f0); u = float(f0 - temp * fT)
        mu_all[j] = mu; p_all[j] = p
        eta[j] = np.r_[-1 / (R * temp), mu / (R * temp)]
        phi[j] = p / (R * temp); q[j] = np.r_[u, cc]
    return {"eta": eta, "phi": phi, "q": q, "mu": mu_all, "p": p_all}


def make_pr_pool(eos, names, n, T_range, P_range, x_min=0.005, phase="liquid", seed=0, fixed_T=None):
    """Generate exactly n valid homogeneous PR states by Sobol rejection."""
    from scipy.stats import qmc
    dim = 2 if fixed_T is not None else 3
    sampler = qmc.Sobol(dim, scramble=True, seed=seed)
    rows = []
    batch_power = max(10, int(np.ceil(np.log2(max(2 * n, 1024)))))
    while len(rows) < n:
        U = sampler.random_base2(batch_power) if not rows else sampler.random(2**batch_power)
        for u in U:
            if fixed_T is None:
                T = T_range[0] + u[0] * (T_range[1] - T_range[0]); P = P_range[0] + u[1] * (P_range[1] - P_range[0]); a = x_min + u[2] * (1 - 2 * x_min)
            else:
                T = fixed_T; P = P_range[0] + u[0] * (P_range[1] - P_range[0]); a = x_min + u[1] * (1 - 2 * x_min)
            x = np.array([a, 1 - a])
            try:
                st = eos.state(T, P, x, phase)
                valid = st["Z"] < 0.5 if phase == "liquid" else st["Z"] > 0.5
                if valid and np.all(np.isfinite(st["c"])):
                    rows.append((T, P, a, st["c"].copy()))
                    if len(rows) >= n:
                        break
            except Exception:
                continue
    out = pr_massieu_from_c([r[0] for r in rows], [r[3] for r in rows], names)
    out.update({"T": np.array([r[0] for r in rows]), "P": np.array([r[1] for r in rows]), "x1": np.array([r[2] for r in rows])})
    return out


def solve_mage_binary(T, P, liquid, vapor, guess=None):
    target = P / (R * T)
    if guess is None:
        guess = 0.5 * (liquid.center[1:] + vapor.center[1:])

    def fun(etai):
        eta = np.r_[-1 / (R * T), etai]
        return np.array([(liquid.evaluate(eta)["phi"] - target) / max(target, 1e-9),
                         (vapor.evaluate(eta)["phi"] - target) / max(target, 1e-9)])

    sol = least_squares(fun, np.asarray(guess, float), max_nfev=160, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    eta = np.r_[-1 / (R * T), sol.x]
    ql = liquid.evaluate(eta); qv = vapor.evaluate(eta)
    xl = ql["c"] / ql["c"].sum(); yv = qv["c"] / qv["c"].sum()
    return {"success": bool(sol.success and np.max(np.abs(sol.fun)) < 1e-6), "eta": eta,
            "x": xl, "y": yv, "pL": ql["phi"] * R * T, "pV": qv["phi"] * R * T,
            "nfev": int(sol.nfev), "residual": float(np.max(np.abs(sol.fun)))}


def massieu_phase_state(T, etai, S, liquid, vapor):
    eta = np.r_[-1 / (R * T), np.asarray(etai, float)]
    L = liquid.evaluate(eta); V = vapor.evaluate(eta)
    pL, pV = L["phi"] * R * T, V["phi"] * R * T
    C = (1 - S) * L["c"] + S * V["c"]
    return {"eta": eta, "etai": np.asarray(etai), "S": float(S), "cL": L["c"], "cV": V["c"],
            "C": C, "z": C / C.sum(), "pL": pL, "pV": pV, "Pthermo": (1 - S) * pL + S * pV}


def pr_equilibrium_state(eos, T, P, z):
    r = eos.flash(T, P, z)
    if not r["success"]:
        raise RuntimeError("PR flash did not converge")
    if r["phase"] == "liquid":
        l = eos.state(T, P, z, "liquid")
        tr = pr_massieu_from_c([T], [l["c"]], eos.names)
        return {"P": P, "z": np.asarray(z), "S": 0.0, "cL": l["c"], "cV": l["c"], "C": l["c"], "etai": tr["eta"][0, 1:], "phase": "liquid", "eos_evals": r["eos_evals"]}
    if r["phase"] == "vapor":
        v = eos.state(T, P, z, "vapor")
        tr = pr_massieu_from_c([T], [v["c"]], eos.names)
        return {"P": P, "z": np.asarray(z), "S": 1.0, "cL": v["c"], "cV": v["c"], "C": v["c"], "etai": tr["eta"][0, 1:], "phase": "vapor", "eos_evals": r["eos_evals"]}
    l = eos.state(T, P, r["x"], "liquid"); v = eos.state(T, P, r["y"], "vapor")
    CL, CV = l["c"].sum(), v["c"].sum(); beta = r["beta"]
    S = (beta / CV) / ((1 - beta) / CL + beta / CV)
    C = (1 - S) * l["c"] + S * v["c"]
    tl = pr_massieu_from_c([T], [l["c"]], eos.names); tv = pr_massieu_from_c([T], [v["c"]], eos.names)
    return {"P": P, "z": np.asarray(z), "S": float(S), "cL": l["c"], "cV": v["c"], "C": C,
            "etai": 0.5 * (tl["eta"][0, 1:] + tv["eta"][0, 1:]), "phase": "two-phase", "eos_evals": r["eos_evals"]}


def phase_flux(state, PL, PR, dist, k, mu_l, mu_g):
    S = np.clip(float(state["S"]), 0.0, 1.0)
    ul = k * (1 - S) ** 2 / mu_l * (PL - PR) / dist
    ug = k * S**2 / mu_g * (PL - PR) / dist
    return state["cL"] * ul + state["cV"] * ug


def natural_map(S, pV, pL, Pscale=2.0e6):
    F = (pV - pL) / Pscale
    return float(S - smooth_clip01(S + F))


def smooth_clip01(value, width=1.0e-4):
    """Differentiable approximation of projection onto [0, 1]."""
    value = np.asarray(value, float)
    return width * (
        np.logaddexp(0.0, value / width)
        - np.logaddexp(0.0, (value - 1.0) / width)
    )


def solve_mage_pressure_step(old, inlet, T, liquid, vapor, phi, dx, dt, k,
                             mu_l, mu_g, Pin, Pout, x0=None, max_nfev=600):
    N = len(old); C0 = np.array([q["C"] for q in old]); scale = np.maximum(np.mean(phi * C0 / dt, axis=0), 1e-8); Pscale = 2e6
    if x0 is None:
        E = np.array([q["etai"] for q in old]); S = np.array([q["S"] for q in old]); P = np.array([q["P"] for q in old])
        x0 = np.r_[E[:, 0], E[:, 1], E[:, 2], S, P / 1e6]
    counter = {"residual_calls": 0, "branch_evaluations": 0}

    def residual(w):
        counter["residual_calls"] += 1
        E = np.column_stack([w[:N], w[N:2*N], w[2*N:3*N]]); S = w[3*N:4*N]; P = w[4*N:] * 1e6
        st = [massieu_phase_state(T, E[j], S[j], liquid, vapor) for j in range(N)]; counter["branch_evaluations"] += 2 * N
        F = [phase_flux(inlet, Pin, P[0], dx / 2, k, mu_l, mu_g)]
        F += [phase_flux(st[j], P[j], P[j + 1], dx, k, mu_l, mu_g) for j in range(N - 1)]
        F += [phase_flux(st[-1], P[-1], Pout, dx / 2, k, mu_l, mu_g)]
        rr = []
        for j, q in enumerate(st): rr.extend((phi * (q["C"] - C0[j]) / dt + (F[j + 1] - F[j]) / dx) / scale)
        for j, q in enumerate(st): rr.extend([(P[j] - q["Pthermo"]) / Pscale, natural_map(q["S"], q["pV"], q["pL"], Pscale)])
        return np.asarray(rr)

    lo = np.r_[np.full(3*N, -30.0), np.zeros(N), np.full(N, 0.8 * Pout / 1e6)]
    hi = np.r_[np.full(3*N, 30.0), np.ones(N), np.full(N, 1.1 * Pin / 1e6)]
    sol = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev,
                        xtol=1e-9, ftol=1e-9, gtol=1e-9, x_scale="jac")
    E = np.column_stack([sol.x[:N], sol.x[N:2*N], sol.x[2*N:3*N]]); S = sol.x[3*N:4*N]; P = sol.x[4*N:] * 1e6
    st = [massieu_phase_state(T, E[j], S[j], liquid, vapor) for j in range(N)]
    for j in range(N): st[j]["P"] = P[j]
    return st, sol, counter


def solve_pr_pressure_step(old, inlet, eos, T, phi, dx, dt, k, mu_l, mu_g, Pin, Pout, x0=None):
    N = len(old); C0 = np.array([q["C"] for q in old]); scale = np.maximum(np.mean(phi * C0 / dt, axis=0), 1e-8)
    if x0 is None:
        P = np.array([q["P"] for q in old]); Z = np.array([q["z"] for q in old]); U = np.log(Z[:, :-1] / Z[:, -1, None]); x0 = np.r_[P / 1e6, U.ravel()]
    counter = {"residual_calls": 0, "local_flash_calls": 0, "PR_EOS_evaluations": 0}

    def softmax(u):
        a = np.r_[u, 0.0]; a -= a.max(); e = np.exp(a); return e / e.sum()

    def residual(w):
        counter["residual_calls"] += 1; P = w[:N] * 1e6; U = w[N:].reshape(N, 2); st = []
        try:
            for j in range(N):
                st.append(pr_equilibrium_state(eos, T, P[j], softmax(U[j]))); counter["local_flash_calls"] += 1; counter["PR_EOS_evaluations"] += st[-1]["eos_evals"]
        except Exception:
            return np.ones(3 * N) * 1e4
        F = [phase_flux(inlet, Pin, P[0], dx / 2, k, mu_l, mu_g)]
        F += [phase_flux(st[j], P[j], P[j + 1], dx, k, mu_l, mu_g) for j in range(N - 1)]
        F += [phase_flux(st[-1], P[-1], Pout, dx / 2, k, mu_l, mu_g)]
        rr = []
        for j, q in enumerate(st): rr.extend((phi * (q["C"] - C0[j]) / dt + (F[j + 1] - F[j]) / dx) / scale)
        return np.asarray(rr)

    lo = np.r_[np.full(N, 0.8 * Pout / 1e6), np.full(2*N, -6.0)]; hi = np.r_[np.full(N, 1.1 * Pin / 1e6), np.full(2*N, 6.0)]
    sol = least_squares(residual, x0, bounds=(lo, hi), max_nfev=100, xtol=1e-9, ftol=1e-9, gtol=1e-9, x_scale="jac")
    P = sol.x[:N] * 1e6; U = sol.x[N:].reshape(N, 2); st = [pr_equilibrium_state(eos, T, P[j], softmax(U[j])) for j in range(N)]
    for j in range(N): st[j]["P"] = P[j]
    return st, sol, counter


class GERGTeacher:
    """Thin wrapper over the vendored Thermopack GERG-2008 implementation."""

    def __init__(self, vendor_path=None):
        root = Path(vendor_path or Path(__file__).resolve().parents[2] / "acgp_p4_validation_1lch8jm1" / "vendor")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from thermopack.multiparameter import multiparam
        self.model = multiparam("CO2,C1,NC4", "GERG2008")
        self.vendor_path = str(root)

    @staticmethod
    def _first(x):
        return x[0] if isinstance(x, tuple) else x

    def state(self, T, P, x, phase):
        x = np.asarray(x, float); x = x / x.sum(); flag = self.model.LIQPH if phase == "liquid" else self.model.VAPPH
        v = float(self.model.specific_volume(float(T), float(P), x, flag)[0])
        mu = np.asarray(self._first(self.model.chemical_potential_tv(float(T), v, x)), float)
        u = float(self._first(self.model.internal_energy_tv(float(T), v, x)))
        c = x / v
        return {"T": float(T), "P": float(P), "x": x, "v": v, "mu": mu, "c": c, "u": u / v,
                "phi": float(P / (R * T)), "eta": np.r_[-1 / (R * T), mu / (R * T)], "q": np.r_[u / v, c]}

    def flash(self, T, P, z):
        r = self.model.two_phase_tpflash(float(T), float(P), np.asarray(z, float))
        phase = {0: "two-phase", 1: "liquid", 2: "vapor", 4: "single"}.get(int(r.phase), str(r.phase))
        return {"phase": phase, "x": np.asarray(r.x, float), "y": np.asarray(r.y, float), "beta": float(r.betaV), "phase_id": int(r.phase)}


def make_gerg_pool(teacher, n, P_range=(1.5e6, 10e6), seed=0, phase="liquid"):
    from scipy.stats import qmc
    sampler = qmc.Sobol(4, scramble=True, seed=seed); rows = []
    while len(rows) < n:
        U = sampler.random(2**14)
        for u in U:
            P = P_range[0] + u[0] * (P_range[1] - P_range[0])
            raw = np.array([u[1], u[2], u[3]]) + 0.02; raw /= raw.sum();
            try:
                rows.append(teacher.state(320.0, P, raw, phase))
            except Exception:
                pass
            if len(rows) >= n: break
    return {k: np.asarray([r[k] for r in rows]) for k in ("T", "P", "x", "mu", "c", "u", "phi", "eta", "q")}


class RegularHelmholtzLLE:
    """Analytic binary regular-solution Helmholtz model from Eq. (regular_helmholtz)."""

    def __init__(self, c0=1.0e4, K=1.0e9, p0=5.0e6, chi=3.0, T=300.0):
        self.c0, self.K, self.p0, self.chi, self.T = c0, K, p0, chi, T

    def state(self, P, x, branch=None):
        x = float(x); x = np.clip(x, 1e-9, 1 - 1e-9)
        g = x * np.log(x) + (1 - x) * np.log(1 - x) + self.chi * x * (1 - x)
        gp = np.log(x / (1 - x)) + self.chi * (1 - 2 * x)

        def pressure(c):
            return self.K * c * (c - self.c0) / self.c0**2 + R * self.T * c * (g + x * (1 - x) * 0 + 0) - (R * self.T * c * g - self.p0)

        # The mixing contribution cancels from p for a homogeneous regular solution.
        def pressure(c): return self.K * c * (c - self.c0) / self.c0**2 + self.p0
        # The quadratic pressure law has a minimum near c=c0/2.  Bracketing
        # from that minimum selects the dense positive root throughout the
        # 4.5--5.5 MPa benchmark interval.
        c = brentq(lambda cc: pressure(cc) - P, 0.5 * self.c0, 4 * self.c0)
        f = 0.5 * self.K * ((c - self.c0) / self.c0) ** 2 - self.p0 + R * self.T * c * g
        mu1 = self.K * (c - self.c0) / self.c0**2 + R * self.T * (g + (1 - x) * gp)
        mu2 = self.K * (c - self.c0) / self.c0**2 + R * self.T * (g - x * gp)
        u = 0.5 * self.K * ((c - self.c0) / self.c0) ** 2 - self.p0
        return {"P": P, "x": x, "c": np.array([c * x, c * (1 - x)]), "u": u,
                "eta": np.array([-1 / (R * self.T), mu1 / (R * self.T), mu2 / (R * self.T)]),
                "phi": P / (R * self.T), "q": np.array([u, c * x, c * (1 - x)])}


def make_lle_pool(model, n, P_range, x_range, seed=0):
    from scipy.stats import qmc
    U = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    rows = [model.state(P_range[0] + u[0] * (P_range[1] - P_range[0]), x_range[0] + u[1] * (x_range[1] - x_range[0])) for u in U]
    return {k: np.asarray([r[k] for r in rows]) for k in ("eta", "phi", "q", "P", "x")}

