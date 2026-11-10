"""Case data loaders for the fresh selector.

Only thermodynamic candidate/validation/test pools are loaded here.  No
published support order or previous selection manifest is read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

# The bundled SciPy build still references aliases removed by NumPy 1.26.
if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input_pools"


def _npz(path: Path) -> dict:
    return {k: np.asarray(v) for k, v in np.load(path, allow_pickle=False).items()}


def _split_prefixed(raw: dict, split: str) -> dict:
    prefix = split + "_"
    return {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}


def load_nonisothermal() -> dict:
    base = INPUT / "nonisothermal_bandgap"
    return {
        "liquid": {s: _npz(base / f"{s if s != 'test' else 'test_exact'}_liquid.npz")
                   for s in ("candidate", "validation", "test")},
        "vapor": {s: _npz(base / f"{s if s != 'test' else 'test_exact'}_vapor.npz")
                  for s in ("candidate", "validation", "test")},
    }


def load_gerg() -> dict:
    base = INPUT / "gerg2008"
    out = {}
    for phase in ("liquid", "vapor"):
        raw = _npz(base / f"gerg_{phase}_pools.npz")
        out[phase] = {s: _split_prefixed(raw, s) for s in ("candidate", "validation", "test")}
    return out


def load_brine() -> dict:
    base = INPUT / "brine_data_expanded"
    out = {}
    for phase in ("liquid", "gas"):
        def convert(path: Path) -> dict:
            d = _npz(path)
            return {"eta": d["eta"], "phi": d["psi"], "q": d["q"],
                    **{k: d[k] for k in d if k not in {"eta", "psi", "q"}}}
        out[phase] = {"candidate": convert(base / f"train_{phase}.npz"),
                      "validation": convert(base / f"monitor_{phase}.npz"),
                      "test": convert(base / f"test_{phase}.npz")}
    return out


class _RegularLLE:
    """Small independent regular-solution teacher used only to build LLE pools."""
    R = 8.31446261815324

    def __init__(self, c0=1.0e4, K=1.0e9, p0=5.0e6, chi=3.0, T=300.0):
        self.c0, self.K, self.p0, self.chi, self.T = c0, K, p0, chi, T

    def state(self, pressure: float, composition: float) -> dict:
        from scipy.optimize import brentq
        x = float(np.clip(composition, 1.0e-9, 1.0 - 1.0e-9))
        g = x * np.log(x) + (1.0 - x) * np.log(1.0 - x) + self.chi * x * (1.0 - x)
        gp = np.log(x / (1.0 - x)) + self.chi * (1.0 - 2.0 * x)
        def p(c):
            return self.K * c * (c - self.c0) / self.c0 ** 2 + self.p0
        c = brentq(lambda cc: p(cc) - pressure, 0.5 * self.c0, 4.0 * self.c0)
        f = 0.5 * self.K * ((c - self.c0) / self.c0) ** 2 - self.p0 + self.R * self.T * c * g
        mu1 = self.K * (c - self.c0) / self.c0 ** 2 + self.R * self.T * (g + (1.0 - x) * gp)
        mu2 = self.K * (c - self.c0) / self.c0 ** 2 + self.R * self.T * (g - x * gp)
        u = 0.5 * self.K * ((c - self.c0) / self.c0) ** 2 - self.p0
        return {"P": pressure, "x": x, "eta": np.array([-1.0 / (self.R * self.T), mu1 / (self.R * self.T), mu2 / (self.R * self.T)]),
                "phi": pressure / (self.R * self.T), "q": np.array([u, c * x, c * (1.0 - x)])}


def _lle_pool(model: _RegularLLE, n: int, x_range: tuple[float, float], seed: int) -> dict:
    from scipy.stats import qmc
    exponent = int(np.ceil(np.log2(n)))
    uv = qmc.Sobol(2, scramble=True, seed=seed).random_base2(exponent)[:n]
    rows = [model.state(4.5e6 + u[0] * 1.0e6,
                        x_range[0] + u[1] * (x_range[1] - x_range[0])) for u in uv]
    return {k: np.asarray([row[k] for row in rows]) for k in ("eta", "phi", "q", "P", "x")}


def load_lle() -> dict:
    model = _RegularLLE()
    out = {}
    for phase, xr, seed in (("L1", (0.005, 0.20), 120), ("L2", (0.80, 0.995), 130)):
        out[phase] = {"candidate": _lle_pool(model, 20000, xr, seed),
                      "validation": _lle_pool(model, 5000, xr, seed + 1),
                      "test": _lle_pool(model, 5000, xr, seed + 2)}
    return out


def load_ternary_flow() -> dict:
    final = INPUT / "ternary_final"
    if final.exists():
        return {
            phase: {
                split: _npz(final / f"{split}_{phase}.npz")
                for split in ("candidate", "validation", "test")
            }
            for phase in ("liquid", "vapor")
        }
    expanded = INPUT / "ternary_curated"
    if not expanded.exists():
        expanded = INPUT / "ternary_augmented"
    if not expanded.exists():
        expanded = INPUT / "ternary_expanded"
    if expanded.exists():
        out = {}
        for phase in ("liquid", "vapor"):
            out[phase] = {
                split: _npz(expanded / f"{split}_{phase}.npz")
                for split in ("train", "validation", "test")
            }
            out[phase]["candidate"] = out[phase].pop("train")
        return out
    path = INPUT / "ternary_branch_pools.npz"
    raw = _npz(path)
    out = {}
    # The archived ternary pool has the same deterministic convention as the
    # flow benchmark: first 80% is the candidate cloud and the final 20% is
    # the validation holdout.  There is no separately generated thermodynamic
    # test pool for this case; the flow trajectory is the downstream check.
    for phase in ("liquid", "vapor"):
        data = {k: raw[f"{phase}_{k}"] for k in ("T", "P", "x", "eta", "phi", "q")}
        n = len(data["eta"])
        n_candidate = int(np.floor(0.8 * n))
        slices = {"candidate": slice(0, n_candidate),
                  "validation": slice(n_candidate, n),
                  "test": slice(n_candidate, n)}
        out[phase] = {s: {k: v[idx] for k, v in data.items()} for s, idx in slices.items()}
    return out


def case_loaders() -> dict:
    return {"nonisothermal": load_nonisothermal, "gerg": load_gerg,
            "liquid_liquid": load_lle, "gas_brine": load_brine,
            "ternary_flow": load_ternary_flow}
