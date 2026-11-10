"""Generate an expanded, independently split ternary PR thermodynamic pool.

The flow trajectory is not sampled here.  Each state is drawn from the fixed
pressure/composition operating domain used by the benchmark and evaluated by
the independent PR flash.  The resulting branch records are written as
separate train/validation/test files for the fresh selector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input_pools"
OUT = INPUT / "ternary_expanded"
THERMO = INPUT / "ternary_solver_src"
if not (THERMO / "thermo.py").exists():
    raise FileNotFoundError(f"bundled thermodynamic solver not found: {THERMO}")
sys.path.insert(0, str(THERMO))
from thermo import PengRobinsonFlash, R, T  # noqa: E402


CENTERS = np.asarray([
    [0.04, 0.08, 0.88], [0.08, 0.12, 0.80],
    [0.15, 0.15, 0.70], [0.25, 0.20, 0.55],
    [0.35, 0.23, 0.42], [0.45, 0.25, 0.30],
    [0.55, 0.25, 0.20], [0.65, 0.20, 0.15],
], dtype=float)


def _q0(flash, pressure, composition, concentration, phase, eos):
    """PR residual internal energy density with the benchmark reference.

    The archived pool uses this EOS expression minus 3 MJ/m^3 as its fixed
    energy reference.  Keeping that convention makes the new q[:, 0] values
    commensurate with the existing score normalization.
    """
    x = np.asarray(composition, dtype=float)
    roots = eos.roots_lnphi(float(pressure), x)
    z_factor = roots[0 if phase == "liquid" else -1][0]
    a_mix = float(x @ eos.aij @ x)
    b_mix = float(x @ eos.bi)
    A = a_mix * pressure / (R * R * T * T)
    B = b_mix * pressure / (R * T)
    term = np.log((z_factor + (1.0 + np.sqrt(2.0)) * B) /
                  (z_factor + (1.0 - np.sqrt(2.0)) * B))

    def a_at(temp):
        alpha = (1.0 + eos.kappa * (1.0 - np.sqrt(temp / eos.Tc))) ** 2
        ai = eos.ai0 * alpha
        return float(x @ np.sqrt(ai[:, None] * ai[None, :]) @ x)

    h = 1.0e-3
    da_dT = (a_at(T + h) - a_at(T - h)) / (2.0 * h)
    u_res = R * T * (z_factor - 1.0) + (
        (T * da_dT - a_mix) / (2.0 * np.sqrt(2.0) * b_mix) * term
    )
    return float(u_res * np.sum(concentration) - 3.0e6)


def _record(eos, pressure, overall, phase, flash):
    if phase == "liquid":
        composition = flash["x"] if flash["phase"] == "two" else overall
        concentration = flash["cL"] if flash["phase"] == "two" else flash["C"]
    else:
        composition = flash["y"] if flash["phase"] == "two" else overall
        concentration = flash["cV"] if flash["phase"] == "two" else flash["C"]
    eta = eos.chemical_potential_coordinate(pressure, composition, flash)
    q = np.asarray([_q0(flash, pressure, composition, concentration, phase, eos),
                    *np.asarray(concentration, dtype=float)], dtype=float)
    return {
        "T": float(T), "P": float(pressure), "x": np.asarray(composition),
        "eta": np.r_[-1.0 / (R * T), eta],
        "phi": float(pressure / (R * T)), "q": q,
    }


def _draw(rng, count, concentration):
    center_ids = rng.integers(0, len(CENTERS), size=count)
    out = np.empty((count, 3), dtype=float)
    for i, center_id in enumerate(center_ids):
        # Keep the operating-center cloud, but reserve a fixed fraction for
        # the full simplex.  The latter prevents an independent validation
        # draw from containing a composition tail absent from training.
        if rng.random() < 0.35:
            alpha = np.full(3, 1.5)
        else:
            alpha = np.maximum(CENTERS[center_id] * concentration, 0.15)
        out[i] = rng.dirichlet(alpha)
    return out


def build_split(name, target, seed):
    rng = np.random.default_rng(seed)
    eos = PengRobinsonFlash()
    records = {"liquid": [], "vapor": []}
    attempts = 0
    while min(len(records["liquid"]), len(records["vapor"])) < target:
        pressures = rng.uniform(1.25e6, 3.25e6, size=256)
        compositions = _draw(rng, len(pressures), 55.0 if name == "train" else 45.0)
        for pressure, overall in zip(pressures, compositions):
            attempts += 1
            try:
                flash = eos.flash(float(pressure), overall)
                phases = ("liquid", "vapor") if flash["phase"] == "two" else (flash["phase"],)
                for phase in phases:
                    if len(records[phase]) < target:
                        records[phase].append(_record(eos, pressure, overall, phase, flash))
            except (FloatingPointError, RuntimeError, ValueError):
                continue
        if attempts > target * 80:
            raise RuntimeError(f"unable to fill {name}: {len(records['liquid'])}/{target} liquid, "
                               f"{len(records['vapor'])}/{target} vapor after {attempts} attempts")

    OUT.mkdir(parents=True, exist_ok=True)
    for phase in ("liquid", "vapor"):
        rows = records[phase]
        payload = {key: np.asarray([row[key] for row in rows])
                   for key in ("T", "P", "x", "eta", "phi", "q")}
        np.savez_compressed(OUT / f"{name}_{phase}.npz", **payload)
    return {"split": name, "target_per_branch": target,
            "counts": {phase: len(records[phase]) for phase in records},
            "attempts": attempts, "flash_calls": eos.calls}


def main():
    reports = [build_split("train", 8192, 2026110601),
               build_split("validation", 4096, 2026110602),
               build_split("test", 4096, 2026110603)]
    (OUT / "metadata.json").write_text(json.dumps({
        "domain": {"P_MPa": [1.25, 3.25], "temperature_K": 320.0,
                   "composition_sampling": "Dirichlet mixture around eight fixed operating centers"},
        "splits": reports,
        "flow_trajectory_used": False,
        "energy_reference": "PR residual internal energy density minus 3e6 J/m^3",
    }, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
