"""Build independent centered ternary validation/test pools for final selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import generate_ternary_support_database as generator

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "input_pools" / "ternary_branch_pools.npz"
OUT = HERE / "input_pools" / "ternary_final"


def make_branch(phase, count, seed):
    rng = np.random.default_rng(seed)
    eos = generator.PengRobinsonFlash()
    rows = []
    while len(rows) < count:
        pressures = rng.uniform(1.25e6, 3.25e6, size=256)
        centers = rng.integers(0, len(generator.CENTERS), size=256)
        compositions = np.asarray([
            rng.dirichlet(np.maximum(generator.CENTERS[i] * 100.0, 0.15))
            for i in centers
        ])
        for pressure, composition in zip(pressures, compositions):
            flash = eos.flash(float(pressure), composition)
            if phase == "liquid" and flash["phase"] not in ("liquid", "two"):
                continue
            if phase == "vapor" and flash["phase"] not in ("vapor", "two"):
                continue
            rows.append(generator._record(eos, pressure, composition, phase, flash))
            if len(rows) >= count:
                break
    return {key: np.asarray([row[key] for row in rows])
            for key in ("T", "P", "x", "eta", "phi", "q")}


def main():
    archive = np.load(ARCHIVE, allow_pickle=False)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for phase in ("liquid", "vapor"):
        n = len(archive[f"{phase}_eta"])
        candidate = {key: archive[f"{phase}_{key}"][:int(0.8 * n)]
                     for key in ("T", "P", "x", "eta", "phi", "q")}
        validation = make_branch(phase, 4096, 2026110701 + (0 if phase == "liquid" else 1))
        test = make_branch(phase, 4096, 2026110703 + (0 if phase == "liquid" else 1))
        np.savez_compressed(OUT / f"candidate_{phase}.npz", **candidate)
        np.savez_compressed(OUT / f"validation_{phase}.npz", **validation)
        np.savez_compressed(OUT / f"test_{phase}.npz", **test)
        report[phase] = {"candidate": len(candidate["eta"]),
                         "validation": len(validation["eta"]),
                         "test": len(test["eta"])}
    (OUT / "metadata.json").write_text(json.dumps({
        "source": "archived offline candidate cloud plus independent centered PR draws",
        "flow_trajectory_used": False, "splits": report,
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
