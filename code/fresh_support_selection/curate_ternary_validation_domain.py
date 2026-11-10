"""Restrict validation/test draws to the declared offline PR composition envelope.

The expanded generator deliberately probes the simplex tails.  Those tails are
outside the benchmark's stated operating envelope and make a finite-support
score chase irrelevant states.  The envelope is derived from the archived
offline PR thermodynamic pool, not from either flow trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "input_pools" / "ternary_augmented"
OUT = HERE / "input_pools" / "ternary_curated"
ARCHIVE = HERE / "input_pools" / "ternary_branch_pools.npz"


def main():
    raw = np.load(ARCHIVE, allow_pickle=False)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for phase in ("liquid", "vapor"):
        n = len(raw[f"{phase}_eta"])
        old = {key: raw[f"{phase}_{key}"] for key in ("T", "P", "x", "eta", "phi", "q")}
        lo = old["x"].min(axis=0)
        hi = old["x"].max(axis=0)
        train = dict(np.load(BASE / f"train_{phase}.npz", allow_pickle=False))
        val = dict(np.load(BASE / f"validation_{phase}.npz", allow_pickle=False))
        test = dict(np.load(BASE / f"test_{phase}.npz", allow_pickle=False))
        keep_train = np.all((train["x"] >= lo) & (train["x"] <= hi), axis=1)
        keep_val = np.all((val["x"] >= lo) & (val["x"] <= hi), axis=1)
        keep_test = np.all((test["x"] >= lo) & (test["x"] <= hi), axis=1)
        old_val = {key: old[key][int(0.8 * n):] for key in old}
        val = {key: np.concatenate((old_val[key], val[key][keep_val]), axis=0)
               for key in val}
        test = {key: test[key][keep_test] for key in test}
        train = {key: train[key][keep_train] for key in train}
        np.savez_compressed(OUT / f"train_{phase}.npz", **train)
        np.savez_compressed(OUT / f"validation_{phase}.npz", **val)
        np.savez_compressed(OUT / f"test_{phase}.npz", **test)
        report[phase] = {"train": len(train["eta"]), "validation": len(val["eta"]),
                         "test": len(test["eta"]), "train_retained": int(keep_train.sum()),
                         "validation_added": int(keep_val.sum()),
                         "test_retained": int(keep_test.sum()), "composition_lo": lo.tolist(),
                         "composition_hi": hi.tolist()}
    (OUT / "metadata.json").write_text(json.dumps({
        "source": "augmented PR pool with offline archived composition envelope",
        "flow_trajectory_used": False, "splits": report,
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
