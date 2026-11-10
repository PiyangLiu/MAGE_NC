"""Add the archived offline candidate clouds to the expanded train pool.

The archived points are thermodynamic PR states, not flow states.  They are
used only as additional training candidates; the expanded validation and test
sets remain independent and are never merged into training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "input_pools" / "ternary_expanded"
OUT = HERE / "input_pools" / "ternary_augmented"
ARCHIVE = HERE / "input_pools" / "ternary_branch_pools.npz"


def main():
    raw = np.load(ARCHIVE, allow_pickle=False)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for phase in ("liquid", "vapor"):
        train = dict(np.load(BASE / f"train_{phase}.npz", allow_pickle=False))
        validation = dict(np.load(BASE / f"validation_{phase}.npz", allow_pickle=False))
        test = dict(np.load(BASE / f"test_{phase}.npz", allow_pickle=False))
        n = len(raw[f"{phase}_eta"])
        archived_candidate = {key: raw[f"{phase}_{key}"][:int(0.8 * n)]
                              for key in ("T", "P", "x", "eta", "phi", "q")}
        archived_validation = {key: raw[f"{phase}_{key}"][int(0.8 * n):]
                               for key in ("T", "P", "x", "eta", "phi", "q")}
        train = {key: np.concatenate((train[key], archived_candidate[key]), axis=0)
                 for key in train}
        validation = {key: np.concatenate((validation[key], archived_validation[key]), axis=0)
                      for key in validation}
        np.savez_compressed(OUT / f"train_{phase}.npz", **train)
        np.savez_compressed(OUT / f"validation_{phase}.npz", **validation)
        np.savez_compressed(OUT / f"test_{phase}.npz", **test)
        report[phase] = {"train": len(train["eta"]), "validation": len(validation["eta"]),
                         "test": len(test["eta"]), "archived_candidate_added": len(archived_candidate["eta"]),
                         "archived_validation_retained": len(archived_validation["eta"])}
    (OUT / "metadata.json").write_text(json.dumps({
        "source": "expanded PR database plus archived offline candidate clouds",
        "flow_trajectory_used": False,
        "splits": report,
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
