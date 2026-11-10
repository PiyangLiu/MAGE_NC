"""Read-only post-run scientific invariant checks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(path: Path):
    b = json.loads((path/"brine_results.json").read_text())
    e = list(csv.DictReader((path/"embedding_results.csv").open()))
    trace = json.loads((path/"embedding_trace.json").read_text())
    assert len(e) == 10
    for phase in ["liquid", "vapour"]:
        r = [row for row in e if row["phase"]==phase]
        assert [int(row["ambient_Nc"]) for row in r] == [2, 3, 5, 8, 10]
        assert len(set(int(row["M"]) for row in r)) == 1
        assert all(row["stopping_threshold_reached"] == "True" for row in r)
        assert len(set(round(float(row["test_phi_MAE_mol_m3"]), 8) for row in r)) == 1
        assert len(set(round(float(row["test_gradient_rel_mean"]), 8) for row in r)) == 1
        seq = trace["selections"][phase]
        assert all(rec["indices"] == seq[0]["indices"] for rec in seq)
        assert trace["PR_derivative_audits"][phase][
            "max_gradient_relative_discrepancy"] < 1e-7
    assert b["not_full_PR_Pitzer_or_eCPA"] and b["is_not_fixed_p_z_flash"]
    assert b["heldout_test"] == 600 and b["timed_queries"] == 500
    assert b["phase_classification_fraction"] >= .99
    assert b["reference_inversion_max_density_residual"] < 1e-9
    assert b["liquid_accuracy"]["mean_gradient_relative"] < .03
    assert b["gas_accuracy"]["mean_gradient_relative"] < .03
    print("PASS: PR thermodynamic derivative, isometric indices and counts,")
    print("      independent-state accuracy and honest benchmark scope flags.")
    print("      CPU speedup (reference/MAGE):",
          round(b["online_speedup_reference_over_mage"], 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    main(Path(ap.parse_args().outdir))
