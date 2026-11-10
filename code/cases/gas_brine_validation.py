"""Run Supplementary Note 5.1: the non-PR gas--brine validation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
CODE = HERE.parents[1]
REPO = CODE.parent
SOURCE = CODE / "unified_selection"


def run(output: Path, quick: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    # Windows Anaconda environments can load Intel OpenMP twice when PyTorch
    # and SciPy are imported together. Keep the public entry point runnable in
    # that environment; these defaults are no-ops when already configured.
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    shutil.copyfile(SOURCE / "results" / "gas_brine_selection.json",
                    output / "gas_brine_selection.json")
    os.environ["UNIFIED_BRINE_OUTPUT_DIR"] = str(output)
    if quick:
        os.environ["UNIFIED_BRINE_QUICK"] = "1"
    else:
        os.environ.pop("UNIFIED_BRINE_QUICK", None)
    sys.path.insert(0, str(SOURCE))
    try:
        import run_brine_authoritative as calculation  # noqa: E402
    except ModuleNotFoundError as exc:
        if not quick or exc.name != "torch":
            raise
        # The compact smoke command remains usable in minimal environments;
        # full reruns still require PyTorch and execute the bundled trainer.
        for name in (
            "gas_brine_authoritative_KKT.csv",
            "gas_brine_authoritative_training.csv",
            "gas_brine_authoritative_summary.json",
        ):
            shutil.copyfile(SOURCE / "results" / name, output / name)
        (output / "case_manifest.json").write_text(
            json.dumps({"case": "Supplementary Note 5.1 gas--brine",
                        "quick": True, "used_bundled_authoritative_outputs": True,
                        "reason": "PyTorch is not installed in this environment"},
                       indent=2) + "\n", encoding="utf-8")
        return

    calculation.main()
    summary = output / "gas_brine_authoritative_summary.json"
    if not summary.exists():
        raise RuntimeError("gas--brine calculation did not write its summary")
    report = json.loads(summary.read_text(encoding="utf-8"))
    (output / "case_manifest.json").write_text(
        json.dumps({"case": "Supplementary Note 5.1 gas--brine",
                    "quick": bool(quick), "summary": report},
                   indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="run a short deterministic KKT smoke test")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "case_results" / "gas_brine_validation")
    args = parser.parse_args()
    run(args.output_dir, args.quick)
    print(f"Gas--brine validation complete: {args.output_dir}")


if __name__ == "__main__":
    main()
