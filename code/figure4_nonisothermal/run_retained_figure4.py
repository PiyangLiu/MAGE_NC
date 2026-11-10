#!/usr/bin/env python3
"""Recompute the earlier retained single-layer-holdout Figure 4 data."""
from pathlib import Path
import argparse, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
import run_mage_examples as examples  # noqa: E402

examples.RESULTS = HERE / "results"
examples.FIGURES = HERE / "figures"
examples.RESULTS.mkdir(parents=True, exist_ok=True)
examples.FIGURES.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    examples.run_nonisothermal(quick=args.quick, reuse_pools=not args.quick)
