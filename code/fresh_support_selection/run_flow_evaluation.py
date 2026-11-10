"""Independent post-selection ternary flow evaluation.

The flow trajectory is intentionally downstream of support selection.  This
script reads only the fresh selector's support orders and never feeds flow
residuals back into the selector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input_pools"
OLD_SOLVER = INPUT / "ternary_solver_src"
sys.path.insert(0, str(OLD_SOLVER))
from solvers import MAGEGlobalSolver, PRGlobalSolver  # noqa: E402
from run_benchmark import compare  # noqa: E402


ARCHIVED_POOL = INPUT / "ternary_branch_pools.npz"
EXPANDED_POOL = INPUT / "ternary_final"
if not EXPANDED_POOL.exists():
    EXPANDED_POOL = INPUT / "ternary_curated"
if not EXPANDED_POOL.exists():
    EXPANDED_POOL = INPUT / "ternary_expanded"
CONFIG = INPUT / "ternary_benchmark_config.json"
GRID = INPUT / "ternary_time_grid.csv"
OUT = HERE / "results" / "ternary_flow_evaluation"


def shared_offset(pool: dict, orders: dict[str, np.ndarray], tau: float) -> float:
    diffs = []
    for phase in ("liquid", "vapor"):
        ids = orders[phase]
        eta = pool[f"{phase}_eta"][ids]
        phi = pool[f"{phase}_phi"][ids]
        q = pool[f"{phase}_q"][ids]
        vals = []
        for lo in range(0, len(pool[f"{phase}_eta"]), 256):
            e = pool[f"{phase}_eta"][lo:lo + 256]
            logits = phi[None, :] + np.einsum("nmd,md->nm", e[:, None, :] - eta[None, :, :], q)
            scaled = logits / tau
            m = np.max(scaled, axis=1, keepdims=True)
            vals.append(tau * (m[:, 0] + np.log(np.exp(scaled - m).sum(1)) - np.log(len(ids))))
        diffs.append(pool[f"{phase}_phi"] - np.concatenate(vals))
    return float(np.mean(np.concatenate(diffs)))


def main() -> None:
    report = json.loads((HERE / "results" / "ternary_flow_selection.json").read_text(encoding="utf-8"))
    if EXPANDED_POOL.exists():
        pool = {}
        for phase in ("liquid", "vapor"):
            train_name = ("candidate" if (EXPANDED_POOL / f"candidate_{phase}.npz").exists()
                          else "train")
            train = dict(np.load(EXPANDED_POOL / f"{train_name}_{phase}.npz", allow_pickle=False))
            for key, value in train.items():
                pool[f"{phase}_{key}"] = value
    else:
        pool = dict(np.load(ARCHIVED_POOL, allow_pickle=False))
    orders = {phase: np.asarray(report["branches"][phase]["support_order"][:report["branches"][phase]["selected_M"]], int)
              for phase in ("liquid", "vapor")}
    selected_taus = {float(report["branches"][phase]["selected_tau"])
                     for phase in ("liquid", "vapor")}
    if len(selected_taus) != 1:
        raise RuntimeError(f"branches selected incompatible tau values: {selected_taus}")
    tau = selected_taus.pop()
    model_dir = HERE / "results" / "ternary_flow_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    offset = shared_offset(pool, orders, tau)
    np.savez_compressed(model_dir / "mage_ternary_1280.npz",
                        liquid_eta=pool["liquid_eta"][orders["liquid"]],
                        liquid_phi=pool["liquid_phi"][orders["liquid"]],
                        liquid_q=pool["liquid_q"][orders["liquid"]],
                        vapor_eta=pool["vapor_eta"][orders["vapor"]],
                        vapor_phi=pool["vapor_phi"][orders["vapor"]],
                        vapor_q=pool["vapor_q"][orders["vapor"]])
    (model_dir / "mage_model_summary.json").write_text(json.dumps({
        "tau": tau, "shared_phi_offset": offset,
        "supports_per_phase": int(min(len(v) for v in orders.values())),
        "supports_per_phase_by_branch": {k: int(len(v)) for k, v in orders.items()},
        "source": "fresh validation-greedy selector",
        "flow_trajectory_used_for_support_selection": False,
    }, indent=2), encoding="utf-8")
    second_flow = len(sys.argv) > 1 and sys.argv[1].lower() in {"second", "casec"}
    flow_config = (INPUT / "caseC_stronger_second_flow" / "caseC_config.json"
                   if second_flow else CONFIG)
    flow_grid = (INPUT / "caseC_stronger_second_flow" / "caseC_time_grid.csv"
                 if second_flow else GRID)
    flow_out = HERE / "results" / ("ternary_second_flow_evaluation" if second_flow else "ternary_flow_evaluation")
    flow_out.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(flow_grid)
    mage_solver = MAGEGlobalSolver(flow_config, model_dir)
    try:
        mage, mage_events, _, mage_summary = mage_solver.run(grid, flow_out)
    except Exception as exc:
        failure = {
            "status": "failed_downstream_validation",
            "support_selection_completed": True,
            "selection_used_flow": False,
            "error": str(exc),
            "note": "The selected thermodynamic prefix was not replaced or enlarged after this failure.",
        }
        (flow_out / "status.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return
    pr_solver = PRGlobalSolver(flow_config)
    pr, pr_events, _, pr_summary = pr_solver.run(grid, flow_out)
    _, comparison = compare(mage, pr, mage_events, pr_events)
    mage.to_csv(flow_out / "mage_global_states.csv", index=False)
    pr.to_csv(flow_out / "pr_global_states.csv", index=False)
    mage_events.to_csv(flow_out / "mage_phase_events.csv", index=False)
    pr_events.to_csv(flow_out / "pr_phase_events.csv", index=False)
    summary = {"support_selection": "fresh validation-error greedy; flow was downstream only",
               "supports_per_phase": {k: int(len(v)) for k, v in orders.items()},
               "tau": tau, "shared_phi_offset": offset,
               "MAGE_solver": mage_summary, "PR_solver": pr_summary,
               "comparison": comparison}
    (flow_out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    (flow_out / "status.json").write_text(json.dumps({
        "status": "completed",
        "support_selection_completed": True,
        "selection_used_flow": False,
        "accepted_steps": int(len(grid)),
        "space_time_states": int(comparison["space_time_states"]),
        "phase_state_mismatches": int(comparison["phase_state_mismatches"]),
        "phase_event_identity": bool(comparison["phase_event_step_cell_transition_identity"]),
        "max_mage_scaled_residual": float(mage_summary["max_scaled_residual"]),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"supports": summary["supports_per_phase"], "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
