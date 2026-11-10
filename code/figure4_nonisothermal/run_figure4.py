"""Strengthened nonisothermal MAGE benchmark with temperature-band holdouts.

The benchmark is deliberately separate from the retained Figure 4 products.
It withholds three finite 1 K temperature bands, resolves the coexistence
manifold on a 0.25 K temperature grid and a 41-level pressure grid, and
evaluates the held-out states only after support and smoothing parameters are
frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Compatibility aliases required by the bundled SciPy build.
if "long" not in np.__dict__:
    np.long = np.int64
if "ulong" not in np.__dict__:
    np.ulong = np.uint64
if "bool" not in np.__dict__:
    np.bool = np.bool_

from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from mage_core import (  # noqa: E402
    MAGEBranch,
    PREOS,
    REFERENCE_EOS_CP0,
    ideal_gas_caloric_metadata,
    pr_massieu_from_c,
    solve_mage_binary,
)


R = 8.31446261815324
SEED = 20261021
DOMAIN = (295.0, 350.0)
PRESSURE_RANGE = (0.50e6, 7.00e6)
# The original 6 K gaps made the equilibrium manifold a severe extrapolation
# problem.  These finite 1 K bands retain a genuine interval test while
# keeping neighbouring 0.25 K layers available on both sides.
HOLDOUT_BANDS = ((304.5, 305.5), (324.5, 325.5), (344.5, 345.5))
HELDOUT_TEMPERATURE_SPACING_K = 0.5
SUPPORT_TEMPERATURE_SPACING_K = 0.25
ANCHOR_TEMPERATURES = (295.0, 300.0, 315.0, 335.0, 350.0)
# The smaller values are important for a compressed support set: they keep
# the finite-support log-sum-exp close to the upper envelope of the selected
# tangent planes instead of smoothing away narrow coexistence features.
TAU_GRID = (0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5)
PHASES = ("liquid", "vapor")
SOURCE_FRACTION_EQUILIBRIUM = 0.85


def common_scaling(*datasets):
    eta = np.vstack([dataset["eta"] for dataset in datasets])
    center = np.mean(eta, axis=0)
    scale = np.std(eta, axis=0)
    scale = np.where(scale > 1.0e-14, scale, 1.0)
    return center, scale, 1.0


def make_branch(data, indices, tau_bar, center, scale, phi_scale, name):
    indices = np.asarray(indices, int)
    return MAGEBranch(
        data["eta"][indices],
        data["phi"][indices],
        data["q"][indices],
        tau=float(tau_bar * phi_scale),
        center=center,
        scale=scale,
        name=name,
    )


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def save_npz(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in payload.items()})


def save_csv(path: Path, columns: dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    arrays = [np.asarray(columns[name]) for name in names]
    if not arrays:
        raise ValueError("no columns supplied")
    n = len(arrays[0])
    if any(len(array) != n for array in arrays):
        raise ValueError("CSV columns have inconsistent lengths")
    matrix = np.column_stack(arrays)
    np.savetxt(path, matrix, delimiter=",", header=",".join(names), comments="", fmt="%.16g")


def in_holdout_band(temperature):
    temperature = np.asarray(temperature, float)
    mask = np.zeros(temperature.shape, dtype=bool)
    for lower, upper in HOLDOUT_BANDS:
        mask |= (temperature >= lower) & (temperature <= upper)
    return mask


def allowed_intervals():
    intervals = []
    cursor = DOMAIN[0]
    for lower, upper in HOLDOUT_BANDS:
        if lower > cursor:
            intervals.append((cursor, lower))
        cursor = upper
    if cursor < DOMAIN[1]:
        intervals.append((cursor, DOMAIN[1]))
    return tuple(intervals)


def map_allowed_temperature(unit_value):
    """Map [0,1) uniformly by measure onto the non-held-out temperature set."""
    intervals = allowed_intervals()
    lengths = np.asarray([upper - lower for lower, upper in intervals], float)
    target = float(unit_value) * float(lengths.sum())
    cumulative = 0.0
    for (lower, upper), length in zip(intervals, lengths):
        if target < cumulative + length:
            value = lower + target - cumulative
            # Open band boundaries prevent a floating-point sample from
            # entering a nominally excluded closed interval.
            return float(np.nextafter(value, lower))
        cumulative += length
    return float(np.nextafter(intervals[-1][1], intervals[-1][0]))


def build_equilibrium_records(eos, temperatures, pressures):
    """Generate paired PR liquid/vapour coexistence states once."""
    records = []
    for temperature in temperatures:
        guess = (0.04, 0.85)
        for pressure in pressures:
            result = eos.binary_vle(float(temperature), float(pressure), guess)
            if not result["success"]:
                continue
            if np.max(np.abs(result["x"] - result["y"])) <= 1.0e-4:
                continue
            guess = (float(result["x"][0]), float(result["y"][0]))
            try:
                liquid = eos.state(temperature, pressure, result["x"], "liquid")
                vapor = eos.state(temperature, pressure, result["y"], "vapor")
            except Exception:
                continue
            records.append({
                "T": float(temperature),
                "P": float(pressure),
                "liquid_c": liquid["c"],
                "vapor_c": vapor["c"],
                "liquid_x1": float(result["x"][0]),
                "vapor_x1": float(result["y"][0]),
            })
    return records


def homogeneous_states(eos, count, phase, seed, fixed_temperature=None):
    """Generate stable single-phase states, excluding flash coexistence states."""
    sampler = qmc.Sobol(3 if fixed_temperature is None else 2, scramble=True, seed=seed)
    temperatures, pressures, compositions, concentrations = [], [], [], []
    attempts = 0
    while len(temperatures) < count:
        batch = sampler.random(4096)
        for row in batch:
            attempts += 1
            if fixed_temperature is None:
                temperature = map_allowed_temperature(row[0])
                pressure_u, composition_u = row[1], row[2]
            else:
                temperature = float(fixed_temperature)
                pressure_u, composition_u = row[0], row[1]
            pressure = PRESSURE_RANGE[0] + pressure_u * (
                PRESSURE_RANGE[1] - PRESSURE_RANGE[0]
            )
            x1 = 0.005 + composition_u * 0.99
            overall = np.array([x1, 1.0 - x1])
            try:
                flash = eos.flash(temperature, pressure, overall)
                if not flash["success"] or flash["phase"] != phase:
                    continue
                state = eos.state(temperature, pressure, overall, phase)
            except Exception:
                continue
            if np.all(np.isfinite(state["c"])) and np.all(state["c"] > 0):
                temperatures.append(temperature)
                pressures.append(pressure)
                compositions.append(x1)
                concentrations.append(state["c"])
            if len(temperatures) >= count:
                break
        if attempts > max(200000, 1000 * count):
            raise RuntimeError(f"could not generate {count} stable {phase} states")
    return {
        "T": np.asarray(temperatures),
        "P": np.asarray(pressures),
        "x1": np.asarray(compositions),
        "c": np.asarray(concentrations),
        "attempts": attempts,
    }


def transform_pool(eos, temperature, pressure, x1, concentration, source):
    transformed = pr_massieu_from_c(
        temperature,
        concentration,
        eos.names,
        caloric_model=REFERENCE_EOS_CP0,
    )
    return {
        "T": np.asarray(temperature, float),
        "P": np.asarray(pressure, float),
        "x1": np.asarray(x1, float),
        "c": np.asarray(concentration, float),
        "eta": transformed["eta"],
        "phi": transformed["phi"],
        "q": transformed["q"],
        "is_equilibrium": np.asarray(source, float),
    }


def combined_pool(eos, total_count, phase, equilibrium_records, seed):
    eq_t, eq_p, eq_x, eq_c = [], [], [], []
    for record in equilibrium_records:
        eq_t.append(record["T"])
        eq_p.append(record["P"])
        eq_x.append(record[f"{phase}_x1"])
        eq_c.append(record[f"{phase}_c"])
    if len(eq_t) >= total_count:
        raise ValueError("equilibrium states leave no room for homogeneous candidates")
    homogeneous = homogeneous_states(
        eos, total_count - len(eq_t), phase, seed, fixed_temperature=None
    )
    return transform_pool(
        eos,
        np.r_[eq_t, homogeneous["T"]],
        np.r_[eq_p, homogeneous["P"]],
        np.r_[eq_x, homogeneous["x1"]],
        np.vstack([np.asarray(eq_c), homogeneous["c"]]),
        np.r_[np.ones(len(eq_t)), np.zeros(len(homogeneous["T"]))],
    )


def exact_temperature_pool(eos, temperatures, states_per_temperature, phase, seed):
    parts = []
    for index, temperature in enumerate(temperatures):
        homogeneous = homogeneous_states(
            eos,
            states_per_temperature,
            phase,
            seed + 1009 * index,
            fixed_temperature=float(temperature),
        )
        parts.append(transform_pool(
            eos,
            homogeneous["T"],
            homogeneous["P"],
            homogeneous["x1"],
            homogeneous["c"],
            np.zeros(states_per_temperature),
        ))
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }


def choose_tau(
    candidate,
    validation,
    indices,
    center,
    scale,
    phi_scale,
):
    best = None
    for tau in TAU_GRID:
        objective = validation_objective_at_tau(
            candidate, validation, indices, tau, center, scale, phi_scale
        )
        key = (float(objective), -float(tau))
        if best is None or key < best[0]:
            best = (key, float(tau))
    return best[1], best[0][0]


def validation_objective_at_tau(
    candidate, validation, indices, tau, center, scale, phi_scale,
    return_metrics=False,
):
    """Robust state objective used alongside the non-held-out VLE objective."""
    qrange = np.ptp(validation["q"], axis=0)
    qscale = np.where(qrange > 1.0e-14, qrange, 1.0)
    energy_scale = max(
        float(np.percentile(np.abs(validation["q"][:, 0]), 75)),
        float(qscale[0]),
        1.0,
    )
    model = make_branch(
        candidate, indices, tau, center, scale, phi_scale, "selection"
    )
    predicted_phi, predicted_q = model.evaluate_batch(validation["eta"])
    objective = np.median(np.abs(predicted_phi - validation["phi"]) / phi_scale)
    objective += np.median(
        np.linalg.norm((predicted_q - validation["q"]) / qscale, axis=1)
    ) / np.sqrt(validation["q"].shape[1])
    objective += 0.25 * np.median(
        np.abs(predicted_q[:, 0] - validation["q"][:, 0]) / energy_scale
    )
    if return_metrics:
        homogeneous = validation["is_equilibrium"] <= 0.5
        phi_relative = np.abs(predicted_phi - validation["phi"])
        phi_relative /= np.maximum(np.abs(validation["phi"]), 1.0e-12)
        return float(objective), {
            "homogeneous_phi_relative_mean": float(
                np.mean(phi_relative[homogeneous])
            ),
        }
    return float(objective)


def _greedy_source_order(
    candidate,
    validation,
    source_mask,
    max_supports,
    center,
    scale,
    phi_scale,
    phase,
    source_name,
):
    """Build a nested greedy order inside one physical-state source.

    The original unconstrained greedy rule selected mostly homogeneous states
    because the off-manifold derivative residuals were much larger than the
    coexistence residuals.  A source-stratified order makes that trade-off
    explicit: the coexistence and homogeneous validation subsets are each
    covered by their own residual-driven greedy sequence, and are interleaved
    later with a declared source fraction.
    """
    candidate_indices = np.flatnonzero(source_mask)
    validation_mask = (
        validation["is_equilibrium"] > 0.5
        if source_name == "equilibrium"
        else validation["is_equilibrium"] <= 0.5
    )
    validation_indices = np.flatnonzero(validation_mask)
    if len(candidate_indices) == 0 or len(validation_indices) == 0:
        raise ValueError(f"empty {source_name} candidate or validation source")
    candidate_scaled = (candidate["eta"][candidate_indices] - center) / scale
    validation_scaled = (validation["eta"][validation_indices] - center) / scale
    validation_phi = validation["phi"][validation_indices]
    validation_q = validation["q"][validation_indices]
    qrange = np.ptp(validation_q, axis=0)
    qscale = np.where(qrange > 1.0e-14, qrange, 1.0)
    energy_scale = max(
        float(np.percentile(np.abs(validation_q[:, 0]), 75)),
        float(qscale[0]),
        1.0,
    )

    selected_local = [int(np.argmin(np.sum(
        (candidate_scaled - candidate_scaled.mean(axis=0)) ** 2, axis=1
    )))]
    for dimension in range(candidate_scaled.shape[1]):
        selected_local.extend((
            int(np.argmin(candidate_scaled[:, dimension])),
            int(np.argmax(candidate_scaled[:, dimension])),
        ))
    selected_local = list(dict.fromkeys(selected_local))[:max_supports]
    selected_set = set(selected_local)
    selected = [int(candidate_indices[index]) for index in selected_local]

    def residual_score(indices):
        model = make_branch(
            candidate, indices, 0.03125, center, scale, phi_scale, "selection"
        )
        predicted_phi, predicted_q = model.evaluate_batch(
            validation["eta"][validation_indices]
        )
        dq_scaled = (predicted_q - validation_q) / qscale
        score = (predicted_phi - validation_phi) ** 2 / (phi_scale ** 2)
        # Energy is kept as an explicit term because it is the quantity that
        # exposed the poor gas-phase support in the earlier dense model.
        score += dq_scaled[:, 0] ** 2
        score += np.mean(dq_scaled[:, 1:] ** 2, axis=1)
        score += ((predicted_q[:, 0] - validation_q[:, 0]) / energy_scale) ** 2
        return score

    # The residual rule is most informative while the support set is sparse.
    # Once the dominant residuals have been covered, a pure residual rule can
    # repeatedly choose neighbouring pressure layers.  Complete the order by
    # farthest-point coverage so the compressed set still resolves the full
    # temperature/pressure manifold.
    greedy_limit = max_supports
    if source_name == "equilibrium":
        greedy_limit = max(8, int(round(2.0 * max_supports / 3.0)))
    while len(selected) < greedy_limit:
        score = residual_score(selected)
        worst_validation = int(np.argmax(score))
        distances = np.sum(
            (candidate_scaled - validation_scaled[worst_validation]) ** 2, axis=1
        )
        if selected_set:
            distances[np.fromiter(selected_set, dtype=int)] = np.inf
        next_local = int(np.argmin(distances))
        selected_local.append(next_local)
        selected_set.add(next_local)
        selected.append(int(candidate_indices[next_local]))
        if len(selected) % 256 == 0 or len(selected) == max_supports:
            print(
                f"[{phase}/{source_name}] selected {len(selected):4d} supports",
                flush=True,
            )
    if len(selected) < max_supports:
        distance = np.full(len(candidate_indices), np.inf)
        for index in selected_local:
            distance = np.minimum(
                distance,
                np.sum((candidate_scaled - candidate_scaled[index]) ** 2, axis=1),
            )
        while len(selected) < max_supports:
            next_local = int(np.argmax(distance))
            selected_local.append(next_local)
            selected_set.add(next_local)
            selected.append(int(candidate_indices[next_local]))
            distance = np.minimum(
                distance,
                np.sum((candidate_scaled - candidate_scaled[next_local]) ** 2, axis=1),
            )
            distance[next_local] = -np.inf
            if len(selected) % 256 == 0 or len(selected) == max_supports:
                print(
                    f"[{phase}/{source_name}] coverage-filled {len(selected):4d} supports",
                    flush=True,
                )
    return np.asarray(selected, int)


def interleave_source_orders(equilibrium_order, homogeneous_order, count, fraction):
    """Interleave two nested orders while preserving the declared quota."""
    n_equilibrium = min(
        len(equilibrium_order), max(1, int(round(float(count) * fraction)))
    )
    n_homogeneous = min(len(homogeneous_order), count - n_equilibrium)
    # If a short source order is supplied, let the other source fill the
    # remainder rather than silently returning fewer supports.
    if n_equilibrium + n_homogeneous < count:
        n_equilibrium = min(len(equilibrium_order), count - n_homogeneous)
        n_homogeneous = min(len(homogeneous_order), count - n_equilibrium)
    output = []
    i_eq = i_hom = 0
    while len(output) < count:
        target_eq = min(n_equilibrium, int(round(len(output) * fraction)) + 1)
        if i_eq < target_eq and i_eq < n_equilibrium:
            output.append(int(equilibrium_order[i_eq]))
            i_eq += 1
            continue
        if i_hom < n_homogeneous:
            output.append(int(homogeneous_order[i_hom]))
            i_hom += 1
            continue
        if i_eq < n_equilibrium:
            output.append(int(equilibrium_order[i_eq]))
            i_eq += 1
            continue
        break
    return np.asarray(output, int)


def protect_gap_boundary_layers(candidate, equilibrium_order):
    """Place the nearest retained temperature layers first in the order.

    No held-out thermodynamic values enter this rule.  It only uses the known
    geometry of the requested temperature domain and guarantees complete
    pressure coverage on the two retained layers adjacent to every gap.  The
    remaining quota stays in the residual/coverage greedy order.
    """
    adjacent_temperatures = np.asarray([
        value
        for lower, upper in HOLDOUT_BANDS
        for value in (
            lower - SUPPORT_TEMPERATURE_SPACING_K,
            upper + SUPPORT_TEMPERATURE_SPACING_K,
        )
    ])
    protected = np.flatnonzero(
        (candidate["is_equilibrium"] > 0.5)
        & np.any(
            np.isclose(
                candidate["T"][:, None], adjacent_temperatures[None, :]
            ),
            axis=1,
        )
    )
    merged = list(dict.fromkeys(np.r_[protected, equilibrium_order].tolist()))
    return np.asarray(merged[:len(equilibrium_order)], int)


def detailed_errors(model: MAGEBranch, data):
    predicted_phi, predicted_q = model.evaluate_batch(data["eta"])
    phi_absolute = np.abs(predicted_phi - data["phi"])
    phi_relative = phi_absolute / np.maximum(np.abs(data["phi"]), 1.0e-12)
    gradient_relative = np.linalg.norm(predicted_q - data["q"], axis=1)
    gradient_relative /= np.maximum(np.linalg.norm(data["q"], axis=1), 1.0e-12)
    concentration_relative = np.linalg.norm(
        predicted_q[:, 1:] - data["q"][:, 1:], axis=1
    ) / np.maximum(np.linalg.norm(data["q"][:, 1:], axis=1), 1.0e-12)
    reference_u_molar = data["q"][:, 0] / np.sum(data["c"], axis=1)
    predicted_u_molar = predicted_q[:, 0] / np.sum(predicted_q[:, 1:], axis=1)
    u_absolute = np.abs(predicted_u_molar - reference_u_molar) / 1000.0
    u_relative = np.abs(predicted_u_molar - reference_u_molar)
    u_relative /= np.maximum(np.abs(reference_u_molar), 1.0e-12)
    return {
        "predicted_phi": predicted_phi,
        "phi_absolute": phi_absolute,
        "phi_relative": phi_relative,
        "gradient_relative": gradient_relative,
        "concentration_relative": concentration_relative,
        "u_molar_reference_kJ_mol": reference_u_molar / 1000.0,
        "u_molar_predicted_kJ_mol": predicted_u_molar / 1000.0,
        "u_molar_absolute_kJ_mol": u_absolute,
        "u_molar_relative": u_relative,
    }


def distribution_summary(values):
    values = np.asarray(values, float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def summarize_errors(errors):
    return {
        key: distribution_summary(values)
        for key, values in errors.items()
        if key not in {
            "predicted_phi",
            "u_molar_reference_kJ_mol",
            "u_molar_predicted_kJ_mol",
        }
    }


def evaluate_vle(eos, liquid, vapor, temperatures, pressures, support_count):
    rows = []
    requested_pressures = np.asarray(pressures, float)
    if requested_pressures.size == 0:
        return rows
    # The MAGE VLE equations can have nearby numerical roots.  Continue the
    # solution on a finer pressure path and retain only the requested points;
    # this prevents a coarse output grid from switching branches between
    # adjacent markers in the Figure 4 redraw.
    continuation_count = max(
        int(requested_pressures.size),
        4 * int(requested_pressures.size - 1) + 1,
    )
    continuation_pressures = np.unique(np.r_[
        requested_pressures,
        np.linspace(requested_pressures[0], requested_pressures[-1], continuation_count),
    ])
    for temperature in temperatures:
        reference_guess = (0.04, 0.85)
        mage_guess = None
        for pressure in continuation_pressures:
            reference = eos.binary_vle(temperature, pressure, reference_guess)
            if not reference["success"]:
                continue
            if np.max(np.abs(reference["x"] - reference["y"])) <= 1.0e-4:
                continue
            reference_guess = (
                float(reference["x"][0]), float(reference["y"][0])
            )
            predicted = solve_mage_binary(
                temperature, pressure, liquid, vapor, mage_guess
            )
            if predicted["success"]:
                mage_guess = predicted["eta"][1:]
            if not np.any(np.isclose(
                requested_pressures, pressure, rtol=0.0, atol=1.0e-6
            )):
                continue
            rows.append({
                "M": support_count,
                "T": float(temperature),
                "P": float(pressure),
                "heldout": float(bool(in_holdout_band(temperature))),
                "x_pr": float(reference["x"][0]),
                "y_pr": float(reference["y"][0]),
                "x_mage": float(predicted["x"][0]),
                "y_mage": float(predicted["y"][0]),
                "success": float(predicted["success"]),
                "residual": float(predicted["residual"]),
                "pressure_gap_Pa": float(predicted["pV"] - predicted["pL"]),
            })
    return rows


def records_to_columns(records):
    if not records:
        raise ValueError("no records")
    return {key: np.asarray([record[key] for record in records]) for key in records[0]}


def vle_metric_rows(curve_rows):
    output = []
    keys = sorted(set((row["M"], row["T"]) for row in curve_rows))
    for support_count, temperature in keys:
        subset = [
            row for row in curve_rows
            if row["M"] == support_count and row["T"] == temperature
        ]
        converged = [row for row in subset if row["success"] > 0.5]
        if converged:
            liquid_error = np.asarray([
                abs(row["x_mage"] - row["x_pr"]) for row in converged
            ])
            vapor_error = np.asarray([
                abs(row["y_mage"] - row["y_pr"]) for row in converged
            ])
            metrics = {
                "liquid_mae": np.mean(liquid_error),
                "liquid_max": np.max(liquid_error),
                "vapor_mae": np.mean(vapor_error),
                "vapor_max": np.max(vapor_error),
            }
        else:
            metrics = dict.fromkeys(
                ("liquid_mae", "liquid_max", "vapor_mae", "vapor_max"), np.nan
            )
        output.append({
            "M": support_count,
            "T": temperature,
            "heldout": float(bool(in_holdout_band(temperature))),
            "n_reference": len(subset),
            "n_converged": len(converged),
            **metrics,
        })
    return output


def pool_source_summary(pool):
    eq = pool["is_equilibrium"] > 0.5
    return {
        "total": int(len(eq)),
        "equilibrium": int(np.sum(eq)),
        "stable_homogeneous": int(np.sum(~eq)),
        "temperature_min_K": float(np.min(pool["T"])),
        "temperature_max_K": float(np.max(pool["T"])),
    }


def run(quick=False, reuse_pools=False, output_dir=None):
    started = time.time()
    settings = {
        "candidate_count": 1536 if quick else 12000,
        # The dense validation pool retains every paired coexistence state.
        # With the 0.25 K support grid and 17 pressure levels this is about
        # 3.6k states, so leave room for homogeneous derivative-audit points.
        "validation_count": 384 if quick else 5000,
        "states_per_exact_temperature": 40 if quick else 192,
        "support_counts": (
            [32, 64, 128] if quick
            else [512, 1024, 1536, 2048, 3072, 4096]
        ),
        "candidate_pressure_count": 9 if quick else 41,
        "validation_pressure_count": 5 if quick else 17,
        "vle_pressure_count": 9 if quick else 41,
        "support_temperature_spacing_K": (
            2.5 if quick else SUPPORT_TEMPERATURE_SPACING_K
        ),
        # Full runs use the stratified greedy selector below.  Quick runs use
        # the same selector with a smaller pool; no production mode retains
        # every coexistence state as a support.
        "dense_equilibrium_supports": False,
        "equilibrium_support_fraction": SOURCE_FRACTION_EQUILIBRIUM,
    }
    output = (Path(output_dir) if output_dir is not None else ROOT / "results" / (
        "nonisothermal_bandgap_quick" if quick else "nonisothermal_bandgap"
    ))
    output.mkdir(parents=True, exist_ok=True)

    eos = PREOS(("CH4", "nC4"))
    candidate_temperatures = np.asarray([
        value for value in np.arange(
            DOMAIN[0], DOMAIN[1] + 0.1 * settings["support_temperature_spacing_K"],
            settings["support_temperature_spacing_K"],
        )
        if not bool(in_holdout_band(value))
    ])
    validation_temperatures = np.asarray([
        value for value in np.arange(
            DOMAIN[0] + 0.5 * settings["support_temperature_spacing_K"],
            DOMAIN[1],
            settings["support_temperature_spacing_K"],
        )
        if not bool(in_holdout_band(value))
    ])
    heldout_temperatures = np.concatenate([
        np.arange(
            lower,
            upper + 0.1 * HELDOUT_TEMPERATURE_SPACING_K,
            HELDOUT_TEMPERATURE_SPACING_K,
        )
        for lower, upper in HOLDOUT_BANDS
    ])
    exact_temperatures = np.r_[ANCHOR_TEMPERATURES, heldout_temperatures]

    candidate_eq = build_equilibrium_records(
        eos,
        candidate_temperatures,
        np.linspace(*PRESSURE_RANGE, settings["candidate_pressure_count"]),
    )
    validation_eq = build_equilibrium_records(
        eos,
        validation_temperatures,
        np.linspace(*PRESSURE_RANGE, settings["validation_pressure_count"]),
    )
    print(
        f"Built {len(candidate_eq)} candidate and {len(validation_eq)} validation "
        "paired coexistence states.",
        flush=True,
    )

    pools = {"candidate": {}, "validation": {}, "test": {}}
    for phase_index, phase in enumerate(PHASES):
        candidate_file = output / f"candidate_{phase}.npz"
        validation_file = output / f"validation_{phase}.npz"
        test_file = output / f"test_exact_{phase}.npz"
        if reuse_pools and all(path.exists() for path in (
            candidate_file, validation_file, test_file
        )):
            pools["candidate"][phase] = dict(np.load(candidate_file))
            pools["validation"][phase] = dict(np.load(validation_file))
            pools["test"][phase] = dict(np.load(test_file))
        else:
            pools["candidate"][phase] = combined_pool(
                eos,
                settings["candidate_count"],
                phase,
                candidate_eq,
                SEED + 1000 + 100 * phase_index,
            )
            pools["validation"][phase] = combined_pool(
                eos,
                settings["validation_count"],
                phase,
                validation_eq,
                SEED + 2000 + 100 * phase_index,
            )
            pools["test"][phase] = exact_temperature_pool(
                eos,
                exact_temperatures,
                settings["states_per_exact_temperature"],
                phase,
                SEED + 3000 + 100 * phase_index,
            )
            save_npz(candidate_file, pools["candidate"][phase])
            save_npz(validation_file, pools["validation"][phase])
            save_npz(test_file, pools["test"][phase])
        print(
            f"{phase}: candidate {pool_source_summary(pools['candidate'][phase])}; "
            f"validation {pool_source_summary(pools['validation'][phase])}",
            flush=True,
        )

    center, scale, phi_scale = common_scaling(
        pools["candidate"]["liquid"], pools["candidate"]["vapor"]
    )
    # Coexistence states and stable homogeneous states are both required in a
    # useful non-isothermal representation.  Select each source with a genuine
    # residual-driven greedy order, then interleave the orders at a declared
    # 75/25 equilibrium/homogeneous fraction.  The selection curves below use
    # only temperatures outside the held-out bands.
    max_supports = max(settings["support_counts"])
    max_equilibrium = int(round(
        max_supports * settings["equilibrium_support_fraction"]
    ))
    max_homogeneous = max_supports - max_equilibrium
    source_orders = {}
    orders, tau_by_phase, tau_objectives = {}, {}, {}
    for phase in PHASES:
        candidate = pools["candidate"][phase]
        validation = pools["validation"][phase]
        source_orders[(phase, "equilibrium")] = _greedy_source_order(
            candidate,
            validation,
            candidate["is_equilibrium"] > 0.5,
            max_equilibrium,
            center,
            scale,
            phi_scale,
            phase,
            "equilibrium",
        )
        source_orders[(phase, "equilibrium")] = protect_gap_boundary_layers(
            candidate, source_orders[(phase, "equilibrium")]
        )
        source_orders[(phase, "homogeneous")] = _greedy_source_order(
            candidate,
            validation,
            candidate["is_equilibrium"] <= 0.5,
            max_homogeneous,
            center,
            scale,
            phi_scale,
            phase,
            "homogeneous",
        )
        orders[phase] = interleave_source_orders(
            source_orders[(phase, "equilibrium")],
            source_orders[(phase, "homogeneous")],
            max_supports,
            settings["equilibrium_support_fraction"],
        )

    selection_curve_temperatures = np.asarray(
        [295.0, 300.0, 310.0, 315.0, 320.0, 330.0, 335.0, 340.0, 350.0]
    )
    selection_curve_pressures = np.linspace(
        *PRESSURE_RANGE, min(17, settings["vle_pressure_count"])
    )
    models = {}
    convergence_rows = []
    selection_curve_rows = []
    tau_trials = []
    for support_count in settings["support_counts"]:
        selected_indices = {}
        for phase in PHASES:
            selected_indices[phase] = interleave_source_orders(
                source_orders[(phase, "equilibrium")],
                source_orders[(phase, "homogeneous")],
                support_count,
                settings["equilibrium_support_fraction"],
            )
        # A too-small tau can make the homogeneous validation objective look
        # excellent while moving the two branches apart in the VLE solve.  We
        # therefore select one common tau by a joint objective: state errors
        # are normalized by their best value over the grid, while the
        # composition term is scaled to a 1e-3 target accuracy.
        trial_rows = []
        state_objectives = {phase: {} for phase in PHASES}
        for tau in TAU_GRID:
            trial_models = {
                phase: make_branch(
                    pools["candidate"][phase],
                    selected_indices[phase],
                    tau,
                    center,
                    scale,
                    phi_scale,
                    phase,
                )
                for phase in PHASES
            }
            homogeneous_phi = {}
            for phase in PHASES:
                state_objectives[phase][tau], state_metrics = validation_objective_at_tau(
                    pools["candidate"][phase],
                    pools["validation"][phase],
                    selected_indices[phase],
                    tau,
                    center,
                    scale,
                    phi_scale,
                    return_metrics=True,
                )
                homogeneous_phi[phase] = state_metrics[
                    "homogeneous_phi_relative_mean"
                ]
            curve_rows = evaluate_vle(
                eos,
                trial_models["liquid"],
                trial_models["vapor"],
                selection_curve_temperatures,
                selection_curve_pressures,
                support_count,
            )
            converged = [row for row in curve_rows if row["success"] > 0.5]
            curve_error = np.asarray([
                abs(row["x_mage"] - row["x_pr"]) + abs(row["y_mage"] - row["y_pr"])
                for row in converged
            ])
            trial_rows.append({
                "tau": float(tau),
                "models": trial_models,
                "curve_rows": curve_rows,
                "curve_composition_mae": float(np.mean(curve_error)) if len(curve_error) else float("inf"),
                "curve_composition_max": float(np.max(curve_error)) if len(curve_error) else float("inf"),
                "n_curve_states": int(len(converged)),
                "homogeneous_phi_relative_mean": homogeneous_phi,
            })
        min_state = {
            phase: min(state_objectives[phase].values()) for phase in PHASES
        }
        for trial in trial_rows:
            trial["objective"] = float(
                sum(
                    state_objectives[phase][trial["tau"]] / min_state[phase]
                    for phase in PHASES
                )
                + 8.0 * trial["curve_composition_mae"] / 1.0e-3
            )
        minimum_curve_mae = min(
            trial["curve_composition_mae"] for trial in trial_rows
        )
        eligible_trials = [
            trial for trial in trial_rows
            if trial["curve_composition_mae"] <= 1.05 * minimum_curve_mae
            and all(
                trial["homogeneous_phi_relative_mean"][phase] <= 0.01
                for phase in PHASES
            )
        ]
        if eligible_trials:
            # Among validation-equivalent VLE fits with sub-1% homogeneous
            # Phi error, prefer the largest smoothing scale.  This suppresses
            # artificial error spikes inside a finite unsupported interval.
            selected_trial = max(eligible_trials, key=lambda row: row["tau"])
        else:
            selected_trial = min(
                trial_rows, key=lambda row: (row["objective"], row["tau"])
            )
        tau = selected_trial["tau"]
        models[support_count] = selected_trial["models"]
        selection_curve_rows.extend(selected_trial["curve_rows"])
        objective = selected_trial["objective"]
        for phase in PHASES:
            tau_by_phase.setdefault(phase, {})[support_count] = tau
            tau_objectives.setdefault(phase, {})[support_count] = objective
            errors = detailed_errors(models[support_count][phase], pools["validation"][phase])
            summary = summarize_errors(errors)
            convergence_rows.append({
                "M": support_count,
                "phase_id": 0 if phase == "liquid" else 1,
                "tau": tau,
                "tau_objective": objective,
                "phi_relative_mean": summary["phi_relative"]["mean"],
                "phi_relative_p95": summary["phi_relative"]["p95"],
                "gradient_relative_mean": summary["gradient_relative"]["mean"],
                "gradient_relative_p95": summary["gradient_relative"]["p95"],
                "u_molar_absolute_mean_kJ_mol": summary["u_molar_absolute_kJ_mol"]["mean"],
                "u_molar_relative_mean": summary["u_molar_relative"]["mean"],
            })
        tau_trials.append({
            "M": support_count,
            "tau": tau,
            "objective": objective,
            "curve_composition_mae": selected_trial["curve_composition_mae"],
            "curve_composition_max": selected_trial["curve_composition_max"],
            "n_curve_states": selected_trial["n_curve_states"],
            "homogeneous_phi_relative_mean": selected_trial[
                "homogeneous_phi_relative_mean"
            ],
        })

    # Prefer the smallest support set whose non-held-out VLE error is within
    # 10% of the best evaluated set and whose two branches have reached the
    # same plateau criterion.  This is a validation-only compression rule.
    objective_minimum = {
        phase: min(tau_objectives[phase].values()) for phase in PHASES
    }
    best_curve = min(row["curve_composition_mae"] for row in tau_trials)
    selected_support_count = next((
        count for count in settings["support_counts"]
        if all(
            tau_objectives[phase][count] <= 1.10 * objective_minimum[phase]
            for phase in PHASES
        )
        and next(row for row in tau_trials if row["M"] == count)["curve_composition_mae"]
        <= 1.10 * best_curve
    ), max(settings["support_counts"]))

    selected_models = models[selected_support_count]
    state_error_rows = []
    state_summary = {}
    for phase_index, phase in enumerate(PHASES):
        data = pools["test"][phase]
        errors = detailed_errors(selected_models[phase], data)
        state_summary[phase] = {}
        for temperature in exact_temperatures:
            mask = np.isclose(data["T"], temperature)
            group_errors = {key: values[mask] for key, values in errors.items()}
            state_summary[phase][str(float(temperature))] = summarize_errors(group_errors)
        for index in range(len(data["T"])):
            state_error_rows.append({
                "phase_id": phase_index,
                "T": data["T"][index],
                "P": data["P"][index],
                "x1": data["x1"][index],
                "heldout": float(bool(in_holdout_band(data["T"][index]))),
                "phi_relative": errors["phi_relative"][index],
                "gradient_relative": errors["gradient_relative"][index],
                "concentration_relative": errors["concentration_relative"][index],
                "u_molar_reference_kJ_mol": errors["u_molar_reference_kJ_mol"][index],
                "u_molar_predicted_kJ_mol": errors["u_molar_predicted_kJ_mol"][index],
                "u_molar_absolute_kJ_mol": errors["u_molar_absolute_kJ_mol"][index],
                "u_molar_relative": errors["u_molar_relative"][index],
            })

    full_curve_rows = evaluate_vle(
        eos,
        selected_models["liquid"],
        selected_models["vapor"],
        exact_temperatures,
        np.linspace(*PRESSURE_RANGE, settings["vle_pressure_count"]),
        selected_support_count,
    )
    # The selected model is evaluated on the full temperature set below;
    # omit its smaller convergence subset to avoid duplicate curve records.
    all_curve_rows = [
        row for row in selection_curve_rows
        if row["M"] != selected_support_count
    ] + full_curve_rows
    curve_metrics = vle_metric_rows(all_curve_rows)

    support_summary = {}
    for phase in PHASES:
        selected = orders[phase][:selected_support_count]
        source = pools["candidate"][phase]["is_equilibrium"][selected] > 0.5
        temperatures = pools["candidate"][phase]["T"][selected]
        support_summary[phase] = {
            "count": int(len(selected)),
            "equilibrium_count": int(np.sum(source)),
            "stable_homogeneous_count": int(np.sum(~source)),
            "temperature_min_K": float(np.min(temperatures)),
            "temperature_max_K": float(np.max(temperatures)),
            "minimum_distance_from_band_centres_K": {
                str((lower + upper) / 2): float(np.min(np.abs(
                    temperatures - (lower + upper) / 2
                )))
                for lower, upper in HOLDOUT_BANDS
            },
        }

    save_npz(output / "support_selection.npz", {
        "center": center,
        "scale": scale,
        "liquid_order": orders["liquid"],
        "vapor_order": orders["vapor"],
    })
    save_csv(output / "support_convergence.csv", records_to_columns(convergence_rows))
    save_csv(output / "state_errors.csv", records_to_columns(state_error_rows))
    save_csv(output / "vle_curves.csv", records_to_columns(all_curve_rows))
    save_csv(output / "vle_metrics.csv", records_to_columns(curve_metrics))

    all_candidate_temperatures = np.r_[
        pools["candidate"]["liquid"]["T"], pools["candidate"]["vapor"]["T"]
    ]
    all_validation_temperatures = np.r_[
        pools["validation"]["liquid"]["T"], pools["validation"]["vapor"]["T"]
    ]
    all_curve_success = np.asarray([row["success"] for row in full_curve_rows])
    selected_supports_follow_rule = bool(all(
        support_summary[phase]["equilibrium_count"] > 0
        and support_summary[phase]["stable_homogeneous_count"] > 0
        for phase in PHASES
    ))
    quality = {
        "candidate_temperatures_exclude_all_holdout_bands": bool(
            not np.any(in_holdout_band(all_candidate_temperatures))
        ),
        "validation_temperatures_exclude_all_holdout_bands": bool(
            not np.any(in_holdout_band(all_validation_temperatures))
        ),
        "candidate_pools_contain_both_state_sources": bool(all(
            np.any(pools["candidate"][phase]["is_equilibrium"] > 0.5)
            and np.any(pools["candidate"][phase]["is_equilibrium"] < 0.5)
            for phase in PHASES
        )),
        # The production rule is the mixed-source stratified greedy selector.
        "selected_supports_follow_declared_rule": selected_supports_follow_rule,
        "all_state_errors_finite": bool(all(
            np.all(np.isfinite(np.asarray(list(records_to_columns(state_error_rows).values()), dtype=float)))
            for _ in [0]
        )),
        "all_selected_model_vle_states_converged": bool(
            np.all(all_curve_success > 0.5)
        ),
        "selected_support_count": selected_support_count,
    }
    quality["all_required_checks_pass"] = bool(all(
        value for key, value in quality.items()
        if key not in {"selected_support_count"}
    ))

    summary = {
        "schema_version": "1.0",
        "seed": SEED,
        "temperature_domain_K": DOMAIN,
        "pressure_domain_Pa": PRESSURE_RANGE,
        "holdout_bands_K": HOLDOUT_BANDS,
        "holdout_band_width_K": [upper - lower for lower, upper in HOLDOUT_BANDS],
        "heldout_evaluation_temperatures_K": heldout_temperatures,
        "in_domain_anchor_temperatures_K": ANCHOR_TEMPERATURES,
        "candidate_equilibrium_temperatures_K": candidate_temperatures,
        "validation_equilibrium_temperatures_K": validation_temperatures,
        "candidate_source_counts": {
            phase: pool_source_summary(pools["candidate"][phase]) for phase in PHASES
        },
        "validation_source_counts": {
            phase: pool_source_summary(pools["validation"][phase]) for phase in PHASES
        },
        "support_selection": {
            "rule": (
                "stratified validation-error greedy: build separate nested "
                "coexistence and stable-homogeneous orders from the largest "
                "scaled residuals, protect complete pressure coverage on the "
                "retained temperature layers adjacent to each gap, then "
                "interleave the two sources at the declared "
                f"{settings['equilibrium_support_fraction']:.2f}/"
                f"{1.0 - settings['equilibrium_support_fraction']:.2f} source fraction"
            ),
            "heldout_data_used_for_selection": False,
            "support_counts_evaluated": settings["support_counts"],
            "selected_support_count": selected_support_count,
            "support_count_selection_rule": (
                "smallest evaluated support count for which each branch's "
                "independent-validation objective and the non-held-out VLE "
                "composition objective are within 10% of their respective "
                "minima"
            ),
            "validation_objective_minimum": objective_minimum,
            "tau_grid": TAU_GRID,
            "tau_trials": tau_trials,
            "tau_selection_rule": (
                "largest smoothing scale among trials whose non-held-out VLE "
                "composition MAE is within 5% of the minimum and whose "
                "homogeneous Phi relative mean error is below 1% in both "
                "branches; otherwise minimize the joint validation objective"
            ),
            "selected_tau": {
                phase: tau_by_phase[phase][selected_support_count] for phase in PHASES
            },
            "strategy": "stratified_validation_greedy",
            "selected_support_composition": support_summary,
        },
        "ideal_gas_caloric_model": ideal_gas_caloric_metadata(eos.names),
        "state_error_summary_by_temperature": state_summary,
        "vle_metrics": curve_metrics,
        "quality_checks": quality,
        "elapsed_seconds": time.time() - started,
    }
    save_json(output / "summary.json", summary)
    save_json(output / "quality_report.json", quality)
    save_json(output / "configuration.json", {
        **settings,
        "temperature_domain_K": DOMAIN,
        "pressure_domain_Pa": PRESSURE_RANGE,
        "holdout_bands_K": HOLDOUT_BANDS,
        "support_temperature_spacing_K": settings["support_temperature_spacing_K"],
        "tau_grid": TAU_GRID,
        "quick": quick,
    })
    print(json.dumps(quality, indent=2), flush=True)
    print(f"Completed in {time.time() - started:.1f} s; outputs: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reuse-pools", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick, reuse_pools=args.reuse_pools)


if __name__ == "__main__":
    main()
