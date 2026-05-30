from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from lagrangian_background_integrator import (
    N_INITIAL,
    OMEGA_PHI0_TARGET,
    TRAJECTORY_PATH,
    cpl_from_trajectory,
    integrate_background,
    project_targets,
    sample_e_at_redshift,
    variants,
)


REPORT_PATH = Path("outputs/reports/e5_137_lagrangian_initial_velocity_scan.md")
TABLE_PATH = Path("outputs/tables/e5_137_lagrangian_initial_velocity_scan.csv")
BEST_TRAJECTORY_PATH = Path("outputs/tables/e5_137_lagrangian_initial_velocity_best_trajectories.csv")

SCAN_INITIAL_MU_XPRIME = np.array(
    [
        -2.2,
        -1.8,
        -1.4,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.4,
        1.8,
        2.2,
    ],
    dtype=float,
)
EARLY_DARK_ENERGY_SANITY_LIMIT = 0.01
W0_WINDOW = 0.02
SHOOT_STEPS = 500
FINAL_STEPS = 1_500
SHOOT_ITERATIONS = 30


def shoot_potential_scale_fast(
    *,
    mu: float,
    harmonics: tuple[tuple[int, float], ...],
    initial_phase: float,
    initial_velocity: float,
) -> float | None:
    low = 1.0e-10
    high = 5.0
    high_record = integrate_background(
        mu=mu,
        potential_scale=high,
        harmonics=harmonics,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
        steps=SHOOT_STEPS,
    )
    while (
        high_record is not None
        and high_record["omega_phi"] < OMEGA_PHI0_TARGET
        and high < 1.0e4
    ):
        high *= 2.0
        high_record = integrate_background(
            mu=mu,
            potential_scale=high,
            harmonics=harmonics,
            initial_phase=initial_phase,
            initial_velocity=initial_velocity,
            steps=SHOOT_STEPS,
        )
    if high_record is None:
        return None
    for _ in range(SHOOT_ITERATIONS):
        mid = 0.5 * (low + high)
        mid_record = integrate_background(
            mu=mu,
            potential_scale=mid,
            harmonics=harmonics,
            initial_phase=initial_phase,
            initial_velocity=initial_velocity,
            steps=SHOOT_STEPS,
        )
        if mid_record is None:
            high = mid
        elif mid_record["omega_phi"] < OMEGA_PHI0_TARGET:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def omega_at_or_above_redshift(trajectory: pd.DataFrame, z_min: float) -> float:
    early = trajectory.loc[trajectory["z"].ge(z_min), "omega_phi"]
    return float(early.max()) if len(early) else float("nan")


def omega_near_redshift(trajectory: pd.DataFrame, z_value: float) -> float:
    index = int(np.argmin(np.abs(trajectory["z"].to_numpy() - z_value)))
    return float(trajectory.iloc[index]["omega_phi"])


def run_scan_case(
    *,
    variant_name: str,
    harmonics: tuple[tuple[int, float], ...],
    mu_label: str,
    mu: float,
    initial_phase: float,
    initial_mu_xprime: float,
) -> tuple[dict[str, float | str | bool], pd.DataFrame] | None:
    if abs(initial_mu_xprime) >= math.sqrt(6.0):
        return None
    initial_velocity = initial_mu_xprime / mu
    potential_scale = shoot_potential_scale_fast(
        mu=mu,
        harmonics=harmonics,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
    )
    if potential_scale is None:
        return None
    trajectory_records = integrate_background(
        mu=mu,
        potential_scale=potential_scale,
        harmonics=harmonics,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
        steps=FINAL_STEPS,
        keep_trajectory=True,
    )
    if not isinstance(trajectory_records, list):
        return None
    trajectory = pd.DataFrame(trajectory_records)
    targets = project_targets()
    w0, wa_derivative, wa_fit_z0_1 = cpl_from_trajectory(trajectory)
    present = trajectory.iloc[-1]
    max_omega_z_gt_10 = omega_at_or_above_redshift(trajectory, 10.0)
    row = {
        "variant": variant_name,
        "mu_label": mu_label,
        "mu_f_over_mpl": mu,
        "initial_phase": initial_phase,
        "initial_mu_xprime": initial_mu_xprime,
        "potential_scale_lambda4_over_3mpl2h02": potential_scale,
        "w0_integrated": w0,
        "wa_from_present_derivative": wa_derivative,
        "wa_cpl_fit_z0_1": wa_fit_z0_1,
        "omega_phi0": float(present["omega_phi"]),
        "mu_xprime0": float(present["mu_xprime"]),
        "omega_phi_initial": float(trajectory.iloc[0]["omega_phi"]),
        "omega_phi_z100": omega_near_redshift(trajectory, 100.0),
        "omega_phi_z10": omega_near_redshift(trajectory, 10.0),
        "max_omega_phi_z_gt_10": max_omega_z_gt_10,
        "E_z1": sample_e_at_redshift(trajectory, 1.0),
        "w0_abs_error": abs(w0 - targets["w0_target"]),
        "wa_derivative_abs_error": abs(wa_derivative - targets["wa_target"]),
        "passes_w0_window": abs(w0 - targets["w0_target"]) < W0_WINDOW,
        "passes_early_sanity": max_omega_z_gt_10 < EARLY_DARK_ENERGY_SANITY_LIMIT,
    }
    trajectory = trajectory.assign(
        variant=variant_name,
        mu_label=mu_label,
        mu_f_over_mpl=mu,
        initial_phase=initial_phase,
        initial_mu_xprime=initial_mu_xprime,
    )
    return row, trajectory


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(rows).to_markdown(index=False, floatfmt=".6g")


def write_report(results: pd.DataFrame) -> None:
    targets = project_targets()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_w0 = results.sort_values("w0_abs_error")
    early_safe = results.loc[results["passes_early_sanity"]].sort_values("w0_abs_error")
    w0_hits = results.loc[results["passes_w0_window"]].sort_values("max_omega_phi_z_gt_10")
    both_hits = results.loc[results["passes_w0_window"] & results["passes_early_sanity"]]
    lines = [
        "# e-5-137 initial-velocity scalar scan",
        "",
        "Status: canonical scalar stress test with nonzero initial field velocity.",
        "",
        "This scan asks whether the previous thawing failure can be repaired by",
        "starting the scalar with kinetic energy at N = ln(a) = -7, while still",
        "shooting Lambda so Omega_phi0 = 0.69 today.",
        "",
        "## Target and scan gates",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| project w0 target | {targets['w0_target']:.12g} |",
        f"| project wa target | {targets['wa_target']:.12g} |",
        f"| required present mu*x' for project w0 | {targets['present_mu_xprime_required_for_w0']:.12g} |",
        f"| w0 hit window | {W0_WINDOW:g} |",
        f"| early Omega_phi sanity limit for z > 10 | {EARLY_DARK_ENERGY_SANITY_LIMIT:g} |",
        "",
        "The early limit is a project sanity cut, not a quoted external constraint.",
        "It is deliberately simple: cases with percent-level dark-energy fraction",
        "well before low redshift are treated as physically suspect until a full",
        "CMB/BBN analysis is supplied.",
        "",
        "## Best rows by w0",
        "",
        markdown_table(
            best_w0,
            [
                "variant",
                "mu_label",
                "initial_mu_xprime",
                "w0_integrated",
                "wa_from_present_derivative",
                "mu_xprime0",
                "max_omega_phi_z_gt_10",
                "passes_early_sanity",
            ],
        ),
        "",
        "## Best rows that pass early sanity",
        "",
        markdown_table(
            early_safe,
            [
                "variant",
                "mu_label",
                "initial_mu_xprime",
                "w0_integrated",
                "wa_from_present_derivative",
                "mu_xprime0",
                "max_omega_phi_z_gt_10",
                "w0_abs_error",
            ],
        ),
        "",
        "## Rows inside the w0 window",
        "",
        markdown_table(
            w0_hits,
            [
                "variant",
                "mu_label",
                "initial_mu_xprime",
                "w0_integrated",
                "wa_from_present_derivative",
                "mu_xprime0",
                "max_omega_phi_z_gt_10",
                "passes_early_sanity",
            ],
        ),
        "",
        "## Gate result",
        "",
    ]
    if both_hits.empty:
        lines.extend(
            [
                "No scanned canonical-scalar case both reaches the project w0 window",
                "and passes the early-Omega sanity cut.",
                "",
                "The canonical route therefore remains blocked in this first pass:",
                "adding initial kinetic energy can move w0 away from -1, but the",
                "price is early scalar energy that is too large for a clean",
                "dark-energy interpretation without a much more detailed early-universe",
                "model.",
            ]
        )
    else:
        lines.extend(
            [
                "At least one scanned row reaches the w0 window and passes the early",
                "sanity cut. These rows should be rerun with a finer shooting grid and",
                "then compared against CMB/BAO background constraints.",
                "",
                markdown_table(
                    both_hits.sort_values("w0_abs_error"),
                    [
                        "variant",
                        "mu_label",
                        "initial_mu_xprime",
                        "w0_integrated",
                        "wa_from_present_derivative",
                        "max_omega_phi_z_gt_10",
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Next mechanism choice",
            "",
            "If this gate stays closed under a finer scan, the next model class should",
            "not be another cosine-only canonical scalar. The natural next candidates",
            "are k-essence, a coupled two-field dark sector, or moving the 137",
            "structure into a massive spectator/vector sector rather than directly",
            "into the quintessence potential.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    rows: list[dict[str, float | str | bool]] = []
    trajectories: list[pd.DataFrame] = []
    for variant in variants(targets):
        for mu_label, mu in (
            ("project_slope", variant.mu_project_slope),
            ("soft_curvature", variant.mu_soft_curvature),
        ):
            for initial_mu_xprime in SCAN_INITIAL_MU_XPRIME:
                result = run_scan_case(
                    variant_name=variant.name,
                    harmonics=variant.harmonics,
                    mu_label=mu_label,
                    mu=mu,
                    initial_phase=variant.steepest_phase,
                    initial_mu_xprime=float(initial_mu_xprime),
                )
                if result is None:
                    continue
                row, trajectory = result
                rows.append(row)
                if len(trajectories) < 8:
                    trajectories.append(trajectory.iloc[::20].copy())

    results = pd.DataFrame(rows)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_PATH, index=False)
    if not results.empty:
        best_keys = set(results.sort_values("w0_abs_error").head(4).index)
        best_keys.update(results.loc[results["passes_early_sanity"]].sort_values("w0_abs_error").head(4).index)
        best_trajectories: list[pd.DataFrame] = []
        # Rerun compact trajectories for the selected rows with stable labels.
        for index in sorted(best_keys):
            row = results.loc[index]
            variant = next(v for v in variants(targets) if v.name == row["variant"])
            rerun = run_scan_case(
                variant_name=variant.name,
                harmonics=variant.harmonics,
                mu_label=str(row["mu_label"]),
                mu=float(row["mu_f_over_mpl"]),
                initial_phase=float(row["initial_phase"]),
                initial_mu_xprime=float(row["initial_mu_xprime"]),
            )
            if rerun is not None:
                _, trajectory = rerun
                best_trajectories.append(trajectory.iloc[::20].copy())
        if best_trajectories:
            pd.concat(best_trajectories, ignore_index=True).to_csv(BEST_TRAJECTORY_PATH, index=False)
    write_report(results)
    print(REPORT_PATH)
    print(TABLE_PATH)
    print(BEST_TRAJECTORY_PATH)
    # Keep the previous trajectory path mentioned by the base integrator visibly separate.
    print(TRAJECTORY_PATH)


if __name__ == "__main__":
    main()
