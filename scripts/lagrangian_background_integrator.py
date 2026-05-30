from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cosmo_gradient.theory import (
    D_STRING,
    N_TOPOLOGICAL,
    delta_phi,
    schwinger_s,
    vacuum_compression_operator,
)


OMEGA_M0 = 0.31
OMEGA_R0 = 9.0e-5
OMEGA_PHI0_TARGET = 0.69
N_INITIAL = -7.0
N_FINAL = 0.0
SHOOT_STEPS = 1_200
FINAL_STEPS = 3_000
REPORT_PATH = Path("outputs/reports/e5_137_lagrangian_background_integration.md")
TABLE_PATH = Path("outputs/tables/e5_137_lagrangian_background_integration.csv")
TRAJECTORY_PATH = Path("outputs/tables/e5_137_lagrangian_background_trajectories.csv")


@dataclass(frozen=True)
class Variant:
    name: str
    harmonics: tuple[tuple[int, float], ...]
    steepest_phase: float
    mu_project_slope: float
    mu_soft_curvature: float
    note: str


def project_targets() -> dict[str, float]:
    n = N_TOPOLOGICAL
    d = D_STRING
    q = vacuum_compression_operator(n)
    delta = delta_phi(n)
    s = schwinger_s()
    t5 = 2.0 / (math.e * n)
    w0 = -1.0 + t5 / n + d * q - delta / n
    wa = -n * delta * (math.pi + s - d * q + q / math.pi)
    kinetic_fraction_of_phi = 0.5 * (1.0 + w0)
    kinetic_fraction_of_critical = OMEGA_PHI0_TARGET * kinetic_fraction_of_phi
    present_mu_xprime_required = math.sqrt(6.0 * kinetic_fraction_of_critical)
    return {
        "q": q,
        "delta_phi": delta,
        "s": s,
        "T5": t5,
        "w0_target": w0,
        "wa_target": wa,
        "kinetic_fraction_of_phi_for_w0": kinetic_fraction_of_phi,
        "kinetic_fraction_of_critical_for_w0": kinetic_fraction_of_critical,
        "present_mu_xprime_required_for_w0": present_mu_xprime_required,
    }


def variants(targets: dict[str, float]) -> tuple[Variant, ...]:
    q = targets["q"]
    delta = targets["delta_phi"]
    n = N_TOPOLOGICAL
    return (
        Variant(
            name="q_delta",
            harmonics=((1, q), (n, delta)),
            steepest_phase=1.5833124319,
            mu_project_slope=0.2932072370,
            mu_soft_curvature=1.81536,
            note="Low-frequency q and delta_phi harmonics.",
        ),
        Variant(
            name="q_delta_137_1over137",
            harmonics=((1, q), (n, delta), (137, 1.0 / 137.0)),
            steepest_phase=2.9008775413,
            mu_project_slope=1.2680431,
            mu_soft_curvature=17.6057,
            note="Literal high-frequency 137 harmonic.",
        ),
        Variant(
            name="q_delta_137_1over1372",
            harmonics=((1, q), (n, delta), (137, 1.0 / 137.0**2)),
            steepest_phase=1.5714686276,
            mu_project_slope=0.299710177,
            mu_soft_curvature=2.34874,
            note="Curvature-tamed 137 harmonic.",
        ),
    )


def potential_and_derivative(x: float, harmonics: tuple[tuple[int, float], ...]) -> tuple[float, float]:
    u = 1.0
    ux = 0.0
    for k, amplitude in harmonics:
        u += amplitude * math.cos(k * x)
        ux += -amplitude * k * math.sin(k * x)
    return u, ux


def rhs(
    n_efolds: float,
    state: np.ndarray,
    *,
    mu: float,
    potential_scale: float,
    harmonics: tuple[tuple[int, float], ...],
) -> np.ndarray | None:
    x, velocity = float(state[0]), float(state[1])
    u, ux = potential_and_derivative(x, harmonics)
    denominator = 1.0 - mu * mu * velocity * velocity / 6.0
    if denominator <= 0.0 or u <= 0.0:
        return None
    matter = OMEGA_M0 * math.exp(-3.0 * n_efolds)
    radiation = OMEGA_R0 * math.exp(-4.0 * n_efolds)
    e2 = (matter + radiation + potential_scale * u) / denominator
    if not math.isfinite(e2) or e2 <= 0.0:
        return None
    dlnh_dn = -1.5 * matter / e2 - 2.0 * radiation / e2 - 0.5 * mu * mu * velocity * velocity
    acceleration = (
        -(3.0 + dlnh_dn) * velocity
        - 3.0 * potential_scale * ux / (mu * mu * e2)
    )
    return np.array([velocity, acceleration], dtype=float)


def background_record(
    n_efolds: float,
    state: np.ndarray,
    *,
    mu: float,
    potential_scale: float,
    harmonics: tuple[tuple[int, float], ...],
) -> dict[str, float] | None:
    x, velocity = float(state[0]), float(state[1])
    u, _ = potential_and_derivative(x, harmonics)
    denominator = 1.0 - mu * mu * velocity * velocity / 6.0
    if denominator <= 0.0 or u <= 0.0:
        return None
    matter = OMEGA_M0 * math.exp(-3.0 * n_efolds)
    radiation = OMEGA_R0 * math.exp(-4.0 * n_efolds)
    e2 = (matter + radiation + potential_scale * u) / denominator
    if not math.isfinite(e2) or e2 <= 0.0:
        return None
    kinetic = e2 * mu * mu * velocity * velocity / 6.0
    potential = potential_scale * u
    rho_phi = kinetic + potential
    w = (kinetic - potential) / rho_phi
    return {
        "N": n_efolds,
        "a": math.exp(n_efolds),
        "z": math.exp(-n_efolds) - 1.0,
        "x_phi_over_f": x,
        "xprime": velocity,
        "mu_xprime": mu * velocity,
        "E": math.sqrt(e2),
        "w_phi": w,
        "omega_phi": rho_phi / e2,
        "kinetic_crit0": kinetic,
        "potential_crit0": potential,
    }


def integrate_background(
    *,
    mu: float,
    potential_scale: float,
    harmonics: tuple[tuple[int, float], ...],
    initial_phase: float,
    initial_velocity: float = 0.0,
    steps: int = SHOOT_STEPS,
    keep_trajectory: bool = False,
) -> list[dict[str, float]] | dict[str, float] | None:
    state = np.array([initial_phase, initial_velocity], dtype=float)
    grid = np.linspace(N_INITIAL, N_FINAL, steps + 1)
    records: list[dict[str, float]] = []
    for index, n_efolds in enumerate(grid):
        record = background_record(
            n_efolds,
            state,
            mu=mu,
            potential_scale=potential_scale,
            harmonics=harmonics,
        )
        if record is None:
            return None
        if keep_trajectory:
            records.append(record)
        if index == len(grid) - 1:
            return records if keep_trajectory else record
        step = grid[index + 1] - n_efolds
        k1 = rhs(n_efolds, state, mu=mu, potential_scale=potential_scale, harmonics=harmonics)
        if k1 is None:
            return None
        k2 = rhs(
            n_efolds + 0.5 * step,
            state + 0.5 * step * k1,
            mu=mu,
            potential_scale=potential_scale,
            harmonics=harmonics,
        )
        if k2 is None:
            return None
        k3 = rhs(
            n_efolds + 0.5 * step,
            state + 0.5 * step * k2,
            mu=mu,
            potential_scale=potential_scale,
            harmonics=harmonics,
        )
        if k3 is None:
            return None
        k4 = rhs(
            n_efolds + step,
            state + step * k3,
            mu=mu,
            potential_scale=potential_scale,
            harmonics=harmonics,
        )
        if k4 is None:
            return None
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return None


def shoot_potential_scale(
    *,
    mu: float,
    harmonics: tuple[tuple[int, float], ...],
    initial_phase: float,
    initial_velocity: float = 0.0,
) -> float | None:
    low = 1.0e-10
    high = 5.0
    high_record = integrate_background(
        mu=mu,
        potential_scale=high,
        harmonics=harmonics,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
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
        )
    if high_record is None:
        return None
    for _ in range(36):
        mid = 0.5 * (low + high)
        mid_record = integrate_background(
            mu=mu,
            potential_scale=mid,
            harmonics=harmonics,
            initial_phase=initial_phase,
            initial_velocity=initial_velocity,
        )
        if mid_record is None:
            high = mid
        elif mid_record["omega_phi"] < OMEGA_PHI0_TARGET:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def cpl_from_trajectory(trajectory: pd.DataFrame) -> tuple[float, float, float]:
    present = trajectory.iloc[-1]
    tail = trajectory.tail(100)
    derivative = float(np.polyfit(tail["N"], tail["w_phi"], 1)[0])
    low_z = trajectory.loc[trajectory["z"].le(1.0)]
    design = np.column_stack([np.ones(len(low_z)), 1.0 - low_z["a"].to_numpy()])
    w0_fit, wa_fit = np.linalg.lstsq(design, low_z["w_phi"].to_numpy(), rcond=None)[0]
    return float(present["w_phi"]), float(-derivative), float(wa_fit)


def sample_e_at_redshift(trajectory: pd.DataFrame, z_value: float) -> float:
    index = int(np.argmin(np.abs(trajectory["z"].to_numpy() - z_value)))
    return float(trajectory.iloc[index]["E"])


def run_case(
    *,
    variant: Variant,
    mu_label: str,
    mu: float,
    phase_label: str,
    initial_phase: float,
    initial_velocity: float = 0.0,
) -> tuple[dict[str, float | str], pd.DataFrame] | None:
    potential_scale = shoot_potential_scale(
        mu=mu,
        harmonics=variant.harmonics,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
    )
    if potential_scale is None:
        return None
    trajectory_records = integrate_background(
        mu=mu,
        potential_scale=potential_scale,
        harmonics=variant.harmonics,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
        steps=FINAL_STEPS,
        keep_trajectory=True,
    )
    if not isinstance(trajectory_records, list):
        return None
    trajectory = pd.DataFrame(trajectory_records)
    w0, wa_derivative, wa_fit_z0_1 = cpl_from_trajectory(trajectory)
    present = trajectory.iloc[-1]
    row = {
        "variant": variant.name,
        "mu_label": mu_label,
        "mu_f_over_mpl": mu,
        "phase_label": phase_label,
        "initial_phase": initial_phase,
        "initial_velocity": initial_velocity,
        "initial_mu_xprime": mu * initial_velocity,
        "potential_scale_lambda4_over_3mpl2h02": potential_scale,
        "w0_integrated": w0,
        "wa_from_present_derivative": wa_derivative,
        "wa_cpl_fit_z0_1": wa_fit_z0_1,
        "omega_phi0": float(present["omega_phi"]),
        "mu_xprime0": float(present["mu_xprime"]),
        "kinetic_crit0": float(present["kinetic_crit0"]),
        "potential_crit0": float(present["potential_crit0"]),
        "E_z0p5": sample_e_at_redshift(trajectory, 0.5),
        "E_z1": sample_e_at_redshift(trajectory, 1.0),
        "E_z2": sample_e_at_redshift(trajectory, 2.0),
        "note": variant.note,
    }
    trajectory = trajectory.assign(
        variant=variant.name,
        mu_label=mu_label,
        mu_f_over_mpl=mu,
        phase_label=phase_label,
        initial_phase=initial_phase,
        initial_velocity=initial_velocity,
    )
    return row, trajectory


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".6g")


def write_report(
    targets: dict[str, float],
    results: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# e-5-137 scalar-background integration",
        "",
        "Status: canonical scalar-field integration for the trial dark-sector",
        "Lagrangian. This is still a toy effective model, not a completed theory.",
        "",
        "## Integrated model",
        "",
        "The integrated action is",
        "",
        "```text",
        "L = (M_Pl^2 / 2) R - 1/2 (partial phi)^2 - Lambda^4 U(phi/f)",
        "U(x) = 1 + sum_i a_i cos(k_i x),  x = phi/f,  mu = f/M_Pl",
        "```",
        "",
        "The numerical integration uses N = ln(a), starts at N = -7, and assumes",
        "a thawing initial condition x'(N_initial) = 0. For each case, Lambda is",
        "shot to give Omega_phi0 = 0.69 today.",
        "",
        "The equations are",
        "",
        "```text",
        "E^2 = [Omega_m0 exp(-3N) + Omega_r0 exp(-4N) + A U(x)]",
        "      / [1 - mu^2 x'^2 / 6]",
        "",
        "x'' + [3 + d ln H / dN] x' + 3 A U_x / [mu^2 E^2] = 0",
        "",
        "d ln H / dN = -3 Omega_m/2 - 2 Omega_r - mu^2 x'^2/2",
        "```",
        "",
        "with A = Lambda^4 / (3 M_Pl^2 H0^2).",
        "",
        "## Project target",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| project w0 target | {targets['w0_target']:.12g} |",
        f"| project wa target | {targets['wa_target']:.12g} |",
        f"| kinetic fraction of phi needed for that w0 | {targets['kinetic_fraction_of_phi_for_w0']:.12g} |",
        f"| kinetic fraction of critical density needed today | {targets['kinetic_fraction_of_critical_for_w0']:.12g} |",
        f"| required present mu*x' | {targets['present_mu_xprime_required_for_w0']:.12g} |",
        "",
        "## Integration summary",
        "",
        markdown_table(
            results,
            [
                "variant",
                "mu_label",
                "phase_label",
                "mu_f_over_mpl",
                "w0_integrated",
                "wa_from_present_derivative",
                "wa_cpl_fit_z0_1",
                "mu_xprime0",
                "E_z1",
            ],
        ),
        "",
        "## First integrated readout",
        "",
        "- The thawing canonical scalar does not reproduce the project-level",
        "  w0 ~= -0.752 from rest. The integrated cases remain close to",
        "  w0 = -1, typically between -1 and about -0.995.",
        "- This is not a numerical accident. The target w0 requires about",
        "  12.4% of the scalar energy to be kinetic today, i.e. a present",
        "  rolling speed mu*x' ~= 0.716. The thawing runs produce much smaller",
        "  present rolling speeds.",
        "- The local slope audit was therefore necessary but insufficient:",
        "  a potential can have enough local slope, yet the cosmological",
        "  trajectory may never reach a fast-rolling state after Hubble friction.",
        "- The direct 137 harmonic is still problematic as a dark-energy harmonic:",
        "  in the integrated thawing setup it also freezes close to -1, while",
        "  the local audit showed that its curvature is expensive unless f is",
        "  very large or the harmonic is strongly suppressed.",
        "",
        "## Gate",
        "",
        "Under canonical thawing initial conditions, this scalar Lagrangian fails",
        "to produce the project's large w0 and wa targets. The next physically",
        "honest options are: add a nonzero kinetic initial condition and audit",
        "early-dark-energy constraints; change to a different dark-sector",
        "mechanism such as k-essence/coupled fields; or reinterpret the project",
        "w0-wa formula as a phenomenological number that is not generated by this",
        "minimal canonical scalar.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    rows: list[dict[str, float | str]] = []
    trajectories: list[pd.DataFrame] = []
    for variant in variants(targets):
        cases = [
            ("project_slope", variant.mu_project_slope, "steepest", variant.steepest_phase),
            ("soft_curvature", variant.mu_soft_curvature, "steepest", variant.steepest_phase),
            ("project_slope", variant.mu_project_slope, "stationary_x0", 0.0),
        ]
        for mu_label, mu, phase_label, phase in cases:
            result = run_case(
                variant=variant,
                mu_label=mu_label,
                mu=mu,
                phase_label=phase_label,
                initial_phase=phase,
            )
            if result is None:
                continue
            row, trajectory = result
            rows.append(row)
            # Keep a compact trajectory grid for the nontrivial steepest cases.
            if phase_label == "steepest":
                trajectories.append(trajectory.iloc[::30].copy())

    results = pd.DataFrame(rows)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_PATH, index=False)
    if trajectories:
        pd.concat(trajectories, ignore_index=True).to_csv(TRAJECTORY_PATH, index=False)
    write_report(targets, results)
    print(REPORT_PATH)
    print(TABLE_PATH)
    print(TRAJECTORY_PATH)


if __name__ == "__main__":
    main()
