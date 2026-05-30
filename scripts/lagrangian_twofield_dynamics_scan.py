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
OMEGA_DARK0_TARGET = 0.69
N_INITIAL = -7.0
N_FINAL = 0.0
SHOOT_STEPS = 360
FINAL_STEPS = 1_200
N_SCAN_CASES = 320
REPORT_PATH = Path("outputs/reports/e5_137_twofield_dynamics_scan.md")
SCAN_TABLE_PATH = Path("outputs/tables/e5_137_twofield_dynamics_scan.csv")
BEST_TRAJECTORY_PATH = Path("outputs/tables/e5_137_twofield_best_trajectories.csv")


@dataclass(frozen=True)
class TwoFieldParams:
    m_phi_sq: float
    m_chi_sq: float
    g_sq: float
    x0: float
    y0: float
    vx0: float
    vy0: float


def project_targets() -> dict[str, float]:
    n = N_TOPOLOGICAL
    d = D_STRING
    q = vacuum_compression_operator(n)
    delta = delta_phi(n)
    s = schwinger_s()
    t5 = 2.0 / (math.e * n)
    w0 = -1.0 + t5 / n + d * q - delta / n
    wa = -n * delta * (math.pi + s - d * q + q / math.pi)
    crossing_fraction = (-1.0 - w0) / wa
    crossing_a = 1.0 - crossing_fraction
    crossing_z = 1.0 / crossing_a - 1.0
    return {
        "q": q,
        "delta_phi": delta,
        "s": s,
        "T5": t5,
        "w0": w0,
        "wa": wa,
        "w_inf": w0 + wa,
        "phantom_crossing_z": crossing_z,
    }


def target_w_of_z(z: np.ndarray | float, targets: dict[str, float]) -> np.ndarray | float:
    return targets["w0"] + targets["wa"] * np.asarray(z) / (1.0 + np.asarray(z))


def potential_terms(x: float, y: float, params: TwoFieldParams) -> tuple[float, float, float]:
    u = (
        1.0
        + 0.5 * params.m_phi_sq * x * x
        + 0.5 * params.m_chi_sq * y * y
        + 0.5 * params.g_sq * x * x * y * y
    )
    ux = params.m_phi_sq * x + params.g_sq * x * y * y
    uy = params.m_chi_sq * y + params.g_sq * x * x * y
    return u, ux, uy


def background_record(
    n_efolds: float,
    state: np.ndarray,
    *,
    potential_scale: float,
    params: TwoFieldParams,
) -> dict[str, float] | None:
    x, y, vx, vy = (float(value) for value in state)
    u, _, _ = potential_terms(x, y, params)
    kinetic_signed = vx * vx - vy * vy
    denominator = 1.0 - kinetic_signed / 6.0
    if denominator <= 0.0 or u <= 0.0:
        return None
    matter = OMEGA_M0 * math.exp(-3.0 * n_efolds)
    radiation = OMEGA_R0 * math.exp(-4.0 * n_efolds)
    e2 = (matter + radiation + potential_scale * u) / denominator
    if not math.isfinite(e2) or e2 <= 0.0:
        return None
    kinetic_crit = e2 * kinetic_signed / 6.0
    potential_crit = potential_scale * u
    rho_dark = kinetic_crit + potential_crit
    if rho_dark <= 0.0:
        return None
    pressure_dark = kinetic_crit - potential_crit
    return {
        "N": n_efolds,
        "a": math.exp(n_efolds),
        "z": math.exp(-n_efolds) - 1.0,
        "x": x,
        "y": y,
        "xprime": vx,
        "yprime": vy,
        "E": math.sqrt(e2),
        "omega_dark": rho_dark / e2,
        "w_dark": pressure_dark / rho_dark,
        "kinetic_signed_crit": kinetic_crit,
        "potential_crit": potential_crit,
        "phantom_kinetic_ratio": (vy * vy / max(vx * vx + vy * vy, 1.0e-15)),
    }


def rhs(
    n_efolds: float,
    state: np.ndarray,
    *,
    potential_scale: float,
    params: TwoFieldParams,
) -> np.ndarray | None:
    record = background_record(
        n_efolds,
        state,
        potential_scale=potential_scale,
        params=params,
    )
    if record is None:
        return None
    x, y, vx, vy = (float(value) for value in state)
    u, ux, uy = potential_terms(x, y, params)
    e2 = record["E"] ** 2
    matter = OMEGA_M0 * math.exp(-3.0 * n_efolds)
    radiation = OMEGA_R0 * math.exp(-4.0 * n_efolds)
    dlnh_dn = -1.5 * matter / e2 - 2.0 * radiation / e2 - 0.5 * (vx * vx - vy * vy)
    ax = -(3.0 + dlnh_dn) * vx - 3.0 * potential_scale * ux / e2
    ay = -(3.0 + dlnh_dn) * vy + 3.0 * potential_scale * uy / e2
    return np.array([vx, vy, ax, ay], dtype=float)


def integrate(
    *,
    potential_scale: float,
    params: TwoFieldParams,
    steps: int,
    keep_trajectory: bool = False,
) -> list[dict[str, float]] | dict[str, float] | None:
    state = np.array([params.x0, params.y0, params.vx0, params.vy0], dtype=float)
    grid = np.linspace(N_INITIAL, N_FINAL, steps + 1)
    trajectory: list[dict[str, float]] = []
    for index, n_efolds in enumerate(grid):
        record = background_record(
            n_efolds,
            state,
            potential_scale=potential_scale,
            params=params,
        )
        if record is None:
            return None
        if keep_trajectory:
            trajectory.append(record)
        if index == len(grid) - 1:
            return trajectory if keep_trajectory else record
        step = grid[index + 1] - n_efolds
        k1 = rhs(n_efolds, state, potential_scale=potential_scale, params=params)
        if k1 is None:
            return None
        k2 = rhs(
            n_efolds + 0.5 * step,
            state + 0.5 * step * k1,
            potential_scale=potential_scale,
            params=params,
        )
        if k2 is None:
            return None
        k3 = rhs(
            n_efolds + 0.5 * step,
            state + 0.5 * step * k2,
            potential_scale=potential_scale,
            params=params,
        )
        if k3 is None:
            return None
        k4 = rhs(
            n_efolds + step,
            state + step * k3,
            potential_scale=potential_scale,
            params=params,
        )
        if k4 is None:
            return None
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return None


def shoot_potential_scale(params: TwoFieldParams) -> float | None:
    low = 1.0e-10
    high = 4.0
    high_record = integrate(potential_scale=high, params=params, steps=SHOOT_STEPS)
    while (
        high_record is not None
        and high_record["omega_dark"] < OMEGA_DARK0_TARGET
        and high < 1.0e4
    ):
        high *= 2.0
        high_record = integrate(potential_scale=high, params=params, steps=SHOOT_STEPS)
    if high_record is None:
        return None
    for _ in range(28):
        mid = 0.5 * (low + high)
        record = integrate(potential_scale=mid, params=params, steps=SHOOT_STEPS)
        if record is None:
            high = mid
        elif record["omega_dark"] < OMEGA_DARK0_TARGET:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def sample_at_redshift(trajectory: pd.DataFrame, z_value: float, column: str) -> float:
    index = int(np.argmin(np.abs(trajectory["z"].to_numpy() - z_value)))
    return float(trajectory.iloc[index][column])


def trajectory_score(trajectory: pd.DataFrame, targets: dict[str, float]) -> dict[str, float | bool]:
    z_eval = np.array([0.0, 0.5, 1.0, 2.0])
    w_model = np.array([sample_at_redshift(trajectory, z, "w_dark") for z in z_eval])
    w_target = target_w_of_z(z_eval, targets)
    w_rms = float(np.sqrt(np.mean((w_model - w_target) ** 2)))
    w0 = float(w_model[0])
    z10_max = float(trajectory.loc[trajectory["z"].ge(10.0), "omega_dark"].max())
    z100 = sample_at_redshift(trajectory, 100.0, "omega_dark")
    crosses = bool(np.nanmin(w_model) < -1.0 < np.nanmax(w_model))
    return {
        "w0": w0,
        "w_z0p5": float(w_model[1]),
        "w_z1": float(w_model[2]),
        "w_z2": float(w_model[3]),
        "w_rms_z0_2": w_rms,
        "w0_abs_error": abs(w0 - targets["w0"]),
        "omega_dark_max_z_gt_10": z10_max,
        "omega_dark_z100": z100,
        "phantom_crosses_w_grid": crosses,
        "early_sanity": z10_max < 0.01,
    }


def random_params(rng: np.random.Generator) -> TwoFieldParams:
    m_phi_sq = float(10.0 ** rng.uniform(-1.4, 0.6))
    m_chi_sq = float(10.0 ** rng.uniform(-1.4, 0.6))
    g_sq = float(10.0 ** rng.uniform(-2.0, 0.8))
    x0 = float(rng.uniform(-1.5, 1.5))
    y0 = float(rng.uniform(-1.5, 1.5))
    vx0 = float(rng.uniform(-2.0, 2.0))
    vy0 = float(rng.uniform(-2.0, 2.0))
    return TwoFieldParams(
        m_phi_sq=m_phi_sq,
        m_chi_sq=m_chi_sq,
        g_sq=g_sq,
        x0=x0,
        y0=y0,
        vx0=vx0,
        vy0=vy0,
    )


def structured_params() -> list[TwoFieldParams]:
    rows: list[TwoFieldParams] = []
    for m_phi_sq in [0.05, 0.2, 1.0]:
        for m_chi_sq in [0.05, 0.2, 1.0]:
            for g_sq in [0.05, 0.5, 2.0]:
                rows.append(
                    TwoFieldParams(
                        m_phi_sq=m_phi_sq,
                        m_chi_sq=m_chi_sq,
                        g_sq=g_sq,
                        x0=0.7,
                        y0=0.7,
                        vx0=0.8,
                        vy0=1.0,
                    )
                )
                rows.append(
                    TwoFieldParams(
                        m_phi_sq=m_phi_sq,
                        m_chi_sq=m_chi_sq,
                        g_sq=g_sq,
                        x0=0.7,
                        y0=-0.7,
                        vx0=1.0,
                        vy0=-1.0,
                    )
                )
    return rows


def run_case(case_id: int, params: TwoFieldParams, targets: dict[str, float]) -> tuple[dict[str, object], pd.DataFrame] | None:
    # Keep the initial effective kinetic away from a singular Friedmann denominator.
    if params.vx0 * params.vx0 - params.vy0 * params.vy0 >= 5.5:
        return None
    potential_scale = shoot_potential_scale(params)
    if potential_scale is None:
        return None
    trajectory_records = integrate(
        potential_scale=potential_scale,
        params=params,
        steps=FINAL_STEPS,
        keep_trajectory=True,
    )
    if not isinstance(trajectory_records, list):
        return None
    trajectory = pd.DataFrame(trajectory_records)
    score = trajectory_score(trajectory, targets)
    row: dict[str, object] = {
        "case_id": case_id,
        "potential_scale_A": potential_scale,
        "m_phi_sq": params.m_phi_sq,
        "m_chi_sq": params.m_chi_sq,
        "g_sq": params.g_sq,
        "x0": params.x0,
        "y0": params.y0,
        "vx0": params.vx0,
        "vy0": params.vy0,
        **score,
    }
    trajectory = trajectory.assign(case_id=case_id)
    return row, trajectory


def scan(targets: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260530)
    params_list = structured_params()
    params_list.extend(random_params(rng) for _ in range(N_SCAN_CASES))
    rows: list[dict[str, object]] = []
    trajectories_by_case: dict[int, pd.DataFrame] = {}
    for case_id, params in enumerate(params_list):
        result = run_case(case_id, params, targets)
        if result is None:
            continue
        row, trajectory = result
        rows.append(row)
        trajectories_by_case[case_id] = trajectory
    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame()
    results = results.sort_values(["w_rms_z0_2", "omega_dark_max_z_gt_10"]).reset_index(drop=True)
    best_ids = set(results.head(5)["case_id"].astype(int))
    safe = results.loc[results["early_sanity"]].sort_values("w_rms_z0_2").head(5)
    best_ids.update(safe["case_id"].astype(int))
    best_trajectories = [
        trajectories_by_case[case_id].iloc[::10].copy()
        for case_id in sorted(best_ids)
        if case_id in trajectories_by_case
    ]
    trajectory_table = pd.concat(best_trajectories, ignore_index=True) if best_trajectories else pd.DataFrame()
    return results, trajectory_table


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(rows).to_markdown(index=False, floatfmt=".6g")


def write_report(targets: dict[str, float], results: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = results.sort_values("w_rms_z0_2") if not results.empty else results
    safe = results.loc[results["early_sanity"]].sort_values("w_rms_z0_2") if not results.empty else results
    crossing = results.loc[results["phantom_crosses_w_grid"]].sort_values("w_rms_z0_2") if not results.empty else results
    both = results.loc[results["early_sanity"] & results["phantom_crosses_w_grid"]].sort_values("w_rms_z0_2") if not results.empty else results
    lines = [
        "# e-5-137 explicit two-field dynamics scan",
        "",
        "Status: coarse FRW integration of an effective canonical+phantom two-field",
        "sector. The phantom field is treated as an EFT component with a cutoff, not",
        "as a UV-complete healthy scalar.",
        "",
        "## Model",
        "",
        "```text",
        "L_dark = -1/2 (partial phi)^2 + 1/2 (partial chi)^2 - V(phi, chi)",
        "",
        "V = Lambda^4 U(x, y)",
        "U = 1 + m_phi^2 x^2/2 + m_chi^2 y^2/2 + g^2 x^2 y^2/2",
        "x = phi/M_Pl, y = chi/M_Pl",
        "```",
        "",
        "The background equations are integrated from `N = ln a = -7` to today.",
        "For each case, `Lambda` is shot so that `Omega_dark0 = 0.69`.",
        "",
        "## Target",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| w0 | {targets['w0']:.12g} |",
        f"| wa | {targets['wa']:.12g} |",
        f"| w(z=0.5) | {float(target_w_of_z(0.5, targets)):.12g} |",
        f"| w(z=1) | {float(target_w_of_z(1.0, targets)):.12g} |",
        f"| phantom crossing redshift | {targets['phantom_crossing_z']:.12g} |",
        "",
        "## Best rows by w(z) RMS over z = 0, 0.5, 1, 2",
        "",
        markdown_table(
            best,
            [
                "case_id",
                "w_rms_z0_2",
                "w0",
                "w_z0p5",
                "w_z1",
                "w_z2",
                "omega_dark_max_z_gt_10",
                "phantom_crosses_w_grid",
                "m_phi_sq",
                "m_chi_sq",
                "g_sq",
            ],
        ),
        "",
        "## Best rows passing early sanity",
        "",
        markdown_table(
            safe,
            [
                "case_id",
                "w_rms_z0_2",
                "w0",
                "w_z0p5",
                "w_z1",
                "w_z2",
                "omega_dark_max_z_gt_10",
                "phantom_crosses_w_grid",
                "m_phi_sq",
                "m_chi_sq",
                "g_sq",
            ],
        ),
        "",
        "## Best rows with a sampled phantom crossing",
        "",
        markdown_table(
            crossing,
            [
                "case_id",
                "w_rms_z0_2",
                "w0",
                "w_z0p5",
                "w_z1",
                "w_z2",
                "omega_dark_max_z_gt_10",
                "early_sanity",
                "m_phi_sq",
                "m_chi_sq",
                "g_sq",
            ],
        ),
        "",
        "## Rows that pass both early sanity and crossing",
        "",
        markdown_table(
            both,
            [
                "case_id",
                "w_rms_z0_2",
                "w0",
                "w_z0p5",
                "w_z1",
                "w_z2",
                "omega_dark_max_z_gt_10",
                "m_phi_sq",
                "m_chi_sq",
                "g_sq",
            ],
        ),
        "",
        "## Gate result",
        "",
    ]
    if both.empty:
        lines.extend(
            [
                "No coarse-scan row simultaneously passes the early-dark-energy sanity cut",
                "and samples the desired phantom crossing.",
                "",
                "The explicit `g^2 phi^2 chi^2` two-field potential is therefore not yet",
                "a successful physical mechanism for the project CPL target. It can move",
                "the system around `w=-1`, but in the scanned region it either misses the",
                "shape or carries too much early dark-sector energy.",
            ]
        )
    else:
        lines.extend(
            [
                "At least one row passes both coarse gates. These cases require a finer",
                "local optimizer and perturbation-stability audit before any physical",
                "claim.",
            ]
        )
    lines.extend(
        [
            "",
            "Next useful refinement: make the interaction explicitly e-5-137-shaped,",
            "for example `g^2 -> g0^2 [1 + q cos(5x) + delta_phi cos(137 y)/137^2]`,",
            "and run a local optimizer around the best coarse rows.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    results, trajectories = scan(targets)
    SCAN_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(SCAN_TABLE_PATH, index=False)
    trajectories.to_csv(BEST_TRAJECTORY_PATH, index=False)
    write_report(targets, results)
    print(REPORT_PATH)
    print(SCAN_TABLE_PATH)
    print(BEST_TRAJECTORY_PATH)


if __name__ == "__main__":
    main()
