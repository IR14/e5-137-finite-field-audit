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
SHOOT_STEPS = 180
FINAL_STEPS = 640
LOCAL_DRAWS_PER_SEED = 6
RNG_SEED = 20260531
CPL_GATE_Z_MAX = 2.0

SEED_TABLE_PATH = Path("outputs/tables/e5_137_twofield_dynamics_scan.csv")
REPORT_PATH = Path("outputs/reports/e5_137_interaction_optimizer.md")
SCAN_TABLE_PATH = Path("outputs/tables/e5_137_interaction_optimizer.csv")
BEST_TRAJECTORY_PATH = Path("outputs/tables/e5_137_interaction_optimizer_best_trajectories.csv")


@dataclass(frozen=True)
class E5137Params:
    m_phi_sq: float
    m_chi_sq: float
    g0_sq: float
    x0: float
    y0: float
    vx0: float
    vy0: float
    alpha_5: float
    alpha_137: float
    mode: str


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
    z_array = np.asarray(z)
    return targets["w0"] + targets["wa"] * z_array / (1.0 + z_array)


def modulation_terms(x: float, y: float, params: E5137Params, targets: dict[str, float]) -> tuple[float, float, float]:
    q = targets["q"]
    delta = targets["delta_phi"]
    n = N_TOPOLOGICAL
    modulation = (
        1.0
        + params.alpha_5 * q * math.cos(5.0 * x)
        + params.alpha_137 * delta * math.cos(n * y) / (n * n)
    )
    dmod_dx = -5.0 * params.alpha_5 * q * math.sin(5.0 * x)
    dmod_dy = -params.alpha_137 * delta * math.sin(n * y) / n
    return modulation, dmod_dx, dmod_dy


def horizon_time_gate(
    z: float,
    x: float,
    params: E5137Params,
    targets: dict[str, float],
) -> tuple[float, float, bool] | None:
    if params.mode == "baseline_no_gate" or z > CPL_GATE_Z_MAX:
        return 1.0, 1.0, False
    phase = math.cos(5.0 * x)
    argument = targets["q"] * math.exp(z) * phase * phase / 117.0
    denominator = 1.0 - argument
    if denominator <= 0.0:
        return None
    return 1.0 / math.sqrt(denominator), denominator, True


def potential_terms(
    x: float,
    y: float,
    params: E5137Params,
    targets: dict[str, float],
) -> tuple[float, float, float, float]:
    x2 = x * x
    y2 = y * y
    modulation = 1.0
    interaction = 0.5 * params.g0_sq * x2 * y2
    u = 1.0 + 0.5 * params.m_phi_sq * x2 + 0.5 * params.m_chi_sq * y2 + interaction
    ux = params.m_phi_sq * x + params.g0_sq * x * y2
    uy = params.m_chi_sq * y + params.g0_sq * x2 * y
    return u, ux, uy, modulation


def background_record(
    n_efolds: float,
    state: np.ndarray,
    *,
    potential_scale: float,
    params: E5137Params,
    targets: dict[str, float],
) -> dict[str, float] | None:
    x, y, vx, vy = (float(value) for value in state)
    u, _, _, modulation = potential_terms(x, y, params, targets)
    z = math.exp(-n_efolds) - 1.0
    gate_record = horizon_time_gate(z, x, params, targets)
    if gate_record is None:
        return None
    horizon_gate, horizon_denominator, horizon_active = gate_record
    kinetic_signed = vx * vx - vy * vy
    denominator = 1.0 - kinetic_signed / 6.0
    if denominator <= 0.0 or u <= 0.0 or modulation <= 0.0:
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
        "z": z,
        "x": x,
        "y": y,
        "xprime": vx,
        "yprime": vy,
        "E": math.sqrt(e2),
        "omega_dark": rho_dark / e2,
        "w_dark": pressure_dark / rho_dark,
        "kinetic_signed_crit": kinetic_crit,
        "potential_crit": potential_crit,
        "modulation": modulation,
        "horizon_gate": horizon_gate,
        "horizon_denominator": horizon_denominator,
        "horizon_active": float(horizon_active),
        "phantom_kinetic_ratio": vy * vy / max(vx * vx + vy * vy, 1.0e-15),
    }


def rhs(
    n_efolds: float,
    state: np.ndarray,
    *,
    potential_scale: float,
    params: E5137Params,
    targets: dict[str, float],
) -> np.ndarray | None:
    record = background_record(
        n_efolds,
        state,
        potential_scale=potential_scale,
        params=params,
        targets=targets,
    )
    if record is None:
        return None
    x, y, vx, vy = (float(value) for value in state)
    _, ux, uy, _ = potential_terms(x, y, params, targets)
    e2 = record["E"] ** 2
    matter = OMEGA_M0 * math.exp(-3.0 * n_efolds)
    radiation = OMEGA_R0 * math.exp(-4.0 * n_efolds)
    dlnh_dn = -1.5 * matter / e2 - 2.0 * radiation / e2 - 0.5 * (vx * vx - vy * vy)
    ax = -(3.0 + dlnh_dn) * vx - 3.0 * potential_scale * ux / e2
    ay = -(3.0 + dlnh_dn) * vy + 3.0 * potential_scale * uy / e2
    return record["horizon_gate"] * np.array([vx, vy, ax, ay], dtype=float)


def integrate(
    *,
    potential_scale: float,
    params: E5137Params,
    targets: dict[str, float],
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
            targets=targets,
        )
        if record is None:
            return None
        if keep_trajectory:
            trajectory.append(record)
        if index == len(grid) - 1:
            return trajectory if keep_trajectory else record
        step = grid[index + 1] - n_efolds
        k1 = rhs(n_efolds, state, potential_scale=potential_scale, params=params, targets=targets)
        if k1 is None:
            return None
        k2 = rhs(
            n_efolds + 0.5 * step,
            state + 0.5 * step * k1,
            potential_scale=potential_scale,
            params=params,
            targets=targets,
        )
        if k2 is None:
            return None
        k3 = rhs(
            n_efolds + 0.5 * step,
            state + 0.5 * step * k2,
            potential_scale=potential_scale,
            params=params,
            targets=targets,
        )
        if k3 is None:
            return None
        k4 = rhs(
            n_efolds + step,
            state + step * k3,
            potential_scale=potential_scale,
            params=params,
            targets=targets,
        )
        if k4 is None:
            return None
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return None


def shoot_potential_scale(params: E5137Params, targets: dict[str, float]) -> float | None:
    low = 1.0e-10
    high = 4.0
    high_record = integrate(
        potential_scale=high,
        params=params,
        targets=targets,
        steps=SHOOT_STEPS,
    )
    while (
        high_record is not None
        and high_record["omega_dark"] < OMEGA_DARK0_TARGET
        and high < 1.0e4
    ):
        high *= 2.0
        high_record = integrate(
            potential_scale=high,
            params=params,
            targets=targets,
            steps=SHOOT_STEPS,
        )
    if high_record is None:
        return None
    for _ in range(28):
        mid = 0.5 * (low + high)
        record = integrate(
            potential_scale=mid,
            params=params,
            targets=targets,
            steps=SHOOT_STEPS,
        )
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


def estimate_crossing_z(trajectory: pd.DataFrame, target_crossing_z: float) -> float:
    low_z = trajectory.loc[trajectory["z"].le(5.0), ["z", "w_dark"]].copy()
    if len(low_z) < 2:
        return math.nan
    z_values = low_z["z"].to_numpy()
    sign_values = low_z["w_dark"].to_numpy() + 1.0
    crossings: list[float] = []
    for index in range(len(sign_values) - 1):
        left = sign_values[index]
        right = sign_values[index + 1]
        if left == 0.0:
            crossings.append(float(z_values[index]))
        elif left * right < 0.0:
            fraction = abs(left) / (abs(left) + abs(right))
            crossings.append(float(z_values[index] + fraction * (z_values[index + 1] - z_values[index])))
    if not crossings:
        return math.nan
    return min(crossings, key=lambda value: abs(value - target_crossing_z))


def trajectory_score(trajectory: pd.DataFrame, targets: dict[str, float]) -> dict[str, float | bool]:
    z_eval = np.array([0.0, targets["phantom_crossing_z"], 0.5, 1.0, 2.0])
    w_model = np.array([sample_at_redshift(trajectory, z, "w_dark") for z in z_eval])
    w_target = target_w_of_z(z_eval, targets)
    w_rms = float(np.sqrt(np.mean((w_model - w_target) ** 2)))
    crossing_z = estimate_crossing_z(trajectory, targets["phantom_crossing_z"])
    crossing_error = abs(crossing_z - targets["phantom_crossing_z"]) if math.isfinite(crossing_z) else math.nan
    w0 = float(w_model[0])
    z10_max = float(trajectory.loc[trajectory["z"].ge(10.0), "omega_dark"].max())
    z100 = sample_at_redshift(trajectory, 100.0, "omega_dark")
    horizon_gate_min = float(trajectory["horizon_gate"].min())
    horizon_gate_max = float(trajectory["horizon_gate"].max())
    horizon_denominator_min = float(trajectory["horizon_denominator"].min())
    early_penalty = max(0.0, z10_max - 0.01)
    crossing_penalty = 0.1 if not math.isfinite(crossing_z) else 0.08 * crossing_error
    objective = w_rms + 3.0 * early_penalty + crossing_penalty + 0.25 * abs(w0 - targets["w0"])
    return {
        "w0": w0,
        "w_z_cross_target": float(w_model[1]),
        "w_z0p5": float(w_model[2]),
        "w_z1": float(w_model[3]),
        "w_z2": float(w_model[4]),
        "w_rms_z0_2": w_rms,
        "objective": float(objective),
        "w0_abs_error": abs(w0 - targets["w0"]),
        "omega_dark_max_z_gt_10": z10_max,
        "omega_dark_z100": z100,
        "crossing_z_est": crossing_z,
        "crossing_z_abs_error": crossing_error,
        "phantom_crosses": math.isfinite(crossing_z),
        "early_sanity": z10_max < 0.01,
        "horizon_gate_min": horizon_gate_min,
        "horizon_gate_max": horizon_gate_max,
        "horizon_denominator_min": horizon_denominator_min,
        "geometric_gate_positive": horizon_denominator_min > 0.0,
    }


def bounded_log_perturb(value: float, rng: np.random.Generator, sigma: float, low: float, high: float) -> float:
    return float(np.clip(math.exp(math.log(value) + rng.normal(0.0, sigma)), low, high))


def make_params_from_seed(
    seed: pd.Series,
    *,
    rng: np.random.Generator,
    mode: str,
    perturb: bool,
) -> E5137Params:
    if perturb:
        m_phi_sq = bounded_log_perturb(float(seed["m_phi_sq"]), rng, 0.45, 0.01, 8.0)
        m_chi_sq = bounded_log_perturb(float(seed["m_chi_sq"]), rng, 0.45, 0.01, 8.0)
        g0_sq = bounded_log_perturb(float(seed["g_sq"]), rng, 0.55, 0.01, 12.0)
        x0 = float(np.clip(float(seed["x0"]) + rng.normal(0.0, 0.30), -2.0, 2.0))
        y0 = float(np.clip(float(seed["y0"]) + rng.normal(0.0, 0.30), -2.0, 2.0))
        vx0 = float(np.clip(float(seed["vx0"]) + rng.normal(0.0, 0.40), -2.5, 2.5))
        vy0 = float(np.clip(float(seed["vy0"]) + rng.normal(0.0, 0.40), -2.5, 2.5))
    else:
        m_phi_sq = float(seed["m_phi_sq"])
        m_chi_sq = float(seed["m_chi_sq"])
        g0_sq = float(seed["g_sq"])
        x0 = float(seed["x0"])
        y0 = float(seed["y0"])
        vx0 = float(seed["vx0"])
        vy0 = float(seed["vy0"])

    if mode in {"baseline_no_gate", "horizon_time_gate"}:
        alpha_5 = 1.0
        alpha_137 = 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return E5137Params(
        m_phi_sq=m_phi_sq,
        m_chi_sq=m_chi_sq,
        g0_sq=g0_sq,
        x0=x0,
        y0=y0,
        vx0=vx0,
        vy0=vy0,
        alpha_5=alpha_5,
        alpha_137=alpha_137,
        mode=mode,
    )


def select_seed_rows() -> pd.DataFrame:
    if not SEED_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing seed table {SEED_TABLE_PATH}; run lagrangian_twofield_dynamics_scan.py first."
        )
    source = pd.read_csv(SEED_TABLE_PATH)
    seed_ids: set[int] = set(source.sort_values("w_rms_z0_2").head(8)["case_id"].astype(int))
    safe = source.loc[source["early_sanity"]].sort_values("w_rms_z0_2").head(8)
    crossing = source.loc[source["phantom_crosses_w_grid"]].sort_values("w_rms_z0_2").head(8)
    both = source.loc[source["early_sanity"] & source["phantom_crosses_w_grid"]].sort_values("w_rms_z0_2").head(8)
    seed_ids.update(safe["case_id"].astype(int))
    seed_ids.update(crossing["case_id"].astype(int))
    seed_ids.update(both["case_id"].astype(int))
    return source.loc[source["case_id"].isin(seed_ids)].sort_values("w_rms_z0_2").reset_index(drop=True)


def candidate_params(seeds: pd.DataFrame, targets: dict[str, float]) -> list[E5137Params]:
    rng = np.random.default_rng(RNG_SEED)
    candidates: list[E5137Params] = []
    for _, seed in seeds.iterrows():
        for mode in ["baseline_no_gate", "horizon_time_gate"]:
            candidates.append(make_params_from_seed(seed, rng=rng, mode=mode, perturb=False))
            for _ in range(LOCAL_DRAWS_PER_SEED):
                params = make_params_from_seed(seed, rng=rng, mode=mode, perturb=True)
                if params.vx0 * params.vx0 - params.vy0 * params.vy0 >= 5.5:
                    continue
                candidates.append(params)
    return candidates


def run_case(
    case_id: int,
    params: E5137Params,
    targets: dict[str, float],
) -> tuple[dict[str, object], pd.DataFrame] | None:
    potential_scale = shoot_potential_scale(params, targets)
    if potential_scale is None:
        return None
    trajectory_records = integrate(
        potential_scale=potential_scale,
        params=params,
        targets=targets,
        steps=FINAL_STEPS,
        keep_trajectory=True,
    )
    if not isinstance(trajectory_records, list):
        return None
    trajectory = pd.DataFrame(trajectory_records)
    score = trajectory_score(trajectory, targets)
    row: dict[str, object] = {
        "case_id": case_id,
        "mode": params.mode,
        "potential_scale_A": potential_scale,
        "m_phi_sq": params.m_phi_sq,
        "m_chi_sq": params.m_chi_sq,
        "g0_sq": params.g0_sq,
        "x0": params.x0,
        "y0": params.y0,
        "vx0": params.vx0,
        "vy0": params.vy0,
        "alpha_5": params.alpha_5,
        "alpha_137": params.alpha_137,
        **score,
    }
    trajectory = trajectory.assign(case_id=case_id, mode=params.mode)
    return row, trajectory


def optimize(targets: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = select_seed_rows()
    candidates = candidate_params(seeds, targets)
    rows: list[dict[str, object]] = []
    trajectories_by_case: dict[int, pd.DataFrame] = {}
    for case_id, params in enumerate(candidates):
        result = run_case(case_id, params, targets)
        if result is None:
            continue
        row, trajectory = result
        rows.append(row)
        trajectories_by_case[case_id] = trajectory
    results = pd.DataFrame(rows)
    if results.empty:
        return seeds, results, pd.DataFrame()
    results = results.sort_values(["objective", "w_rms_z0_2"]).reset_index(drop=True)
    best_ids: set[int] = set(results.head(8)["case_id"].astype(int))
    for mode in ["baseline_no_gate", "horizon_time_gate"]:
        mode_rows = results.loc[results["mode"].eq(mode)].sort_values(["objective", "w_rms_z0_2"]).head(5)
        best_ids.update(mode_rows["case_id"].astype(int))
        both = results.loc[
            results["mode"].eq(mode) & results["early_sanity"] & results["phantom_crosses"]
        ].sort_values(["objective", "w_rms_z0_2"]).head(5)
        best_ids.update(both["case_id"].astype(int))
    best_trajectories = [
        trajectories_by_case[case_id].iloc[::8].copy()
        for case_id in sorted(best_ids)
        if case_id in trajectories_by_case
    ]
    trajectory_table = pd.concat(best_trajectories, ignore_index=True) if best_trajectories else pd.DataFrame()
    return seeds, results, trajectory_table


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 10) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(rows).to_markdown(index=False, floatfmt=".6g")


def write_report(targets: dict[str, float], seeds: pd.DataFrame, results: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = results.loc[results["mode"].eq("baseline_no_gate")] if not results.empty else results
    gated = results.loc[results["mode"].eq("horizon_time_gate")] if not results.empty else results
    baseline_both = baseline.loc[baseline["early_sanity"] & baseline["phantom_crosses"]] if not baseline.empty else baseline
    gated_both = gated.loc[gated["early_sanity"] & gated["phantom_crosses"]] if not gated.empty else gated
    baseline_best = baseline.sort_values(["objective", "w_rms_z0_2"]) if not baseline.empty else baseline
    gated_best = gated.sort_values(["objective", "w_rms_z0_2"]) if not gated.empty else gated
    columns = [
        "case_id",
        "mode",
        "objective",
        "w_rms_z0_2",
        "w0",
        "w_z0p5",
        "w_z1",
        "w_z2",
        "omega_dark_max_z_gt_10",
        "crossing_z_est",
        "horizon_gate_min",
        "horizon_gate_max",
        "horizon_denominator_min",
        "geometric_gate_positive",
    ]
    deep_columns = [
        "case_id",
        "mode",
        "objective",
        "w_rms_z0_2",
        "w0",
        "w_z0p5",
        "w_z1",
        "w_z2",
        "omega_dark_max_z_gt_10",
        "phantom_crosses",
        "early_sanity",
        "m_phi_sq",
        "m_chi_sq",
        "g0_sq",
        "horizon_gate_max",
    ]
    best_baseline = baseline_best.iloc[0] if not baseline_best.empty else None
    best_gated = gated_best.iloc[0] if not gated_best.empty else None
    best_gated_both = (
        gated_both.sort_values(["objective", "w_rms_z0_2"]).iloc[0]
        if not gated_both.empty
        else None
    )
    lines = [
        "# e-5-137 horizon time-dilation gate",
        "",
        "Status: local FRW integration around the best rows from the previous",
        "canonical+phantom two-field scan. The kinetic-braiding branch is not used.",
        "This pass tests the author's horizon time-dilation hypothesis as a",
        "geometric low-redshift evolution gate.",
        "",
        "## Lagrangian and gate layer",
        "",
        "```text",
        "L_dark = -1/2 (partial phi)^2 + 1/2 (partial chi)^2 - Lambda^4 U(x,y)",
        "x = phi/M_Pl, y = chi/M_Pl",
        "",
        "U = 1 + m_phi^2 x^2/2 + m_chi^2 y^2/2",
        "    + g0^2 x^2 y^2/2",
        "",
        "dt_observer = dt_local * H_gate(z,x)",
        "H_gate(z,x) = 1 / sqrt(1 - (q / 117) exp(z) cos^2(5x))",
        "```",
        "",
        "The gate is applied only in the CPL comparison window `0 <= z <= 2`.",
        "The literal formula is exponentially singular at high redshift, so the",
        "early-universe integration keeps `H_gate = 1` outside that observer",
        "window and still audits early dark energy separately.",
        "",
        "## Target",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| q | {targets['q']:.12g} |",
        f"| delta_phi | {targets['delta_phi']:.12g} |",
        f"| w0 | {targets['w0']:.12g} |",
        f"| wa | {targets['wa']:.12g} |",
        f"| w(z=0.5) | {float(target_w_of_z(0.5, targets)):.12g} |",
        f"| w(z=1) | {float(target_w_of_z(1.0, targets)):.12g} |",
        f"| phantom crossing redshift | {targets['phantom_crossing_z']:.12g} |",
        "",
        "## Seed rows used",
        "",
        markdown_table(
            seeds,
            ["case_id", "w_rms_z0_2", "w0", "w_z0p5", "w_z1", "omega_dark_max_z_gt_10", "early_sanity"],
            rows=14,
        ),
        "",
        "## Best baseline rows without horizon gate",
        "",
        markdown_table(baseline_best, columns),
        "",
        "## Best horizon-gated rows",
        "",
        markdown_table(gated_best, columns),
        "",
        "## Horizon-gated rows passing early sanity and crossing",
        "",
        markdown_table(gated_both.sort_values(["objective", "w_rms_z0_2"]) if not gated_both.empty else gated_both, columns),
        "",
        "## Deepest horizon-gated rows by w(z=1)",
        "",
        markdown_table(gated.sort_values("w_z1") if not gated.empty else gated, deep_columns),
        "",
        "## Gate result",
        "",
    ]
    if best_baseline is not None and best_gated is not None:
        delta_w1 = float(best_gated["w_z1"] - best_baseline["w_z1"])
        lines.extend(
            [
                "The horizon gate is numerically well behaved in the low-redshift",
                "window, but its invariant strength is tiny:",
                f"`q/117 = {targets['q'] / 117.0:.6g}`.",
                "",
                f"Best baseline row has `w(z=1) = {best_baseline['w_z1']:.6g}`.",
                f"Best horizon-gated row has `w(z=1) = {best_gated['w_z1']:.6g}`.",
                f"The best-row shift is `Delta w(z=1) = {delta_w1:.6g}`.",
            ]
        )
    if best_gated_both is None:
        lines.extend(
            [
                "",
                "No horizon-gated row simultaneously passes early sanity and the sampled",
                "phantom crossing in this local search.",
            ]
        )
    else:
        target_w1 = float(target_w_of_z(1.0, targets))
        residual_w1 = float(best_gated_both["w_z1"] - target_w1)
        deepest_gated = gated.sort_values("w_z1").iloc[0] if not gated.empty else None
        lines.extend(
            [
                "",
                "Best horizon-gated crossing row:",
                f"`w_rms_z0_2 = {best_gated_both['w_rms_z0_2']:.6g}`,",
                f"`w0 = {best_gated_both['w0']:.6g}`,",
                f"`w(z=1) = {best_gated_both['w_z1']:.6g}`,",
                f"`target w(z=1) = {target_w1:.6g}`,",
                f"`residual = {residual_w1:.6g}`.",
            ]
        )
        if deepest_gated is not None:
            deepest_residual = float(deepest_gated["w_z1"] - target_w1)
            lines.extend(
                [
                    "",
                    "The deepest `w(z=1)` row is closer to the target, but it fails the",
                    "physical gates:",
                    f"`w(z=1) = {deepest_gated['w_z1']:.6g}`,",
                    f"`residual = {deepest_residual:.6g}`,",
                    f"`Omega_dark(z>10)_max = {deepest_gated['omega_dark_max_z_gt_10']:.6g}`,",
                    f"`early_sanity = {deepest_gated['early_sanity']}`,",
                    f"`phantom_crosses = {deepest_gated['phantom_crosses']}`.",
                ]
            )
    lines.extend(
        [
            "",
            "## Stability audit",
            "",
            "The geometric denominator remains positive for accepted gated rows, so",
            "the horizon factor itself does not introduce a denominator singularity",
            "inside the CPL window. However, this does not automatically prove a",
            "fundamental no-ghost/no-gradient completion: the underlying two-field",
            "toy still contains the explicit phantom-sign component used in the",
            "previous scan. The result is therefore a background gate audit, not a",
            "UV-stability proof.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    seeds, results, trajectories = optimize(targets)
    SCAN_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(SCAN_TABLE_PATH, index=False)
    trajectories.to_csv(BEST_TRAJECTORY_PATH, index=False)
    write_report(targets, seeds, results)
    print(REPORT_PATH)
    print(SCAN_TABLE_PATH)
    print(BEST_TRAJECTORY_PATH)


if __name__ == "__main__":
    main()
