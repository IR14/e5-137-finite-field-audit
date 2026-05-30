from __future__ import annotations

import math
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


REPORT_PATH = Path("outputs/reports/e5_137_quintom_gate.md")
SCAN_TABLE_PATH = Path("outputs/tables/e5_137_quintom_interaction_scan.csv")
BEST_PROFILE_PATH = Path("outputs/tables/e5_137_quintom_best_interaction_profile.csv")

Z_GRID = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
W_PLUS_GRID = np.linspace(-0.70, 0.20, 46)
W_MINUS_GRID = np.linspace(-3.00, -1.62, 70)


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


def w_cpl_of_a(a: np.ndarray, w0: float, wa: float) -> np.ndarray:
    return w0 + wa * (1.0 - a)


def rho_cpl_ratio_of_a(a: np.ndarray, w0: float, wa: float) -> np.ndarray:
    return a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))


def phantom_fraction(w_eff: np.ndarray, w_plus: float, w_minus: float) -> np.ndarray:
    return (w_eff - w_plus) / (w_minus - w_plus)


def interaction_profile(
    *,
    w_plus: float,
    w_minus: float,
    z_grid: np.ndarray,
    targets: dict[str, float],
) -> pd.DataFrame | None:
    a = 1.0 / (1.0 + z_grid)
    w_eff = w_cpl_of_a(a, targets["w0"], targets["wa"])
    delta_w = w_minus - w_plus
    f_minus = phantom_fraction(w_eff, w_plus, w_minus)
    if np.any(f_minus < -1.0e-12) or np.any(f_minus > 1.0 + 1.0e-12):
        return None

    rho = rho_cpl_ratio_of_a(a, targets["w0"], targets["wa"])
    dwdn = -targets["wa"] * a
    fprime = dwdn / delta_w
    # I_plus is defined by rho_plus' + 3(1+w_plus)rho_plus = I_plus.
    # The phantom component receives -I_plus so that total energy is conserved.
    interaction_plus = -fprime - 3.0 * f_minus * (1.0 - f_minus) * delta_w
    interaction_plus_rho_plus = interaction_plus / np.clip(1.0 - f_minus, 1.0e-12, None)
    interaction_to_phantom_rho = -interaction_plus / np.clip(f_minus, 1.0e-12, None)
    independent_derivative = -3.0 * delta_w * delta_w * f_minus * (1.0 - f_minus)
    target_derivative = dwdn

    return pd.DataFrame(
        {
            "z": z_grid,
            "a": a,
            "w_eff": w_eff,
            "rho_de_over_today": rho,
            "f_phantom": f_minus,
            "f_nonphantom": 1.0 - f_minus,
            "target_dw_dln_a": target_derivative,
            "independent_dw_dln_a": independent_derivative,
            "interaction_plus_over_rho_de": interaction_plus,
            "interaction_plus_over_rho_plus": interaction_plus_rho_plus,
            "interaction_to_phantom_over_rho_phantom": interaction_to_phantom_rho,
            "energy_flow_direction": np.where(
                interaction_plus > 0.0,
                "to_nonphantom_from_phantom",
                "to_phantom_from_nonphantom",
            ),
        }
    )


def scan_pairs(targets: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    best_profile: pd.DataFrame | None = None
    best_score = float("inf")
    for w_plus in W_PLUS_GRID:
        for w_minus in W_MINUS_GRID:
            if not (w_plus > targets["w0"] and w_minus <= targets["w_inf"]):
                continue
            profile = interaction_profile(
                w_plus=float(w_plus),
                w_minus=float(w_minus),
                z_grid=Z_GRID,
                targets=targets,
            )
            if profile is None:
                continue
            interaction = profile["interaction_plus_over_rho_de"].to_numpy(dtype=float)
            max_abs_interaction = float(np.max(np.abs(interaction)))
            rms_interaction = float(np.sqrt(np.mean(interaction * interaction)))
            score = max_abs_interaction + 0.25 * rms_interaction
            rows.append(
                {
                    "w_plus": float(w_plus),
                    "w_minus": float(w_minus),
                    "f_phantom_today": float(profile.loc[profile["z"].eq(0.0), "f_phantom"].iloc[0]),
                    "f_phantom_z1": float(profile.loc[profile["z"].eq(1.0), "f_phantom"].iloc[0]),
                    "f_phantom_z10": float(profile.loc[profile["z"].eq(10.0), "f_phantom"].iloc[0]),
                    "max_abs_interaction_plus_over_rho_de": max_abs_interaction,
                    "rms_interaction_plus_over_rho_de": rms_interaction,
                    "interaction_score": score,
                    "min_interaction_plus_over_rho_de": float(np.min(interaction)),
                    "max_interaction_plus_over_rho_de": float(np.max(interaction)),
                    "dominant_flow": (
                        "mixed"
                        if np.min(interaction) < 0.0 < np.max(interaction)
                        else (
                            "to_phantom_from_nonphantom"
                            if np.max(interaction) <= 0.0
                            else "to_nonphantom_from_phantom"
                        )
                    ),
                }
            )
            if score < best_score:
                best_score = score
                best_profile = profile.assign(w_plus=float(w_plus), w_minus=float(w_minus))
    scan = pd.DataFrame(rows).sort_values("interaction_score").reset_index(drop=True)
    if best_profile is None:
        best_profile = pd.DataFrame()
    return scan, best_profile


def independent_quintom_derivative_sign(targets: dict[str, float]) -> pd.DataFrame:
    rows = []
    examples = [(-0.70, -1.70), (-0.50, -2.00), (0.0, -2.50)]
    a = np.array([1.0])
    w_eff = w_cpl_of_a(a, targets["w0"], targets["wa"])[0]
    target_derivative = -targets["wa"]
    for w_plus, w_minus in examples:
        f = float(phantom_fraction(np.array([w_eff]), w_plus, w_minus)[0])
        delta_w = w_minus - w_plus
        independent_derivative = -3.0 * delta_w * delta_w * f * (1.0 - f)
        rows.append(
            {
                "w_plus": w_plus,
                "w_minus": w_minus,
                "f_phantom_today": f,
                "target_dw_dln_a_today": target_derivative,
                "independent_dw_dln_a_today": independent_derivative,
                "sign_matches_target": np.sign(target_derivative) == np.sign(independent_derivative),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[columns].head(rows).to_markdown(index=False, floatfmt=".6g")


def write_report(
    targets: dict[str, float],
    independent: pd.DataFrame,
    scan: pd.DataFrame,
    best_profile: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = scan.iloc[0] if not scan.empty else None
    lines = [
        "# e-5-137 quintom interaction gate",
        "",
        "Status: two-component dark-sector reconstruction for the project CPL target.",
        "",
        "## Lagrangian class",
        "",
        "A minimal effective quintom sector can be written as",
        "",
        "```text",
        "L_dark = -1/2 (partial phi)^2 - V_phi(phi)",
        "         +1/2 (partial chi)^2 - V_chi(chi)",
        "         - V_int(phi, chi)",
        "```",
        "",
        "where `phi` is non-phantom and `chi` is an effective phantom component.",
        "Equivalently, at the background-fluid level:",
        "",
        "```text",
        "rho_+' + 3(1+w_+) rho_+ = I_+",
        "rho_-' + 3(1+w_-) rho_- = -I_+",
        "w_eff = (w_+ rho_+ + w_- rho_-) / (rho_+ + rho_-)",
        "```",
        "",
        "The prime denotes d/d ln a. `I_+` is the interaction needed to force the",
        "two components to follow the project CPL target.",
        "",
        "## Project target",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| w0 | {targets['w0']:.12g} |",
        f"| wa | {targets['wa']:.12g} |",
        f"| w0 + wa | {targets['w_inf']:.12g} |",
        f"| phantom crossing redshift | {targets['phantom_crossing_z']:.12g} |",
        "",
        "## Independent two-fluid no-go",
        "",
        "For independent constant-w components with positive densities,",
        "",
        "```text",
        "dw_eff / d ln a = -3 f(1-f) (w_- - w_+)^2 <= 0",
        "```",
        "",
        "but the project CPL target has",
        "",
        "```text",
        "dw/d ln a | today = -wa > 0",
        "```",
        "",
        "so the sign is wrong before any detailed fitting.",
        "",
        markdown_table(
            independent,
            [
                "w_plus",
                "w_minus",
                "f_phantom_today",
                "target_dw_dln_a_today",
                "independent_dw_dln_a_today",
                "sign_matches_target",
            ],
            rows=10,
        ),
        "",
        "## Best interacting reconstructions",
        "",
        markdown_table(
            scan,
            [
                "w_plus",
                "w_minus",
                "f_phantom_today",
                "f_phantom_z1",
                "f_phantom_z10",
                "interaction_score",
                "max_abs_interaction_plus_over_rho_de",
                "rms_interaction_plus_over_rho_de",
                "dominant_flow",
            ],
            rows=12,
        ),
        "",
        "## Best profile",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                f"Best grid row: `w_+ = {best['w_plus']:.6g}`, `w_- = {best['w_minus']:.6g}`.",
                "",
                markdown_table(
                    best_profile,
                    [
                        "z",
                        "w_eff",
                        "f_phantom",
                        "target_dw_dln_a",
                        "independent_dw_dln_a",
                        "interaction_plus_over_rho_de",
                        "energy_flow_direction",
                    ],
                    rows=20,
                ),
            ]
        )
    else:
        lines.append("_No valid interacting rows found._")
    lines.extend(
        [
            "",
            "## Gate result",
            "",
            "The two-field route is mathematically possible only as an interacting",
            "effective sector. Independent positive-density quintom components have",
            "the wrong derivative sign for this project target.",
            "",
            "The interaction reconstruction is also not gentle: the best coarse grid",
            "still needs order-unity energy-transfer rates relative to rho_DE across",
            "the tested redshift range. This is acceptable as an effective-fluid toy,",
            "but it is not yet a healthy fundamental Lagrangian.",
            "",
            "Next physical gate: replace the constant-w fluid reconstruction by an",
            "explicit two-field dynamical system with a cutoff and check whether",
            "the same interaction profile can arise from a simple potential such as",
            "`V_int = g^2 phi^2 chi^2 / 2` without producing unstable early energy.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    independent = independent_quintom_derivative_sign(targets)
    scan, best_profile = scan_pairs(targets)
    SCAN_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    scan.to_csv(SCAN_TABLE_PATH, index=False)
    best_profile.to_csv(BEST_PROFILE_PATH, index=False)
    write_report(targets, independent, scan, best_profile)
    print(REPORT_PATH)
    print(SCAN_TABLE_PATH)
    print(BEST_PROFILE_PATH)


if __name__ == "__main__":
    main()
