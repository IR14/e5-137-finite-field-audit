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


OMEGA_DE_0 = 0.69
GRID_SIZE = 1_000_000
MU_VALUES = (0.3, 1.0, 5.0, 10.0, 20.0, 50.0)
REPORT_PATH = Path("outputs/reports/e5_137_lagrangian_first_pass.md")
TABLE_PATH = Path("outputs/tables/e5_137_lagrangian_first_pass.csv")


@dataclass(frozen=True)
class HarmonicVariant:
    name: str
    harmonics: tuple[tuple[int, float], ...]
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
    lambda_target = math.sqrt(3.0 * (1.0 + w0) / OMEGA_DE_0)
    return {
        "q": q,
        "delta_phi": delta,
        "s": s,
        "T5": t5,
        "w0_target": w0,
        "wa_target": wa,
        "lambda_target": lambda_target,
    }


def variants(targets: dict[str, float]) -> tuple[HarmonicVariant, ...]:
    q = targets["q"]
    delta = targets["delta_phi"]
    n = N_TOPOLOGICAL
    return (
        HarmonicVariant(
            name="q_delta",
            harmonics=((1, q), (n, delta)),
            note="Only the low-frequency e-5-137 q and delta_phi harmonics.",
        ),
        HarmonicVariant(
            name="q_delta_137_1over137",
            harmonics=((1, q), (n, delta), (137, 1.0 / 137.0)),
            note="Literal 137 harmonic with amplitude 1/137.",
        ),
        HarmonicVariant(
            name="q_delta_137_1over1372",
            harmonics=((1, q), (n, delta), (137, 1.0 / 137.0**2)),
            note="Curvature-tamed 137 harmonic with amplitude 1/137^2.",
        ),
    )


def potential_terms(
    x: np.ndarray,
    harmonics: tuple[tuple[int, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.ones_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)
    for k, amplitude in harmonics:
        u += amplitude * np.cos(k * x)
        ux += -amplitude * k * np.sin(k * x)
        uxx += -amplitude * k * k * np.cos(k * x)
    return u, ux, uxx


def local_w_from_lambda(lambda_value: np.ndarray) -> np.ndarray:
    return -1.0 + OMEGA_DE_0 * lambda_value**2 / 3.0


def scan_variant(
    variant: HarmonicVariant,
    targets: dict[str, float],
    x: np.ndarray,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    u, ux, uxx = potential_terms(x, variant.harmonics)
    gradient = ux / u
    curvature = uxx / u
    max_gradient_index = int(np.argmax(np.abs(gradient)))
    max_curvature_abs = float(np.max(np.abs(curvature)))
    lambda_target = targets["lambda_target"]
    mu_required = float(abs(gradient[max_gradient_index]) / lambda_target)
    m2_at_mu_required = float(
        3.0 * OMEGA_DE_0 / mu_required**2 * curvature[max_gradient_index]
    )
    summary = {
        "variant": variant.name,
        "note": variant.note,
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "max_abs_ux_over_u": float(abs(gradient[max_gradient_index])),
        "phase_at_max_gradient": float(x[max_gradient_index]),
        "mu_required_for_project_w0": mu_required,
        "m2_over_h0_sq_at_mu_required": m2_at_mu_required,
        "mu_min_for_worst_curvature_below_h0": float(
            math.sqrt(3.0 * OMEGA_DE_0 * max_curvature_abs)
        ),
    }

    rows = []
    for mu in MU_VALUES:
        lam = gradient / mu
        w = local_w_from_lambda(lam)
        best_index = int(np.argmin(np.abs(w - targets["w0_target"])))
        rows.append(
            {
                "variant": variant.name,
                "mu_f_over_mpl": float(mu),
                "best_local_w0": float(w[best_index]),
                "phase_x": float(x[best_index]),
                "lambda_mpl_vprime_over_v": float(lam[best_index]),
                "m2_over_h0_sq": float(
                    3.0 * OMEGA_DE_0 / mu**2 * curvature[best_index]
                ),
                "target_abs_error": float(abs(w[best_index] - targets["w0_target"])),
            }
        )
    return summary, rows


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".6g")


def write_report(
    targets: dict[str, float],
    summaries: pd.DataFrame,
    scans: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# e-5-137 Lagrangian first-pass audit",
        "",
        "Status: local effective-field-theory bookkeeping, not a completed physical model.",
        "",
        "## Trial Lagrangian",
        "",
        "Use a dimensionless phase x = phi / f and mu = f / M_Pl.",
        "",
        "```text",
        "L = (M_Pl^2 / 2) R - 1/2 (partial phi)^2 - Lambda^4 U(phi/f)",
        "U(x) = 1 + sum_i a_i cos(k_i x)",
        "lambda(phi) = M_Pl V_phi / V = (1/mu) U_x / U",
        "m_eff^2 / H0^2 = 3 Omega_DE0 (1/mu^2) U_xx / U",
        "local 1 + w ~= Omega_DE0 lambda^2 / 3",
        "```",
        "",
        "This is a local slow-roll/slope audit. It does not integrate the full",
        "background equation of motion, so the `wa` target is recorded but not fitted.",
        "",
        "## Frozen project numbers",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| q | {targets['q']:.12g} |",
        f"| delta_phi | {targets['delta_phi']:.12g} |",
        f"| s | {targets['s']:.12g} |",
        f"| T5 | {targets['T5']:.12g} |",
        f"| project w0 target | {targets['w0_target']:.12g} |",
        f"| project wa target | {targets['wa_target']:.12g} |",
        f"| lambda needed for project w0 at Omega_DE0={OMEGA_DE_0:g} | {targets['lambda_target']:.12g} |",
        "",
        "## Variant summary",
        "",
        markdown_table(
            summaries,
            [
                "variant",
                "u_min",
                "u_max",
                "max_abs_ux_over_u",
                "mu_required_for_project_w0",
                "m2_over_h0_sq_at_mu_required",
                "mu_min_for_worst_curvature_below_h0",
            ],
        ),
        "",
        "Reading rule: `mu_required_for_project_w0` is the field-range scale needed",
        "at the steepest phase point to reproduce the project-level `w0` target.",
        "`mu_min_for_worst_curvature_below_h0` is a conservative softness check:",
        "if mu is lower than that value, some parts of the potential are heavier",
        "than H0 in curvature units.",
        "",
        "## Fixed-mu scan",
        "",
        markdown_table(
            scans,
            [
                "variant",
                "mu_f_over_mpl",
                "best_local_w0",
                "phase_x",
                "lambda_mpl_vprime_over_v",
                "m2_over_h0_sq",
                "target_abs_error",
            ],
        ),
        "",
        "## First physical readout",
        "",
        "- The low-frequency `q_delta` potential is naturally soft for mu above",
        "  roughly 1.8, but then it stays extremely close to w = -1. To reproduce",
        "  the project's w0 ~= -0.752 locally, it needs mu ~= 0.293 and an",
        "  effective curvature of order 2.23 H0^2 at the steepest phase point.",
        "- A literal `cos(137 x)/137` term can produce the required slope with",
        "  mu of order 1, but its worst-case curvature is severe; keeping all",
        "  such modes softer than H0 requires mu ~= 17.6.",
        "- Replacing the literal 137 amplitude by 1/137^2 tames curvature, but the",
        "  model again behaves much like the low-frequency case and needs",
        "  sub-Planckian mu to reach the project w0 target.",
        "- Therefore the first Lagrangian gate is clear: the 137 structure cannot",
        "  be inserted as a naive high-frequency dark-energy harmonic unless the",
        "  model accepts a very large field range, suppresses the harmonic, or",
        "  moves the 137 sector into another field/operator rather than directly",
        "  into the quintessence potential.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    x = np.linspace(0.0, 2.0 * math.pi, GRID_SIZE, endpoint=False)
    summary_rows = []
    scan_rows = []
    for variant in variants(targets):
        summary, rows = scan_variant(variant, targets, x)
        summary_rows.append(summary)
        scan_rows.extend(rows)

    summaries = pd.DataFrame(summary_rows)
    scans = pd.DataFrame(scan_rows)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    scans.to_csv(TABLE_PATH, index=False)
    write_report(targets, summaries, scans)
    print(REPORT_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
