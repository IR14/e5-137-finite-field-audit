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


OMEGA_M0 = 0.31
OMEGA_R0 = 9.0e-5
OMEGA_DE0 = 0.69
REPORT_PATH = Path("outputs/reports/e5_137_kessence_gate.md")
KINETIC_TABLE_PATH = Path("outputs/tables/e5_137_kessence_kinetic_gate.csv")
CPL_TABLE_PATH = Path("outputs/tables/e5_137_kessence_cpl_background.csv")


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
        "phantom_crossing_a": crossing_a,
        "phantom_crossing_z": crossing_z,
    }


def cpl_w(z: float, w0: float, wa: float) -> float:
    return w0 + wa * z / (1.0 + z)


def cpl_de_density_ratio(z: float, w0: float, wa: float) -> float:
    a = 1.0 / (1.0 + z)
    return a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * wa * (a - 1.0))


def cpl_background_table(targets: dict[str, float]) -> pd.DataFrame:
    rows = []
    for z in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        density_ratio = cpl_de_density_ratio(z, targets["w0"], targets["wa"])
        e2 = (
            OMEGA_M0 * (1.0 + z) ** 3
            + OMEGA_R0 * (1.0 + z) ** 4
            + OMEGA_DE0 * density_ratio
        )
        omega_de = OMEGA_DE0 * density_ratio / e2
        rows.append(
            {
                "z": z,
                "a": 1.0 / (1.0 + z),
                "w_cpl": cpl_w(z, targets["w0"], targets["wa"]),
                "rho_de_over_rho_de0": density_ratio,
                "E_H_over_H0": math.sqrt(e2),
                "omega_de": omega_de,
                "requires_phantom": cpl_w(z, targets["w0"], targets["wa"]) < -1.0,
            }
        )
    return pd.DataFrame(rows)


def solve_canonical_y_for_w(w: float) -> float:
    # F(y)=y-1, rho=y+1, w=(y-1)/(y+1).
    return (1.0 + w) / (1.0 - w)


def quadratic_w(y: float, kappa: float) -> float:
    # F(y) = -1 + y + kappa y^2.
    return (-1.0 + y + kappa * y * y) / (1.0 + y + 3.0 * kappa * y * y)


def quadratic_cs2(y: float, kappa: float) -> float:
    return (1.0 + 2.0 * kappa * y) / (1.0 + 6.0 * kappa * y)


def solve_quadratic_y_for_w(w: float, kappa: float) -> float | None:
    # (w*3k-k)y^2 + (w-1)y + (w+1) = 0.
    a = kappa * (3.0 * w - 1.0)
    b = w - 1.0
    c = w + 1.0
    if abs(a) < 1.0e-14:
        if abs(b) < 1.0e-14:
            return None
        y = -c / b
        return y if y >= 0.0 else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    roots = [(-b + math.sqrt(discriminant)) / (2.0 * a), (-b - math.sqrt(discriminant)) / (2.0 * a)]
    positive = [root for root in roots if root >= 0.0 and math.isfinite(root)]
    return min(positive) if positive else None


def tachyon_y_for_w(w: float) -> float:
    # F(y)=-sqrt(1-2y), w=-(1-2y), c_s^2=1-2y.
    return 0.5 * (1.0 + w)


def kinetic_gate_rows(targets: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_points = [
        ("today_w0", targets["w0"]),
        ("z1_cpl", cpl_w(1.0, targets["w0"], targets["wa"])),
        ("asymptotic_w0_plus_wa", targets["w_inf"]),
    ]
    for label, w in test_points:
        y = solve_canonical_y_for_w(w)
        rows.append(
            {
                "target": label,
                "model": "canonical_F_y_minus_1",
                "w": w,
                "y_solution": y,
                "rho_positive": y + 1.0 > 0.0,
                "ghost_free": y >= 0.0,
                "cs2": 1.0,
                "gradient_stable": True,
                "passes_single_field_stable_gate": y >= 0.0 and w >= -1.0,
            }
        )
        for kappa in [0.5, 1.0, 5.0]:
            yq = solve_quadratic_y_for_w(w, kappa)
            ghost_free = bool(yq is not None and (1.0 + 2.0 * kappa * yq) > 0.0)
            cs2 = quadratic_cs2(yq, kappa) if yq is not None else float("nan")
            rho = 1.0 + yq + 3.0 * kappa * yq * yq if yq is not None else float("nan")
            rows.append(
                {
                    "target": label,
                    "model": f"quadratic_F_kappa_{kappa:g}",
                    "w": w,
                    "y_solution": yq if yq is not None else float("nan"),
                    "rho_positive": bool(rho > 0.0) if yq is not None else False,
                    "ghost_free": ghost_free,
                    "cs2": cs2,
                    "gradient_stable": bool(cs2 > 0.0) if math.isfinite(cs2) else False,
                    "passes_single_field_stable_gate": bool(
                        yq is not None and rho > 0.0 and ghost_free and cs2 > 0.0 and w >= -1.0
                    ),
                }
            )
        yt = tachyon_y_for_w(w)
        cs2_t = 1.0 - 2.0 * yt
        rows.append(
            {
                "target": label,
                "model": "tachyon_DBI_F_minus_sqrt_1_minus_2y",
                "w": w,
                "y_solution": yt,
                "rho_positive": 0.0 <= yt < 0.5,
                "ghost_free": 0.0 <= yt < 0.5,
                "cs2": cs2_t,
                "gradient_stable": cs2_t > 0.0,
                "passes_single_field_stable_gate": 0.0 <= yt < 0.5 and cs2_t > 0.0 and w >= -1.0,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int | None = None) -> str:
    if rows is not None:
        frame = frame.head(rows)
    return frame[columns].to_markdown(index=False, floatfmt=".6g")


def write_report(targets: dict[str, float], kinetic: pd.DataFrame, cpl: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stable_today = kinetic.loc[
        kinetic["target"].eq("today_w0") & kinetic["passes_single_field_stable_gate"]
    ]
    stable_z1 = kinetic.loc[
        kinetic["target"].eq("z1_cpl") & kinetic["passes_single_field_stable_gate"]
    ]
    lines = [
        "# e-5-137 k-essence gate",
        "",
        "Status: single-field k-essence no-go check for the project w0-wa target.",
        "",
        "## General form",
        "",
        "For a single-field k-essence sector,",
        "",
        "```text",
        "S = integral d^4x sqrt(-g) [ M_Pl^2 R/2 + P(phi, X) ]",
        "X = -1/2 g^munu partial_mu phi partial_nu phi",
        "p = P",
        "rho = 2 X P_X - P",
        "rho + p = 2 X P_X",
        "c_s^2 = P_X / (P_X + 2 X P_XX)",
        "```",
        "",
        "A ghost-free single-field model has `P_X > 0`; with `X >= 0`, this gives",
        "`rho + p >= 0`, therefore `w = p/rho >= -1` when rho is positive.",
        "",
        "## Project CPL target",
        "",
        "| quantity | value |",
        "|:--|--:|",
        f"| w0 | {targets['w0']:.12g} |",
        f"| wa | {targets['wa']:.12g} |",
        f"| w(z -> infinity) = w0 + wa | {targets['w_inf']:.12g} |",
        f"| phantom crossing redshift | {targets['phantom_crossing_z']:.12g} |",
        "",
        "The target is non-phantom today, but becomes phantom above the crossing",
        "redshift. That is the central obstruction for stable single-field",
        "k-essence.",
        "",
        "## CPL background implied by target",
        "",
        markdown_table(
            cpl,
            [
                "z",
                "w_cpl",
                "rho_de_over_rho_de0",
                "E_H_over_H0",
                "omega_de",
                "requires_phantom",
            ],
        ),
        "",
        "## Kinetic-sector algebra",
        "",
        markdown_table(
            kinetic,
            [
                "target",
                "model",
                "w",
                "y_solution",
                "cs2",
                "rho_positive",
                "ghost_free",
                "passes_single_field_stable_gate",
            ],
        ),
        "",
        "## Gate result",
        "",
    ]
    if not stable_today.empty and stable_z1.empty:
        lines.extend(
            [
                "A healthy k-essence kinetic sector can reproduce the present-day",
                "`w0` value algebraically. For example, tachyon/DBI-like kinetics need",
                "`y = (1+w0)/2`, with positive sound speed.",
                "",
                "But the same single-field stable class cannot reproduce the full",
                "`w0-wa` target, because already by z=1 the CPL value is below -1.",
                "That requires phantom behavior, a ghost, a two-field quintom",
                "construction, or modified gravity/effective-fluid physics.",
            ]
        )
    else:
        lines.append("The scan did not find even a stable present-day kinetic realization.")
    lines.extend(
        [
            "",
            "## Next Lagrangian class",
            "",
            "The next honest move is not another single-field `P(X, phi)` scan.",
            "To keep the project `w0-wa` pair, move to a two-component dark sector:",
            "",
            "```text",
            "L_dark = -1/2 (partial phi)^2 - V_phi(phi)",
            "         +1/2 (partial chi)^2 - V_chi(chi)",
            "         - g^2 phi^2 chi^2 / 2",
            "```",
            "",
            "or replace the phantom component by an explicitly effective modified-gravity",
            "fluid. The two-field route can cross `w=-1`; the price is that the",
            "phantom/negative-kinetic component must be treated as an effective field",
            "with a cutoff, not a fundamental UV-complete scalar.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = project_targets()
    kinetic = kinetic_gate_rows(targets)
    cpl = cpl_background_table(targets)
    KINETIC_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    kinetic.to_csv(KINETIC_TABLE_PATH, index=False)
    cpl.to_csv(CPL_TABLE_PATH, index=False)
    write_report(targets, kinetic, cpl)
    print(REPORT_PATH)
    print(KINETIC_TABLE_PATH)
    print(CPL_TABLE_PATH)


if __name__ == "__main__":
    main()
