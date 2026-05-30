"""Phase 4 modular-symbolic audit for e-5-137 style hypotheses.

This is intentionally an audit, not a proof engine. It searches a small,
pre-declared expression grammar and reports both numerical hits and failures.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path


N = 5
D = 26
MOD = 137
PHI = MOD - 1
P_BASIS = (2, 3, 5, 13, 137)

ALPHA_INV_TARGET = 137.035999084
DELTA_ALPHA_TARGET = ALPHA_INV_TARGET - MOD
KP_TARGET = 1836.15267343
A_MU_TARGET = 0.0011659207

Q = N * (N - 2) / (math.e**4 * math.pi**3)
T5 = 2 / (math.e * N)
S_TARGET = 1 / (ALPHA_INV_TARGET * 2 * math.pi)


@dataclass(frozen=True)
class Hit:
    expression: str
    value: float
    ppm_error: float
    complexity: int


def ppm(value: float, target: float) -> float:
    return abs(value - target) / abs(target) * 1_000_000.0


def signed_residue(value: int) -> int:
    residue = value % MOD
    return residue - MOD if residue > MOD // 2 else residue


def modular_inverse(value: int) -> int:
    return pow(value % MOD, -1, MOD)


def coefficient_bank() -> list[tuple[str, int, int]]:
    """Return coefficient labels, signed integer embeddings, and complexity."""
    raw: list[tuple[str, int, int]] = []
    seeds = {1, 2, 3, 5, 6, 13, N, D, PHI}
    for value in sorted(seeds):
        raw.append((str(value), signed_residue(value), 1 if value in P_BASIS else 2))
        raw.append((f"-{value}", -signed_residue(value), 1 if value in P_BASIS else 2))
    for value in (2, 3, 5, 13, D):
        inv = signed_residue(modular_inverse(value))
        raw.append((f"inv{value}", inv, 2))
        raw.append((f"-inv{value}", -inv, 2))
    # Deduplicate exact signed values, keeping the simplest label.
    best: dict[int, tuple[str, int, int]] = {}
    for label, signed, complexity in raw:
        if signed not in best or complexity < best[signed][2]:
            best[signed] = (label, signed, complexity)
    return sorted(best.values(), key=lambda item: (item[2], abs(item[1]), item[0]))


def alpha_basis() -> list[tuple[str, float, int]]:
    return [
        ("q", Q, 1),
        ("T5*s", T5 * S_TARGET, 2),
        ("q*s", Q * S_TARGET, 2),
        ("q*T5", Q * T5, 2),
        ("q*T5*s", Q * T5 * S_TARGET, 3),
        ("q^2", Q**2, 2),
        ("s", S_TARGET, 1),
        ("s^2", S_TARGET**2, 2),
        ("T5*s^2", T5 * S_TARGET**2, 3),
    ]


def kp_base_terms() -> list[tuple[str, float, int]]:
    return [
        ("phi/(2N)", PHI / (2 * N), 2),
        ("D/2", D / 2, 1),
        ("(phi-N)/(2N)", (PHI - N) / (2 * N), 3),
        ("(137-N)/(2N)", (MOD - N) / (2 * N), 3),
        ("13", 13.0, 1),
        ("D/N + 2*P13/P2", D / N + 2 * 13 / 2, 3),
    ]


def kp_correction_basis(s_value: float) -> list[tuple[str, float, int]]:
    return [
        ("q", Q, 1),
        ("T5*s", T5 * s_value, 2),
        ("q*s", Q * s_value, 2),
        ("q*T5", Q * T5, 2),
        ("s", s_value, 1),
        ("q^2", Q**2, 2),
    ]


def search_linear_combo(
    target: float,
    bases: list[tuple[str, float, int]],
    max_terms: int,
    top_n: int,
    offset: float = 0.0,
    offset_label: str = "",
) -> list[Hit]:
    coeffs = coefficient_bank()
    hits: list[Hit] = []
    for n_terms in range(1, max_terms + 1):
        for basis_terms in itertools.combinations(bases, n_terms):
            for coeff_terms in itertools.product(coeffs, repeat=n_terms):
                value = offset
                parts: list[str] = []
                complexity = 0
                for (basis_label, basis_value, basis_complexity), (
                    coeff_label,
                    coeff_value,
                    coeff_complexity,
                ) in zip(basis_terms, coeff_terms):
                    value += coeff_value * basis_value
                    parts.append(f"{coeff_label}*{basis_label}")
                    complexity += basis_complexity + coeff_complexity
                expression = " + ".join(parts)
                if offset_label:
                    expression = f"{offset_label} + {expression}"
                    complexity += 1
                hits.append(
                    Hit(
                        expression=expression,
                        value=value,
                        ppm_error=ppm(value, target),
                        complexity=complexity,
                    )
                )
    hits.sort(key=lambda hit: (hit.ppm_error, hit.complexity, len(hit.expression)))
    return hits[:top_n]


def search_alpha(top_n: int) -> list[Hit]:
    delta_hits = search_linear_combo(
        DELTA_ALPHA_TARGET,
        alpha_basis(),
        max_terms=3,
        top_n=top_n,
    )
    return [
        Hit(
            expression=f"137 + ({hit.expression})",
            value=MOD + hit.value,
            ppm_error=ppm(MOD + hit.value, ALPHA_INV_TARGET),
            complexity=hit.complexity + 1,
        )
        for hit in delta_hits
    ]


def search_kp(alpha_inv: float, top_n: int, max_terms: int = 3) -> list[Hit]:
    s_value = 1 / (alpha_inv * 2 * math.pi)
    bracket_target = KP_TARGET / alpha_inv
    all_hits: list[Hit] = []
    for base_label, base_value, base_complexity in kp_base_terms():
        corrections = search_linear_combo(
            bracket_target,
            kp_correction_basis(s_value),
            max_terms=max_terms,
            top_n=top_n,
            offset=base_value,
            offset_label=base_label,
        )
        for correction in corrections:
            kp_value = alpha_inv * correction.value
            all_hits.append(
                Hit(
                    expression=f"alpha_inv*({correction.expression})",
                    value=kp_value,
                    ppm_error=ppm(kp_value, KP_TARGET),
                    complexity=correction.complexity + base_complexity,
                )
            )
    all_hits.sort(key=lambda hit: (hit.ppm_error, hit.complexity, len(hit.expression)))
    return all_hits[:top_n]


def out_of_sample_muon_anomaly(alpha_inv: float, alpha_hit: Hit, kp_hit: Hit) -> list[Hit]:
    """Evaluate non-fitted a_mu projections from the frozen alpha/Kp scan.

    These forms are deliberately tiny and pre-declared. The scan does not tune
    coefficients against a_mu; this is the out-of-sample check.
    """
    s_value = 1 / (alpha_inv * 2 * math.pi)
    forms = [
        ("s", s_value, 1),
        ("s*(1+s)", s_value * (1 + s_value), 2),
        ("s*(1+T5*s)", s_value * (1 + T5 * s_value), 3),
        ("s*(1+q*s)", s_value * (1 + Q * s_value), 3),
        ("s/(1-T5*s)", s_value / (1 - T5 * s_value), 3),
        ("s/(1-q*s)", s_value / (1 - Q * s_value), 3),
        ("s*(1+q)", s_value * (1 + Q), 2),
    ]
    return [
        Hit(
            expression=f"{label}  [alpha={alpha_hit.expression}; Kp={kp_hit.expression}]",
            value=value,
            ppm_error=ppm(value, A_MU_TARGET),
            complexity=complexity + alpha_hit.complexity + kp_hit.complexity,
        )
        for label, value, complexity in forms
    ]


def write_hits(path: Path, hits: list[Hit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["expression", "value", "ppm_error", "complexity"])
        writer.writeheader()
        for hit in hits:
            writer.writerow(
                {
                    "expression": hit.expression,
                    "value": f"{hit.value:.17g}",
                    "ppm_error": f"{hit.ppm_error:.9g}",
                    "complexity": hit.complexity,
                }
            )


def main() -> None:
    output_tables = Path("outputs/tables")
    output_reports = Path("outputs/reports")

    print("Phase 4 modular scan initialization")
    print(f"P={P_BASIS} MOD={MOD} phi={PHI} D={D} N={N}")
    print(f"q={Q:.17g} T5={T5:.17g} s_target={S_TARGET:.17g}")
    print(f"alpha_inv_target={ALPHA_INV_TARGET:.12f} delta_alpha={DELTA_ALPHA_TARGET:.12f}")
    print(f"Kp_target={KP_TARGET:.11f} a_mu_target={A_MU_TARGET:.10f}")

    alpha_hits = search_alpha(top_n=25)
    best_alpha = alpha_hits[0]
    kp_hits = search_kp(best_alpha.value, top_n=25, max_terms=3)
    best_kp = kp_hits[0]
    kp_wide_hits = search_kp(best_alpha.value, top_n=25, max_terms=4)
    best_kp_wide = kp_wide_hits[0]
    a_mu_hits = sorted(
        out_of_sample_muon_anomaly(best_alpha.value, best_alpha, best_kp_wide),
        key=lambda hit: (hit.ppm_error, hit.complexity),
    )

    write_hits(output_tables / "phase4_modular_alpha_hits.csv", alpha_hits)
    write_hits(output_tables / "phase4_modular_kp_hits.csv", kp_hits)
    write_hits(output_tables / "phase4_modular_kp_wide_hits.csv", kp_wide_hits)
    write_hits(output_tables / "phase4_modular_amu_oos.csv", a_mu_hits)

    alpha_pass = best_alpha.ppm_error < 0.01
    kp_pass = best_kp.ppm_error < 0.5
    kp_wide_pass = best_kp_wide.ppm_error < 0.5
    amu_pass = a_mu_hits[0].ppm_error < 500
    survived = alpha_pass and kp_wide_pass and amu_pass

    print("best alpha:", best_alpha)
    print("best Kp strict:", best_kp)
    print("best Kp wide:", best_kp_wide)
    print("best a_mu OOS:", a_mu_hits[0])
    print(
        "survived="
        f"{survived} alpha_pass={alpha_pass} kp_strict_pass={kp_pass} "
        f"kp_wide_pass={kp_wide_pass} amu_pass={amu_pass}"
    )

    report = [
        "# Phase 4 modular number-theory scan",
        "",
        "This report initializes the pure theoretical-number audit requested for Phase 4.",
        "It is an exploratory symbolic-regression scan, not a physical derivation.",
        "",
        "## Fixed Constants",
        "",
        f"- prime modulus: `{MOD}`",
        f"- Euler phi: `phi(137) = {PHI}`",
        f"- decomposition: `{PHI} = {D}*{N} + 6`",
        f"- `q = {Q:.17g}`",
        f"- `T5 = {T5:.17g}`",
        f"- target `alpha^-1 = {ALPHA_INV_TARGET:.12f}`",
        f"- target `Kp = Mp/me = {KP_TARGET:.11f}`",
        f"- out-of-sample target `a_mu = {A_MU_TARGET:.10f}`",
        "",
        "## Best Alpha Candidate",
        "",
        f"- expression: `{best_alpha.expression}`",
        f"- value: `{best_alpha.value:.12f}`",
        f"- ppm error: `{best_alpha.ppm_error:.6g}`",
        f"- complexity: `{best_alpha.complexity}`",
        f"- threshold `<0.01 ppm`: `{'pass' if alpha_pass else 'failed'}`",
        "",
        "## Best Proton-Ratio Candidate: Strict Grammar",
        "",
        f"- expression: `{best_kp.expression}`",
        f"- value: `{best_kp.value:.11f}`",
        f"- ppm error: `{best_kp.ppm_error:.6g}`",
        f"- complexity: `{best_kp.complexity}`",
        f"- threshold `<0.5 ppm`: `{'pass' if kp_pass else 'failed'}`",
        "",
        "## Best Proton-Ratio Candidate: Wide Grammar",
        "",
        "This allows one additional correction term. It is useful as a boundary test,",
        "but its complexity is high enough that it should be treated as a numerical",
        "hit, not as a derivation.",
        "",
        f"- expression: `{best_kp_wide.expression}`",
        f"- value: `{best_kp_wide.value:.11f}`",
        f"- ppm error: `{best_kp_wide.ppm_error:.6g}`",
        f"- complexity: `{best_kp_wide.complexity}`",
        f"- threshold `<0.5 ppm`: `{'pass' if kp_wide_pass else 'failed'}`",
        "",
        "## Out-of-Sample Muon Anomaly",
        "",
        f"- best pre-declared projection: `{a_mu_hits[0].expression}`",
        f"- value: `{a_mu_hits[0].value:.12f}`",
        f"- ppm error: `{a_mu_hits[0].ppm_error:.6g}`",
        f"- threshold `<500 ppm`: `{'pass' if amu_pass else 'failed'}`",
        "",
        "## Verdict",
        "",
        (
            "The Phase 4 grammar survived all gates."
            if survived
            else "The Phase 4 grammar did not survive the out-of-sample gate. "
            "Alpha and Kp may admit numerical hits in the chosen grammar, "
            "but the frozen operators do not predict the muon anomaly at the required level."
        ),
        "",
        "## Tables",
        "",
        "- `outputs/tables/phase4_modular_alpha_hits.csv`",
        "- `outputs/tables/phase4_modular_kp_hits.csv`",
        "- `outputs/tables/phase4_modular_kp_wide_hits.csv`",
        "- `outputs/tables/phase4_modular_amu_oos.csv`",
    ]
    report_path = output_reports / "phase4_modular_scan_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
