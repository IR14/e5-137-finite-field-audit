"""Audit whether numerical hits can be explained by known approximations.

This is an anti-numerology diagnostic. It does not try to improve any model.
It asks whether comparable accuracy appears from:

1. continued-fraction/rational approximations with bounded denominators;
2. a small dictionary of common e/pi expressions with small integer coefficients;
3. random targets drawn from the same numerical ranges.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA_INV = 137.035999084
ELECTRON_MASS_MEV = 0.51099895069
MUON_MASS_MEV = 105.6583755
TAU_MASS_MEV = 1776.86
PROTON_ELECTRON_RATIO = 1836.15267343

N = 5
D = 26
Q = N * (N - 2) / (math.e**4 * math.pi**3)
T5 = 2 / (math.e * N)


@dataclass(frozen=True)
class Target:
    name: str
    value: float
    null_low: float
    null_high: float


@dataclass(frozen=True)
class ExpressionHit:
    expression: str
    value: float
    ppm_error: float


def ppm(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1e-300) * 1_000_000.0


def targets() -> list[Target]:
    muon_ratio = MUON_MASS_MEV / ELECTRON_MASS_MEV
    tau_ratio = TAU_MASS_MEV / ELECTRON_MASS_MEV
    return [
        Target("alpha_inv", ALPHA_INV, 137.0, 137.08),
        Target("delta_alpha_inv_minus_137", ALPHA_INV - 137.0, 0.001, 0.08),
        Target("muon_to_electron_mass_ratio", muon_ratio, 190.0, 220.0),
        Target("tau_to_electron_mass_ratio", tau_ratio, 3300.0, 3600.0),
        Target("proton_to_electron_mass_ratio", PROTON_ELECTRON_RATIO, 1800.0, 1870.0),
        Target("q_operator", Q, 0.001, 0.03),
        Target("T5_operator", T5, 0.05, 0.25),
    ]


def rational_best(value: float, max_denominator: int) -> tuple[Fraction, float, float]:
    fraction = Fraction(value).limit_denominator(max_denominator)
    approx = float(fraction)
    return fraction, approx, ppm(approx, value)


def rational_summary(
    target_items: list[Target],
    denominators: list[int],
    random_targets: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for target in target_items:
        random_values = rng.uniform(target.null_low, target.null_high, size=random_targets)
        for denominator in denominators:
            fraction, approx, observed_ppm = rational_best(target.value, denominator)
            random_ppm = np.array(
                [
                    rational_best(float(random_value), denominator)[2]
                    for random_value in random_values
                ],
                dtype=float,
            )
            rows.append(
                {
                    "target": target.name,
                    "value": f"{target.value:.17g}",
                    "max_denominator": denominator,
                    "best_fraction": f"{fraction.numerator}/{fraction.denominator}",
                    "best_value": f"{approx:.17g}",
                    "observed_ppm": observed_ppm,
                    "random_targets": random_targets,
                    "empirical_p_random_as_good": (
                        1.0 + float(np.sum(random_ppm <= observed_ppm))
                    )
                    / (len(random_ppm) + 1.0),
                    "random_ppm_median": float(np.median(random_ppm)),
                    "random_ppm_p05": float(np.quantile(random_ppm, 0.05)),
                    "random_ppm_p01": float(np.quantile(random_ppm, 0.01)),
                    "random_ppm_min": float(np.min(random_ppm)),
                }
            )
    return pd.DataFrame(rows)


def coefficient_bank() -> list[tuple[str, float]]:
    coeffs = [1, 2, 3, 5, 13, 26, 137]
    signed = sorted({float(c) for c in coeffs} | {-float(c) for c in coeffs} | {0.0})
    return [(f"{int(c)}", c) for c in signed]


def basis_bank() -> list[tuple[str, float]]:
    return [
        ("1", 1.0),
        ("e", math.e),
        ("pi", math.pi),
        ("pi/e", math.pi / math.e),
        ("e/pi", math.e / math.pi),
        ("pi-e", math.pi - math.e),
        ("e*pi", math.e * math.pi),
        ("pi^2", math.pi**2),
        ("e^2", math.e**2),
        ("e^pi", math.e**math.pi),
        ("pi^e", math.pi**math.e),
        ("log(137)", math.log(137)),
        ("sqrt(137)", math.sqrt(137)),
        ("1/e", 1 / math.e),
        ("1/pi", 1 / math.pi),
    ]


def expression_values(max_terms: int) -> tuple[np.ndarray, list[str]]:
    coeffs = coefficient_bank()
    bases = basis_bank()
    values: list[float] = []
    labels: list[str] = []

    # Constant offsets are included as the zero-term layer.
    for coeff_label, coeff_value in coeffs:
        values.append(coeff_value)
        labels.append(coeff_label)

    for n_terms in range(1, max_terms + 1):
        for basis_terms in itertools.combinations(bases, n_terms):
            for coeff_terms in itertools.product(coeffs, repeat=n_terms):
                if all(coeff_value == 0.0 for _, coeff_value in coeff_terms):
                    continue
                value = 0.0
                parts: list[str] = []
                for (basis_label, basis_value), (coeff_label, coeff_value) in zip(
                    basis_terms, coeff_terms
                ):
                    if coeff_value == 0.0:
                        continue
                    value += coeff_value * basis_value
                    parts.append(f"{coeff_label}*{basis_label}")
                if not math.isfinite(value):
                    continue
                values.append(value)
                labels.append(" + ".join(parts))

    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    return array[order], [labels[index] for index in order]


def nearest_expression(
    sorted_values: np.ndarray,
    sorted_labels: list[str],
    target: float,
) -> ExpressionHit:
    index = int(np.searchsorted(sorted_values, target))
    candidates = [max(0, index - 1), min(len(sorted_values) - 1, index)]
    best_index = min(candidates, key=lambda candidate: abs(sorted_values[candidate] - target))
    return ExpressionHit(
        expression=sorted_labels[best_index],
        value=float(sorted_values[best_index]),
        ppm_error=ppm(float(sorted_values[best_index]), target),
    )


def dictionary_summary(
    target_items: list[Target],
    max_terms: int,
    random_targets: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_values, sorted_labels = expression_values(max_terms=max_terms)
    rng = np.random.default_rng(seed)
    hit_rows: list[dict[str, float | int | str]] = []
    null_rows: list[dict[str, float | int | str]] = []

    for target in target_items:
        observed = nearest_expression(sorted_values, sorted_labels, target.value)
        random_values = rng.uniform(target.null_low, target.null_high, size=random_targets)
        random_ppm = np.array(
            [
                nearest_expression(sorted_values, sorted_labels, float(random_value)).ppm_error
                for random_value in random_values
            ],
            dtype=float,
        )
        hit_rows.append(
            {
                "target": target.name,
                "value": f"{target.value:.17g}",
                "best_expression": observed.expression,
                "best_value": f"{observed.value:.17g}",
                "observed_ppm": observed.ppm_error,
                "max_terms": max_terms,
            }
        )
        null_rows.append(
            {
                "target": target.name,
                "observed_ppm": observed.ppm_error,
                "random_targets": random_targets,
                "empirical_p_random_as_good": (
                    1.0 + float(np.sum(random_ppm <= observed.ppm_error))
                )
                / (len(random_ppm) + 1.0),
                "random_ppm_median": float(np.median(random_ppm)),
                "random_ppm_p05": float(np.quantile(random_ppm, 0.05)),
                "random_ppm_p01": float(np.quantile(random_ppm, 0.01)),
                "random_ppm_min": float(np.min(random_ppm)),
            }
        )

    return pd.DataFrame(hit_rows), pd.DataFrame(null_rows)


def write_report(
    path: Path,
    rational_frame: pd.DataFrame,
    dictionary_hits: pd.DataFrame,
    dictionary_null: pd.DataFrame,
    denominators: list[int],
    random_targets: int,
    max_terms: int,
) -> None:
    latest_rational = rational_frame[
        rational_frame["max_denominator"] == max(denominators)
    ].copy()
    lines = [
        "# Known-Approximation Audit",
        "",
        "This is an anti-numerology diagnostic. It asks whether numerical",
        "agreement can be reproduced by ordinary continued-fraction approximations",
        "or by a small dictionary of common `e`/`pi` expressions with small integer",
        "coefficients.",
        "",
        "The audit does not validate or refute a physical model by itself. Its role",
        "is to flag when ppm-level agreement is not surprising under simple",
        "approximation mechanisms.",
        "",
        "## Setup",
        "",
        f"- random targets per range: `{random_targets}`",
        f"- continued-fraction denominator grid: `{denominators}`",
        f"- dictionary terms per expression: up to `{max_terms}`",
        "- dictionary basis: `1`, `e`, `pi`, `pi/e`, `e/pi`, `pi-e`, `e*pi`,",
        "  `pi^2`, `e^2`, `e^pi`, `pi^e`, `log(137)`, `sqrt(137)`, `1/e`, `1/pi`",
        "- integer coefficients: `0`, `±1`, `±2`, `±3`, `±5`, `±13`, `±26`, `±137`",
        "",
        "## Continued-Fraction Summary",
        "",
        "Best rational approximations at the largest tested denominator bound:",
        "",
        latest_rational[
            [
                "target",
                "best_fraction",
                "observed_ppm",
                "empirical_p_random_as_good",
                "random_ppm_median",
                "random_ppm_p01",
            ]
        ].to_markdown(index=False),
        "",
        "## e/pi Dictionary Hits",
        "",
        dictionary_hits.to_markdown(index=False),
        "",
        "## e/pi Dictionary Random-Target Null",
        "",
        dictionary_null.to_markdown(index=False),
        "",
        "## Interpretation Rule",
        "",
        "- A high `empirical_p_random_as_good` means random targets are hit about as",
        "  well as the real target, so the approximation is not strong evidence.",
        "- A low value means the target is less easily matched by this particular",
        "  simple approximation family. This still does not prove physics; it only",
        "  survives this narrow anti-numerology filter.",
        "- Continued fractions are intentionally a harsh warning: arbitrary rational",
        "  coefficients become very powerful very quickly.",
        "",
        "## Practical Consequence",
        "",
        "Any future symbolic-regression claim should report a comparable null-rate",
        "against random targets and a complexity penalty. Sub-ppm agreement alone",
        "is not a reliable discovery criterion.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-targets", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-terms", type=int, default=2)
    parser.add_argument(
        "--denominator",
        action="append",
        type=int,
        dest="denominators",
        default=[10, 100, 1000, 10_000, 100_000],
    )
    parser.add_argument("--output-prefix", default="known_approximation_audit")
    args = parser.parse_args()

    target_items = targets()
    rational_frame = rational_summary(
        target_items,
        denominators=args.denominators,
        random_targets=args.random_targets,
        seed=args.seed,
    )
    dictionary_hits, dictionary_null = dictionary_summary(
        target_items,
        max_terms=args.max_terms,
        random_targets=args.random_targets,
        seed=args.seed + 1,
    )

    tables_dir = Path("outputs/tables")
    reports_dir = Path("outputs/reports")
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rational_path = tables_dir / f"{args.output_prefix}_continued_fractions.csv"
    dictionary_hits_path = tables_dir / f"{args.output_prefix}_dictionary_hits.csv"
    dictionary_null_path = tables_dir / f"{args.output_prefix}_dictionary_null.csv"
    report_path = reports_dir / f"{args.output_prefix}.md"

    rational_frame.to_csv(rational_path, index=False)
    dictionary_hits.to_csv(dictionary_hits_path, index=False)
    dictionary_null.to_csv(dictionary_null_path, index=False)
    write_report(
        report_path,
        rational_frame=rational_frame,
        dictionary_hits=dictionary_hits,
        dictionary_null=dictionary_null,
        denominators=args.denominators,
        random_targets=args.random_targets,
        max_terms=args.max_terms,
    )

    print(rational_path)
    print(dictionary_hits_path)
    print(dictionary_null_path)
    print(report_path)
    print(dictionary_null.to_string(index=False))


if __name__ == "__main__":
    main()
