"""Adversarial null evaluator for the Phase 4 modular expression grammar."""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

import phase4_modular_scan as phase4


def _combo_count(n_bases: int, n_coeffs: int, max_terms: int) -> int:
    return sum(math.comb(n_bases, k) * (n_coeffs**k) for k in range(1, max_terms + 1))


def _linear_values(
    bases: list[tuple[str, float, int]],
    max_terms: int,
    offsets: Iterable[float] = (0.0,),
) -> np.ndarray:
    coeffs = phase4.coefficient_bank()
    offsets = list(offsets)
    total = len(offsets) * _combo_count(len(bases), len(coeffs), max_terms)
    values = np.empty(total, dtype=np.float64)
    cursor = 0
    coeff_values = [float(item[1]) for item in coeffs]
    for offset in offsets:
        for n_terms in range(1, max_terms + 1):
            for basis_terms in itertools.combinations(bases, n_terms):
                basis_values = [float(term[1]) for term in basis_terms]
                for coeff_tuple in itertools.product(coeff_values, repeat=n_terms):
                    values[cursor] = offset + sum(
                        coeff * basis for coeff, basis in zip(coeff_tuple, basis_values)
                    )
                    cursor += 1
    return values


def _nearest_ppm(sorted_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sorted_values, targets)
    left = np.clip(indices - 1, 0, len(sorted_values) - 1)
    right = np.clip(indices, 0, len(sorted_values) - 1)
    nearest = np.minimum(np.abs(sorted_values[left] - targets), np.abs(sorted_values[right] - targets))
    return nearest / np.abs(targets) * 1_000_000.0


def _summarize_null(
    label: str,
    observed_ppm: float,
    threshold_ppm: float,
    random_ppm: np.ndarray,
) -> dict[str, float | str]:
    empirical_p = (1.0 + float(np.sum(random_ppm <= observed_ppm))) / (len(random_ppm) + 1.0)
    threshold_rate = float(np.mean(random_ppm <= threshold_ppm))
    return {
        "target": label,
        "observed_ppm": observed_ppm,
        "threshold_ppm": threshold_ppm,
        "n_random_targets": float(len(random_ppm)),
        "empirical_p_vs_observed": empirical_p,
        "random_hit_rate_at_threshold": threshold_rate,
        "random_ppm_median": float(np.median(random_ppm)),
        "random_ppm_p05": float(np.quantile(random_ppm, 0.05)),
        "random_ppm_p01": float(np.quantile(random_ppm, 0.01)),
        "random_ppm_min": float(np.min(random_ppm)),
    }


def run_adversarial_evaluation(n_random: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    alpha_hits = phase4.search_alpha(top_n=1)
    best_alpha = alpha_hits[0]
    kp_strict = phase4.search_kp(best_alpha.value, top_n=1, max_terms=3)[0]
    kp_wide = phase4.search_kp(best_alpha.value, top_n=1, max_terms=4)[0]

    alpha_values = phase4.MOD + _linear_values(phase4.alpha_basis(), max_terms=3)
    alpha_values = np.sort(alpha_values[np.isfinite(alpha_values)])
    alpha_random_targets = phase4.MOD + rng.uniform(0.0, 0.08, size=n_random)
    alpha_random_ppm = _nearest_ppm(alpha_values, alpha_random_targets)

    s_value = 1 / (best_alpha.value * 2 * math.pi)
    kp_offsets = [term[1] for term in phase4.kp_base_terms()]
    kp_bases = phase4.kp_correction_basis(s_value)
    kp_strict_values = best_alpha.value * _linear_values(kp_bases, max_terms=3, offsets=kp_offsets)
    kp_strict_values = np.sort(kp_strict_values[np.isfinite(kp_strict_values)])
    kp_wide_values = best_alpha.value * _linear_values(kp_bases, max_terms=4, offsets=kp_offsets)
    kp_wide_values = np.sort(kp_wide_values[np.isfinite(kp_wide_values)])
    kp_random_targets = rng.uniform(1800.0, 1870.0, size=n_random)
    kp_strict_random_ppm = _nearest_ppm(kp_strict_values, kp_random_targets)
    kp_wide_random_ppm = _nearest_ppm(kp_wide_values, kp_random_targets)

    rows = [
        _summarize_null("alpha_inv_local_delta", best_alpha.ppm_error, 0.01, alpha_random_ppm),
        _summarize_null("Kp_strict_local", kp_strict.ppm_error, 0.5, kp_strict_random_ppm),
        _summarize_null("Kp_wide_local", kp_wide.ppm_error, 0.5, kp_wide_random_ppm),
    ]
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, output_path: Path, n_random: int, seed: int) -> None:
    lines = [
        "# Phase 4 adversarial symbolic-regression evaluation",
        "",
        "This report estimates how often the same expression grammar hits random",
        "targets of comparable numerical scale. It is a look-elsewhere sanity check",
        "for Phase 4; low numerical error alone is not treated as physical evidence.",
        "",
        "## Null Setup",
        "",
        f"- random targets: `{n_random}`",
        f"- seed: `{seed}`",
        "- alpha null: `137 + Uniform(0, 0.08)`",
        "- Kp null: `Uniform(1800, 1870)`",
        "- grammar: Phase 4 coefficient bank from GF(137) signed residues and inverses",
        "- strict Kp: up to 3 correction terms",
        "- wide Kp: up to 4 correction terms",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Reading Rule",
        "",
        "- `empirical_p_vs_observed` answers: how often a random target is hit at least as well as the real target.",
        "- `random_hit_rate_at_threshold` answers: how often a random target passes the declared ppm cutoff.",
        "- A high random hit rate means the grammar is too dense to support a strong claim.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-targets", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--output-prefix",
        default="phase4_adversarial",
        help="Output stem under outputs/tables and outputs/reports.",
    )
    args = parser.parse_args()

    print("Phase 4 adversarial evaluator")
    print(f"random_targets={args.random_targets} seed={args.seed}")
    summary = run_adversarial_evaluation(args.random_targets, args.seed)
    table_path = Path("outputs/tables") / f"{args.output_prefix}_summary.csv"
    report_path = Path("outputs/reports") / f"{args.output_prefix}_report.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    write_report(summary, report_path, args.random_targets, args.seed)
    print(table_path)
    print(report_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
