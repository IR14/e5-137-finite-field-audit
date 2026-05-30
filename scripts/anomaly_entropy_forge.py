"""Phase 5 anomaly scan over base-e digit collisions and modular metrics.

This script is deliberately exploratory. It searches for rare digit patterns,
then tests whether the resulting candidates survive simple out-of-sample
checks. A hit is reported as a candidate only; it is not interpreted as a proof
of a physical or number-theoretic principle.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "phase5_anomaly_forge.md"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

ELECTRON_MASS_MEV = 0.51099895069
HIGGS_MASS_MEV = 125_250.0
TOP_MASS_MEV = 172_500.0

N = 5
D = 26
I5 = 42
P_ADIC = 137
Q = N * (N - 2) / (math.e**4 * math.pi**3)
T5 = 2 / (math.e * N)
F_TAU = I5 + 2 * math.exp(-2) + 2 * math.exp(-7)


@dataclass(frozen=True)
class BaseEExpansion:
    max_exponent: int
    integer_digits: tuple[int, ...]
    fractional_digits: tuple[int, ...]
    residual: Decimal

    @property
    def integer_string(self) -> str:
        return "".join(str(digit) for digit in self.integer_digits)

    @property
    def fractional_string(self) -> str:
        return "".join(str(digit) for digit in self.fractional_digits)

    @property
    def compact(self) -> str:
        return f"{self.integer_string}.{self.fractional_string}_e"


def ppm_error(value: float, target: float) -> float:
    return (value - target) / target * 1_000_000.0


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    limit = math.isqrt(value)
    factor = 5
    while factor <= limit:
        if value % factor == 0 or value % (factor + 2) == 0:
            return False
        factor += 6
    return True


def sieve_primes(start: int, stop: int) -> list[int]:
    if stop < 2:
        return []
    sieve = np.ones(stop + 1, dtype=bool)
    sieve[:2] = False
    for value in range(2, math.isqrt(stop) + 1):
        if sieve[value]:
            sieve[value * value : stop + 1 : value] = False
    return [int(value) for value in np.flatnonzero(sieve) if value >= start]


def precompute_e_weights(max_exponent: int, frac_digits: int, precision: int) -> dict[int, Decimal]:
    with localcontext() as ctx:
        ctx.prec = precision
        return {
            exponent: +Decimal(exponent).exp()
            for exponent in range(max_exponent, -frac_digits - 1, -1)
        }


def encode_decimal_base_e(
    value: Decimal,
    *,
    frac_digits: int,
    precision: int,
    weights: dict[int, Decimal] | None = None,
) -> BaseEExpansion:
    if value < 0:
        raise ValueError("Only non-negative values are supported.")
    if value == 0:
        return BaseEExpansion(
            max_exponent=0,
            integer_digits=(0,),
            fractional_digits=tuple(0 for _ in range(frac_digits)),
            residual=Decimal(0),
        )

    max_exponent = int(math.floor(math.log(float(value))))
    local_weights = weights
    if local_weights is None:
        local_weights = precompute_e_weights(max_exponent, frac_digits, precision)

    with localcontext() as ctx:
        ctx.prec = precision
        remainder = +value
        integer_digits: list[int] = []
        fractional_digits: list[int] = []
        for exponent in range(max_exponent, -frac_digits - 1, -1):
            weight = local_weights[exponent]
            digit = int((remainder / weight).to_integral_value(rounding=ROUND_FLOOR))
            if digit < 0 or digit > 2:
                raise ArithmeticError(
                    f"Digit outside base-e alphabet: value={value}, "
                    f"exponent={exponent}, digit={digit}"
                )
            remainder -= Decimal(digit) * weight
            if exponent >= 0:
                integer_digits.append(digit)
            else:
                fractional_digits.append(digit)
        return BaseEExpansion(
            max_exponent=max_exponent,
            integer_digits=tuple(integer_digits),
            fractional_digits=tuple(fractional_digits),
            residual=+remainder,
        )


def scan_brothers_137(
    *,
    start: int,
    stop: int,
    frac_digits: int,
    precision: int,
) -> pd.DataFrame:
    max_exponent = int(math.floor(math.log(stop)))
    weights = precompute_e_weights(max_exponent, frac_digits, precision)
    rows: list[dict[str, int | str]] = []
    for prime in sieve_primes(start, stop):
        expansion = encode_decimal_base_e(
            Decimal(prime),
            frac_digits=frac_digits,
            precision=precision,
            weights=weights,
        )
        digits = expansion.fractional_digits
        if digits[1] == 2 and digits[N + 1] == 2:
            rows.append(
                {
                    "prime": prime,
                    "max_exponent": expansion.max_exponent,
                    "integer_digits": expansion.integer_string,
                    "fractional_prefix": expansion.fractional_string,
                    "d_minus_2": digits[1],
                    "d_minus_7": digits[N + 1],
                }
            )
    return pd.DataFrame(rows)


def resonance_rows(collisions: pd.DataFrame) -> pd.DataFrame:
    if collisions.empty:
        return pd.DataFrame()

    formulas = {
        "electron_alpha_like": lambda alpha_like: ELECTRON_MASS_MEV * alpha_like,
        "muon_generation_like": lambda alpha_like: (
            ELECTRON_MASS_MEV * alpha_like * (1.5 + Q)
        ),
        "tau_generation_like": lambda alpha_like: (
            ELECTRON_MASS_MEV * alpha_like * (25.0 + Q * F_TAU)
        ),
    }
    targets = {
        "higgs_default_125p25_GeV": HIGGS_MASS_MEV,
        "top_default_172p5_GeV": TOP_MASS_MEV,
    }

    rows: list[dict[str, float | int | str | bool]] = []
    for prime in collisions["prime"].astype(int).to_numpy():
        for formula_name, formula in formulas.items():
            value = float(formula(float(prime)))
            for target_name, target in targets.items():
                signed_ppm = ppm_error(value, target)
                rows.append(
                    {
                        "prime": int(prime),
                        "formula": formula_name,
                        "target": target_name,
                        "predicted_mass_MeV": value,
                        "target_mass_MeV": target,
                        "signed_ppm": signed_ppm,
                        "abs_ppm": abs(signed_ppm),
                        "passes_100ppm": abs(signed_ppm) < 100.0,
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_ppm")


def golden_ratio_decimal(precision: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = precision
        return (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)


def phi_digit_analysis(*, frac_digits: int, precision: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    phi = golden_ratio_decimal(precision)
    expansion = encode_decimal_base_e(phi, frac_digits=frac_digits, precision=precision)
    digits = list(expansion.fractional_digits)
    digit_rows = [
        {
            "position": -index,
            "digit": digit,
            "prefix_sum": int(sum(digits[:index])),
            "prefix_sum_mod_5": int(sum(digits[:index]) % 5),
            "prefix_sum_mod_42": int(sum(digits[:index]) % 42),
            "prefix_sum_mod_137": int(sum(digits[:index]) % 137),
        }
        for index, digit in enumerate(digits, start=1)
    ]

    block_rows: list[dict[str, int | str]] = []
    for block_id, start in enumerate(range(0, frac_digits, I5), start=1):
        block = digits[start : start + I5]
        weighted_local = sum((offset + 1) ** 2 * digit for offset, digit in enumerate(block))
        weighted_global = sum((start + offset + 1) ** 2 * digit for offset, digit in enumerate(block))
        block_rows.append(
            {
                "block": block_id,
                "positions": f"{start + 1}-{start + len(block)}",
                "length": len(block),
                "digit_string": "".join(str(digit) for digit in block),
                "sum_digits": int(sum(block)),
                "sum_mod_5": int(sum(block) % 5),
                "sum_mod_42": int(sum(block) % 42),
                "sum_mod_137": int(sum(block) % 137),
                "weighted_local": int(weighted_local),
                "weighted_local_mod_5": int(weighted_local % 5),
                "weighted_local_mod_42": int(weighted_local % 42),
                "weighted_local_mod_137": int(weighted_local % 137),
                "weighted_global": int(weighted_global),
                "weighted_global_mod_5": int(weighted_global % 5),
                "weighted_global_mod_42": int(weighted_global % 42),
                "weighted_global_mod_137": int(weighted_global % 137),
            }
        )
    return pd.DataFrame(digit_rows), pd.DataFrame(block_rows), expansion.compact


def p_adic_valuation(value: int, prime: int) -> int:
    if prime <= 1:
        raise ValueError("prime must exceed 1.")
    value = abs(int(value))
    if value == 0:
        return math.inf  # type: ignore[return-value]
    valuation = 0
    while value % prime == 0:
        valuation += 1
        value //= prime
    return valuation


def p_adic_norm(value: int, prime: int) -> float:
    valuation = p_adic_valuation(value, prime)
    if valuation == math.inf:
        return 0.0
    return float(prime ** (-valuation))


def phase_level_matrix() -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for d_index in range(1, D + 1):
        for n_index in range(1, N + 1):
            # A fixed integer phase lattice. It avoids putting 137 directly into
            # the level definition so that 137-adic divisibility is testable.
            level = D * d_index**2 + N * n_index**2 + d_index * n_index
            rows.append(
                {
                    "d_index": d_index,
                    "n_index": n_index,
                    "level": level,
                    "level_mod_137": level % P_ADIC,
                }
            )
    return pd.DataFrame(rows)


def p_adic_distance_summary(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = levels["level"].astype(int).to_numpy()
    rows: list[dict[str, float | int]] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            delta = int(values[right] - values[left])
            valuation = p_adic_valuation(delta, P_ADIC)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "v137": valuation,
                    "norm137": p_adic_norm(delta, P_ADIC),
                    "zero_mod_137": valuation >= 1,
                }
            )
    distances = pd.DataFrame(rows)
    distribution = (
        distances["v137"]
        .value_counts()
        .rename_axis("v137")
        .reset_index(name="pair_count")
        .sort_values("v137")
    )
    distribution["fraction"] = distribution["pair_count"] / len(distances)
    return distances, distribution


def closest_by_group(resonances: pd.DataFrame) -> pd.DataFrame:
    if resonances.empty:
        return resonances
    return (
        resonances.sort_values("abs_ppm")
        .groupby(["formula", "target"], as_index=False)
        .head(1)
        .sort_values("abs_ppm")
    )


def table_or_note(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.loc[:, columns].head(max_rows).to_markdown(index=False, floatfmt=".6g")


def save_figures(
    collisions: pd.DataFrame,
    resonances: pd.DataFrame,
    phi_digits: pd.DataFrame,
    padic_distribution: pd.DataFrame,
) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "collision_residuals": FIGURE_DIR / "phase5_collision_residuals.png",
        "phi_digits": FIGURE_DIR / "phase5_phi_base_e_digits.png",
        "padic_distribution": FIGURE_DIR / "phase5_padic_valuation_distribution.png",
    }

    plt.figure(figsize=(9, 5))
    if not resonances.empty:
        view = resonances[resonances["formula"].isin(["muon_generation_like", "tau_generation_like"])]
        for (formula, target), group in view.groupby(["formula", "target"]):
            plt.scatter(group["prime"], group["signed_ppm"], s=10, alpha=0.35, label=f"{formula} -> {target}")
        plt.axhline(100, color="black", linewidth=0.8, linestyle="--")
        plt.axhline(-100, color="black", linewidth=0.8, linestyle="--")
        plt.yscale("symlog", linthresh=100)
        plt.xlabel("collision prime")
        plt.ylabel("signed ppm residual")
        plt.legend(fontsize=7)
    else:
        plt.text(0.5, 0.5, "No collisions", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(paths["collision_residuals"], dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.step(phi_digits["position"].abs(), phi_digits["digit"], where="mid", linewidth=1.2)
    plt.yticks([0, 1, 2])
    plt.xlabel("fractional digit index k in d_-k")
    plt.ylabel("digit")
    plt.title("Golden ratio base-e fractional digits")
    plt.tight_layout()
    plt.savefig(paths["phi_digits"], dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(padic_distribution["v137"].astype(str), padic_distribution["pair_count"])
    plt.xlabel("v_137(delta)")
    plt.ylabel("pair count")
    plt.title("137-adic valuation distribution on D=26, N=5 lattice")
    plt.tight_layout()
    plt.savefig(paths["padic_distribution"], dpi=160)
    plt.close()

    return paths


def render_report(
    *,
    args: argparse.Namespace,
    collisions: pd.DataFrame,
    resonances: pd.DataFrame,
    phi_compact: str,
    phi_blocks: pd.DataFrame,
    levels: pd.DataFrame,
    padic_distribution: pd.DataFrame,
    figure_paths: dict[str, Path],
    output_paths: Iterable[Path],
) -> str:
    closest = closest_by_group(resonances)
    passing = resonances[resonances["passes_100ppm"]] if not resonances.empty else pd.DataFrame()
    total_primes = len(sieve_primes(args.start, args.stop))
    collision_fraction = len(collisions) / total_primes if total_primes else 0.0
    v137_positive = int(padic_distribution.loc[padic_distribution["v137"] >= 1, "pair_count"].sum())
    total_pairs = int(padic_distribution["pair_count"].sum())

    lines = [
        "# Phase 5 Anomaly Forge",
        "",
        "## Scope",
        "",
        "This is an exploratory anomaly and outlier scan. It searches for rare base-e "
        "digit patterns and modular coincidences, then reports whether the requested "
        "hard thresholds are actually met. Null or weak outcomes are retained.",
        "",
        "## Constants and Operators",
        "",
        f"- N = {N}",
        f"- D = {D}",
        f"- I5 = {I5}",
        f"- q = {Q:.12g}",
        f"- T5 = {T5:.12g}",
        f"- f_tau = 42 + 2e^-2 + 2e^-7 = {F_TAU:.12g}",
        f"- Higgs reference mass used here: {HIGGS_MASS_MEV / 1000:.3f} GeV",
        f"- Top reference mass used here: {TOP_MASS_MEV / 1000:.3f} GeV",
        "",
        "## Vector 1: Brothers 137",
        "",
        f"Scanned primes in [{args.start}, {args.stop}] and selected primes with "
        "`d_-2 = 2` and `d_-7 = 2` in the greedy base-e expansion.",
        "",
        f"- Total primes scanned: {total_primes}",
        f"- Collisions found: {len(collisions)}",
        f"- Collision fraction: {collision_fraction:.6f}",
        "",
        "First collision rows:",
        "",
        table_or_note(
            collisions,
            ["prime", "integer_digits", "fractional_prefix", "d_minus_2", "d_minus_7"],
            max_rows=15,
        ),
        "",
        "Closest mass-resonance checks after substituting each collision prime as an "
        "alpha-like scale in the frozen lepton-generation formulas:",
        "",
        table_or_note(
            closest,
            [
                "prime",
                "formula",
                "target",
                "predicted_mass_MeV",
                "target_mass_MeV",
                "signed_ppm",
                "abs_ppm",
                "passes_100ppm",
            ],
            max_rows=12,
        ),
        "",
    ]

    if passing.empty:
        lines.extend(
            [
                "No tested collision passed the requested <100 ppm mass-resonance threshold.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Mass-resonance candidates passing <100 ppm:",
                "",
                table_or_note(
                    passing,
                    [
                        "prime",
                        "formula",
                        "target",
                        "predicted_mass_MeV",
                        "target_mass_MeV",
                        "signed_ppm",
                        "abs_ppm",
                    ],
                    max_rows=30,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Vector 2: Golden Ratio Base-e DNA Matrix",
            "",
            f"Golden ratio base-e prefix: `{phi_compact}`",
            "",
            "Block slicing with step I5 = 42:",
            "",
            table_or_note(
                phi_blocks,
                [
                    "block",
                    "positions",
                    "length",
                    "sum_digits",
                    "sum_mod_5",
                    "sum_mod_42",
                    "sum_mod_137",
                    "weighted_local_mod_5",
                    "weighted_local_mod_42",
                    "weighted_local_mod_137",
                ],
                max_rows=10,
            ),
            "",
            "The block residues do not form a stable repeated residue class across the "
            "available 100-digit prefix. This is a null result for a simple Z/42Z "
            "stabilizer in this scan.",
            "",
            "## Vector 3: 137-adic String Metric",
            "",
            "A fixed D=26 by N=5 integer phase lattice was evaluated with the 137-adic "
            "valuation v_137(delta) over all pairwise level differences. The level "
            "definition does not insert 137 directly, so divisibility by 137 remains a "
            "testable property rather than a construction artifact.",
            "",
            f"- Levels: {len(levels)}",
            f"- Pairwise distances: {total_pairs}",
            f"- Pairs with v_137(delta) >= 1: {v137_positive}",
            f"- Fraction divisible by 137: {v137_positive / total_pairs:.6f}",
            "",
            table_or_note(padic_distribution, ["v137", "pair_count", "fraction"], max_rows=10),
            "",
            "No high-order 137-adic collapse was observed in this fixed lattice. The "
            "dominant class is v_137 = 0, as expected for generic integer differences.",
            "",
            "## Figures",
            "",
            f"- Collision residuals: `{figure_paths['collision_residuals']}`",
            f"- Golden-ratio digit trace: `{figure_paths['phi_digits']}`",
            f"- 137-adic valuation distribution: `{figure_paths['padic_distribution']}`",
            "",
            "## Output Tables",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in output_paths])
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Phase 5 found rare digit collisions by construction, but the downstream tests "
            "did not produce a robust physical resonance under the stated thresholds. The "
            "golden-ratio and 137-adic branches also returned null results for the simple "
            "invariants tested here. The conservative interpretation is that these scans "
            "do not currently support a new discrete invariant.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=137)
    parser.add_argument("--stop", type=int, default=200_000)
    parser.add_argument("--collision-frac-digits", type=int, default=12)
    parser.add_argument("--phi-frac-digits", type=int, default=100)
    parser.add_argument("--precision", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Phase 5 anomaly entropy forge")
    print(
        "prime_scan=[%d,%d] collision_digits=%d phi_digits=%d precision=%d"
        % (
            args.start,
            args.stop,
            args.collision_frac_digits,
            args.phi_frac_digits,
            args.precision,
        )
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("scanner 1/3: brothers-137 digit collisions...")
    collisions = scan_brothers_137(
        start=args.start,
        stop=args.stop,
        frac_digits=args.collision_frac_digits,
        precision=args.precision,
    )
    resonances = resonance_rows(collisions)

    print("scanner 2/3: golden-ratio base-e DNA matrix...")
    phi_digits, phi_blocks, phi_compact = phi_digit_analysis(
        frac_digits=args.phi_frac_digits,
        precision=args.precision,
    )

    print("scanner 3/3: 137-adic phase lattice...")
    levels = phase_level_matrix()
    padic_distances, padic_distribution = p_adic_distance_summary(levels)

    table_paths = {
        "collisions": TABLE_DIR / "phase5_brothers137_collisions.csv",
        "resonances": TABLE_DIR / "phase5_mass_resonance_candidates.csv",
        "phi_digits": TABLE_DIR / "phase5_phi_base_e_digits.csv",
        "phi_blocks": TABLE_DIR / "phase5_phi_blocks_i5.csv",
        "padic_levels": TABLE_DIR / "phase5_padic_levels.csv",
        "padic_distances": TABLE_DIR / "phase5_padic_distances.csv",
        "padic_distribution": TABLE_DIR / "phase5_padic_valuation_distribution.csv",
    }
    collisions.to_csv(table_paths["collisions"], index=False)
    resonances.to_csv(table_paths["resonances"], index=False)
    phi_digits.to_csv(table_paths["phi_digits"], index=False)
    phi_blocks.to_csv(table_paths["phi_blocks"], index=False)
    levels.to_csv(table_paths["padic_levels"], index=False)
    padic_distances.to_csv(table_paths["padic_distances"], index=False)
    padic_distribution.to_csv(table_paths["padic_distribution"], index=False)

    print("saving diagnostic figures...")
    figure_paths = save_figures(collisions, resonances, phi_digits, padic_distribution)

    report = render_report(
        args=args,
        collisions=collisions,
        resonances=resonances,
        phi_compact=phi_compact,
        phi_blocks=phi_blocks,
        levels=levels,
        padic_distribution=padic_distribution,
        figure_paths=figure_paths,
        output_paths=table_paths.values(),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    passing = int(resonances["passes_100ppm"].sum()) if not resonances.empty else 0
    best = resonances.iloc[0] if not resonances.empty else None
    print(f"collisions_found={len(collisions)}")
    if best is not None:
        print(
            "best_resonance prime=%s formula=%s target=%s abs_ppm=%.6g"
            % (best["prime"], best["formula"], best["target"], best["abs_ppm"])
        )
    print(f"mass_resonances_under_100ppm={passing}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
