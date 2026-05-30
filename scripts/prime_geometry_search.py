"""Base-e digit geometry audit for primes versus composites.

This is an exploratory statistical diagnostic, not a proof strategy for the
Riemann hypothesis. It tests whether a fixed base-e digit encoder produces
simple distributional differences between primes and composites in a bounded
integer window.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "prime_geometry_base_e.md"
DEFAULT_FEATURES = PROJECT_ROOT / "outputs" / "tables" / "prime_geometry_base_e_features.csv"
DEFAULT_TESTS = PROJECT_ROOT / "outputs" / "tables" / "prime_geometry_base_e_mannwhitney.csv"


@dataclass(frozen=True)
class BaseEDigits:
    """Greedy beta-expansion digits for base e with alphabet {0, 1, 2}."""

    number: int
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


@dataclass(frozen=True)
class MetricTest:
    metric: str
    prime_median: float
    composite_median: float
    delta_median: float
    u_statistic: float
    rank_biserial: float
    p_value: float
    bonferroni_p_value: float
    significant_uncorrected: bool
    significant_bonferroni: bool


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    limit = int(math.isqrt(value))
    factor = 5
    while factor <= limit:
        if value % factor == 0 or value % (factor + 2) == 0:
            return False
        factor += 6
    return True


def primes_in_range(start: int, stop: int) -> list[int]:
    return [value for value in range(start, stop + 1) if is_prime(value)]


def composites_in_range(start: int, stop: int) -> list[int]:
    return [value for value in range(start, stop + 1) if value > 1 and not is_prime(value)]


def encode_base_e(number: int, *, frac_digits: int = 50, precision: int = 120) -> BaseEDigits:
    """Encode an integer into a greedy positional expansion in base e.

    For beta=e, the greedy beta-expansion uses digits in
    ``{0, ..., floor(beta)}``, i.e. ``{0, 1, 2}``. The returned fractional part
    is truncated after ``frac_digits`` places.
    """

    if number < 0:
        raise ValueError("Only non-negative integers are supported.")
    if frac_digits < 0:
        raise ValueError("frac_digits must be non-negative.")
    if precision <= frac_digits + 20:
        raise ValueError("precision should exceed frac_digits by a comfortable margin.")

    if number == 0:
        return BaseEDigits(
            number=0,
            max_exponent=0,
            integer_digits=(0,),
            fractional_digits=tuple(0 for _ in range(frac_digits)),
            residual=Decimal(0),
        )

    max_exponent = int(math.floor(math.log(number)))
    with localcontext() as ctx:
        ctx.prec = precision
        remainder = Decimal(number)
        integer_digits: list[int] = []
        fractional_digits_out: list[int] = []

        for exponent in range(max_exponent, -frac_digits - 1, -1):
            weight = Decimal(exponent).exp()
            digit = int((remainder / weight).to_integral_value(rounding=ROUND_FLOOR))
            if digit < 0 or digit > 2:
                raise ArithmeticError(
                    f"Base-e digit outside alphabet {{0,1,2}}: "
                    f"number={number}, exponent={exponent}, digit={digit}"
                )
            remainder -= Decimal(digit) * weight
            if exponent >= 0:
                integer_digits.append(digit)
            else:
                fractional_digits_out.append(digit)

        return BaseEDigits(
            number=number,
            max_exponent=max_exponent,
            integer_digits=tuple(integer_digits),
            fractional_digits=tuple(fractional_digits_out),
            residual=+remainder,
        )


def shannon_entropy(digits: np.ndarray) -> float:
    counts = np.bincount(digits, minlength=3).astype(float)
    probabilities = counts[counts > 0] / float(len(digits))
    return float(-np.sum(probabilities * np.log2(probabilities)))


def max_run_length(digits: np.ndarray, digit: int) -> int:
    best = 0
    current = 0
    for item in digits:
        if int(item) == digit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def autocorrelation(digits: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(digits):
        raise ValueError("lag must be between 1 and len(digits)-1.")
    centered = digits.astype(float) - float(np.mean(digits))
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.dot(centered[:-lag], centered[lag:]))
    return numerator / denominator


def extract_features(
    number: int,
    label: str,
    *,
    frac_digits: int,
    precision: int,
    max_lag: int,
) -> dict[str, float | int | str]:
    encoded = encode_base_e(number, frac_digits=frac_digits, precision=precision)
    fractional = np.asarray(encoded.fractional_digits, dtype=np.int8)
    row: dict[str, float | int | str] = {
        "class": label,
        "number": number,
        "max_exponent": encoded.max_exponent,
        "integer_digits": encoded.integer_string,
        "fractional_digits": encoded.fractional_string,
        "base_e_compact": encoded.compact,
        "entropy": shannon_entropy(fractional),
        "max_run_0": max_run_length(fractional, 0),
        "max_run_2": max_run_length(fractional, 2),
        "max_run_any": max(max_run_length(fractional, digit) for digit in (0, 1, 2)),
        "count_0": int(np.sum(fractional == 0)),
        "count_1": int(np.sum(fractional == 1)),
        "count_2": int(np.sum(fractional == 2)),
    }
    for lag in range(1, max_lag + 1):
        row[f"autocorr_lag_{lag}"] = autocorrelation(fractional, lag)
    return row


def build_samples(
    *,
    start: int,
    stop: int,
    n_primes: int,
    n_composites: int,
    seed: int,
    matched_composite_window: bool,
) -> tuple[list[int], list[int]]:
    primes = primes_in_range(start, stop)
    if len(primes) < n_primes:
        raise ValueError(f"Only found {len(primes)} primes in [{start}, {stop}], need {n_primes}.")
    selected_primes = primes[:n_primes]

    composite_start = min(selected_primes) if matched_composite_window else start
    composite_stop = max(selected_primes) if matched_composite_window else stop
    composites = composites_in_range(composite_start, composite_stop)
    if len(composites) < n_composites:
        raise ValueError(
            f"Only found {len(composites)} composites in "
            f"[{composite_start}, {composite_stop}], need {n_composites}."
        )

    rng = np.random.default_rng(seed)
    selected_composites = sorted(
        int(value) for value in rng.choice(composites, size=n_composites, replace=False)
    )
    return selected_primes, selected_composites


def feature_table(
    primes: list[int],
    composites: list[int],
    *,
    frac_digits: int,
    precision: int,
    max_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for number in primes:
        rows.append(
            extract_features(
                number,
                "prime",
                frac_digits=frac_digits,
                precision=precision,
                max_lag=max_lag,
            )
        )
    for number in composites:
        rows.append(
            extract_features(
                number,
                "composite",
                frac_digits=frac_digits,
                precision=precision,
                max_lag=max_lag,
            )
        )
    return pd.DataFrame(rows)


def mann_whitney_tests(
    features: pd.DataFrame,
    *,
    alpha: float,
    max_lag: int,
) -> pd.DataFrame:
    metrics = [
        "entropy",
        "max_run_0",
        "max_run_2",
        "max_run_any",
        "count_0",
        "count_1",
        "count_2",
    ] + [f"autocorr_lag_{lag}" for lag in range(1, max_lag + 1)]

    prime_rows = features[features["class"] == "prime"]
    composite_rows = features[features["class"] == "composite"]
    n_prime = len(prime_rows)
    n_composite = len(composite_rows)
    correction_count = len(metrics)
    tests: list[MetricTest] = []

    for metric in metrics:
        prime_values = prime_rows[metric].astype(float).to_numpy()
        composite_values = composite_rows[metric].astype(float).to_numpy()
        result = mannwhitneyu(prime_values, composite_values, alternative="two-sided")
        p_value = float(result.pvalue)
        bonferroni_p = min(1.0, p_value * correction_count)
        rank_biserial = (2.0 * float(result.statistic) / (n_prime * n_composite)) - 1.0
        prime_median = float(np.median(prime_values))
        composite_median = float(np.median(composite_values))
        tests.append(
            MetricTest(
                metric=metric,
                prime_median=prime_median,
                composite_median=composite_median,
                delta_median=prime_median - composite_median,
                u_statistic=float(result.statistic),
                rank_biserial=rank_biserial,
                p_value=p_value,
                bonferroni_p_value=bonferroni_p,
                significant_uncorrected=p_value < alpha,
                significant_bonferroni=bonferroni_p < alpha,
            )
        )

    return pd.DataFrame([test.__dict__ for test in tests]).sort_values("p_value")


def markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int | None = None) -> str:
    view = frame.loc[:, columns]
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_markdown(index=False, floatfmt=".6g")


def render_report(
    *,
    args: argparse.Namespace,
    features: pd.DataFrame,
    tests: pd.DataFrame,
    primes: list[int],
    composites: list[int],
    report_path: Path,
    feature_path: Path,
    tests_path: Path,
) -> str:
    significant_uncorrected = tests[tests["significant_uncorrected"]]
    significant_bonferroni = tests[tests["significant_bonferroni"]]
    lag5 = tests[tests["metric"] == "autocorr_lag_5"].iloc[0]
    example_137 = encode_base_e(137, frac_digits=14, precision=args.precision)
    class_counts = features["class"].value_counts().to_dict()

    lines = [
        "# Prime Geometry Search in Base-e",
        "",
        "## Scope",
        "",
        "This report is an exploratory audit of greedy base-e digit expansions for primes "
        "versus composites. It is not evidence for the Riemann hypothesis and it does not "
        "attempt to prove primality from digit geometry. The success criterion requested "
        "for this run is a Mann-Whitney two-sided p-value below 0.01 for simple, predefined "
        "features.",
        "",
        "## Encoder",
        "",
        "For an integer X, the script computes the greedy beta-expansion in beta = e. "
        "Because floor(e) = 2, each digit lies in the alphabet {0, 1, 2}. The fractional "
        f"part is truncated at {args.frac_digits} digits with Decimal precision "
        f"{args.precision}.",
        "",
        f"Reference check: 137 -> `{example_137.compact}` with the first 14 fractional "
        "digits shown.",
        "",
        "## Dataset",
        "",
        f"- Integer range scanned: [{args.start}, {args.stop}]",
        f"- Prime class: {class_counts.get('prime', 0)} consecutive primes "
        f"from {min(primes)} to {max(primes)}",
        f"- Composite class: {class_counts.get('composite', 0)} random composites "
        f"from {min(composites)} to {max(composites)}",
        f"- Composite sampling seed: {args.seed}",
        f"- Composite window matched to selected prime span: {args.matched_composite_window}",
        "",
        "## Feature Tests",
        "",
        markdown_table(
            tests,
            [
                "metric",
                "prime_median",
                "composite_median",
                "delta_median",
                "rank_biserial",
                "p_value",
                "bonferroni_p_value",
            ],
        ),
        "",
        "## N=5 Resonance Check",
        "",
        f"The requested N=5 autocorrelation feature has p = {lag5['p_value']:.6g} "
        f"(Bonferroni p = {lag5['bonferroni_p_value']:.6g}), "
        f"prime median = {lag5['prime_median']:.6g}, "
        f"composite median = {lag5['composite_median']:.6g}.",
        "",
        "## Success Criterion",
        "",
    ]

    if significant_uncorrected.empty:
        lines.extend(
            [
                "No predefined feature reached the requested uncorrected p < 0.01 threshold.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The following predefined features reached uncorrected p < 0.01:",
                "",
                markdown_table(
                    significant_uncorrected,
                    [
                        "metric",
                        "prime_median",
                        "composite_median",
                        "delta_median",
                        "rank_biserial",
                        "p_value",
                        "bonferroni_p_value",
                    ],
                ),
                "",
            ]
        )

    if significant_bonferroni.empty:
        lines.extend(
            [
                "After Bonferroni correction over the tested feature family, no feature "
                "remains significant at p < 0.01. This is the conservative reading.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "After Bonferroni correction, the following features remain below p < 0.01:",
                "",
                markdown_table(
                    significant_bonferroni,
                    [
                        "metric",
                        "prime_median",
                        "composite_median",
                        "delta_median",
                        "rank_biserial",
                        "p_value",
                        "bonferroni_p_value",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            f"- Feature matrix: `{feature_path}`",
            f"- Mann-Whitney summary: `{tests_path}`",
            f"- Report: `{report_path}`",
            "",
            "## Interpretation Guardrail",
            "",
            "Any positive marker in this report should be treated as a candidate for a "
            "pre-registered out-of-sample test on disjoint integer ranges and different "
            "base choices. The current run is a bounded exploratory scan.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--stop", type=int, default=5000)
    parser.add_argument("--n-primes", type=int, default=500)
    parser.add_argument("--n-composites", type=int, default=500)
    parser.add_argument("--frac-digits", type=int, default=50)
    parser.add_argument("--precision", type=int, default=120)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--matched-composite-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample composites from the same numeric span covered by the selected primes.",
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--tests-path", type=Path, default=DEFAULT_TESTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Prime geometry base-e audit")
    print(
        "range=[%d,%d] n_primes=%d n_composites=%d frac_digits=%d precision=%d seed=%d"
        % (
            args.start,
            args.stop,
            args.n_primes,
            args.n_composites,
            args.frac_digits,
            args.precision,
            args.seed,
        )
    )

    primes, composites = build_samples(
        start=args.start,
        stop=args.stop,
        n_primes=args.n_primes,
        n_composites=args.n_composites,
        seed=args.seed,
        matched_composite_window=args.matched_composite_window,
    )
    print(f"selected prime span: {min(primes)}..{max(primes)}")
    print(f"selected composite span: {min(composites)}..{max(composites)}")
    print("generating base-e digit matrices...")

    features = feature_table(
        primes,
        composites,
        frac_digits=args.frac_digits,
        precision=args.precision,
        max_lag=args.max_lag,
    )
    tests = mann_whitney_tests(features, alpha=args.alpha, max_lag=args.max_lag)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.feature_path.parent.mkdir(parents=True, exist_ok=True)
    args.tests_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.feature_path, index=False)
    tests.to_csv(args.tests_path, index=False)

    report = render_report(
        args=args,
        features=features,
        tests=tests,
        primes=primes,
        composites=composites,
        report_path=args.report_path,
        feature_path=args.feature_path,
        tests_path=args.tests_path,
    )
    args.report_path.write_text(report, encoding="utf-8")

    best = tests.iloc[0]
    n_uncorrected = int(tests["significant_uncorrected"].sum())
    n_bonferroni = int(tests["significant_bonferroni"].sum())
    print(
        "best metric: %s p=%.6g bonferroni_p=%.6g"
        % (best["metric"], best["p_value"], best["bonferroni_p_value"])
    )
    print(f"significant uncorrected={n_uncorrected}, bonferroni={n_bonferroni}")
    print(f"wrote {args.feature_path}")
    print(f"wrote {args.tests_path}")
    print(f"wrote {args.report_path}")


if __name__ == "__main__":
    main()
