"""Minimal complex-alphabet audit over {0, 1, i, pi, e}.

Expression cost is the number of atoms used. The default maximum cost is five
atoms, matching the requested alphabet-sized budget. The target quantities are
real, so candidates with non-negligible imaginary parts are excluded from the
nearest-hit table. The imaginary unit can still contribute through identities
such as i^2 = -1.

This is an anti-numerology stress test, not a formula search for publication.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
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
class Candidate:
    expression: str
    value: complex
    cost: int


def ppm(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1e-300) * 1_000_000.0


def targets() -> list[Target]:
    return [
        Target("alpha_inv", ALPHA_INV, 137.0, 137.08),
        Target("delta_alpha_inv_minus_137", ALPHA_INV - 137.0, 0.001, 0.08),
        Target("muon_to_electron_mass_ratio", MUON_MASS_MEV / ELECTRON_MASS_MEV, 190.0, 220.0),
        Target("tau_to_electron_mass_ratio", TAU_MASS_MEV / ELECTRON_MASS_MEV, 3300.0, 3600.0),
        Target("proton_to_electron_mass_ratio", PROTON_ELECTRON_RATIO, 1800.0, 1870.0),
        Target("q_operator", Q, 0.001, 0.03),
        Target("T5_operator", T5, 0.05, 0.25),
    ]


def _key(value: complex, digits: int) -> tuple[float, float]:
    return (round(value.real, digits), round(value.imag, digits))


def _is_finite_bounded(value: complex, bound: float) -> bool:
    return (
        math.isfinite(value.real)
        and math.isfinite(value.imag)
        and abs(value) <= bound
    )


def _add_candidate(
    layer: dict[tuple[float, float], Candidate],
    expression: str,
    value: complex,
    cost: int,
    *,
    bound: float,
    digits: int,
) -> None:
    if not _is_finite_bounded(value, bound):
        return
    key = _key(value, digits)
    old = layer.get(key)
    if old is None or len(expression) < len(old.expression):
        layer[key] = Candidate(expression=expression, value=value, cost=cost)


def _binary_candidates(left: Candidate, right: Candidate) -> list[tuple[str, complex]]:
    a = left.value
    b = right.value
    la = left.expression
    lb = right.expression
    out: list[tuple[str, complex]] = [
        (f"({la}+{lb})", a + b),
        (f"({la}-{lb})", a - b),
        (f"({lb}-{la})", b - a),
        (f"({la}*{lb})", a * b),
    ]
    if abs(b) > 1e-12:
        out.append((f"({la}/{lb})", a / b))
    if abs(a) > 1e-12:
        out.append((f"({lb}/{la})", b / a))

    # Exponentiation is tightly bounded to keep the grammar interpretable:
    # - complex bases only get small integer real powers;
    # - positive real bases may get bounded real exponents.
    for base_label, base, exp_label, exp in ((la, a, lb, b), (lb, b, la, a)):
        if abs(exp.imag) >= 1e-12:
            continue
        exp_real = exp.real
        exp_round = round(exp_real)
        if (
            abs(exp_real - exp_round) < 1e-12
            and -5 <= exp_round <= 5
            and not (abs(base) < 1e-12 and exp_round <= 0)
        ):
            try:
                out.append((f"({base_label}^{exp_label})", base ** int(exp_round)))
            except (OverflowError, ZeroDivisionError, ValueError):
                pass
        elif abs(base.imag) < 1e-12 and base.real > 0 and abs(exp_real) <= 6:
            try:
                out.append((f"({base_label}^{exp_label})", complex(base.real**exp_real, 0.0)))
            except (OverflowError, ValueError):
                pass
    return out


def atoms_for_alphabet(alphabet: str) -> list[Candidate]:
    minimal = [
        Candidate("0", 0j, 1),
        Candidate("1", 1 + 0j, 1),
        Candidate("i", 1j, 1),
        Candidate("pi", complex(math.pi, 0.0), 1),
        Candidate("e", complex(math.e, 0.0), 1),
    ]
    if alphabet == "minimal":
        return minimal
    if alphabet != "model":
        raise ValueError(f"Unknown alphabet: {alphabet}")

    model_atoms = [
        Candidate("2", 2 + 0j, 1),
        Candidate("3", 3 + 0j, 1),
        Candidate("5", 5 + 0j, 1),
        Candidate("13", 13 + 0j, 1),
        Candidate("26", 26 + 0j, 1),
        Candidate("137", 137 + 0j, 1),
        Candidate("phi137", 136 + 0j, 1),
        Candidate("F26", 121_393 + 0j, 1),
        Candidate("I5", 42 + 0j, 1),
        Candidate("q", complex(Q, 0.0), 1),
        Candidate("T5", complex(T5, 0.0), 1),
        Candidate("cos1_5", complex(math.cos(1 / N), 0.0), 1),
        Candidate("cos2_5", complex(math.cos(2 / N), 0.0), 1),
        Candidate("delta_phi", complex(math.cos(1 / N) - math.cos(2 / N), 0.0), 1),
    ]
    return minimal + model_atoms


def generate_candidates(
    max_cost: int,
    *,
    alphabet: str = "minimal",
    bound: float = 1e6,
    digits: int = 12,
) -> list[Candidate]:
    atoms = atoms_for_alphabet(alphabet)
    layers: list[dict[tuple[float, float], Candidate]] = [
        {} for _ in range(max_cost + 1)
    ]
    for atom in atoms:
        _add_candidate(
            layers[1],
            atom.expression,
            atom.value,
            atom.cost,
            bound=bound,
            digits=digits,
        )

    for cost in range(2, max_cost + 1):
        for left_cost in range(1, cost):
            right_cost = cost - left_cost
            left_items = list(layers[left_cost].values())
            right_items = list(layers[right_cost].values())
            for left in left_items:
                for right in right_items:
                    for expression, value in _binary_candidates(left, right):
                        _add_candidate(
                            layers[cost],
                            expression,
                            value,
                            cost,
                            bound=bound,
                            digits=digits,
                        )

    candidates: list[Candidate] = []
    for cost in range(1, max_cost + 1):
        candidates.extend(layers[cost].values())
    return candidates


def real_candidates(candidates: list[Candidate], imag_tol: float) -> tuple[np.ndarray, list[Candidate]]:
    real = [candidate for candidate in candidates if abs(candidate.value.imag) <= imag_tol]
    real.sort(key=lambda candidate: candidate.value.real)
    values = np.asarray([candidate.value.real for candidate in real], dtype=float)
    return values, real


def nearest(values: np.ndarray, candidates: list[Candidate], target: float) -> Candidate:
    index = int(np.searchsorted(values, target))
    indices = [max(0, index - 1), min(len(values) - 1, index)]
    best_index = min(indices, key=lambda idx: abs(values[idx] - target))
    return candidates[best_index]


def audit(
    max_cost: int,
    random_targets: int,
    seed: int,
    *,
    alphabet: str,
    imag_tol: float,
    bound: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    candidates = generate_candidates(max_cost=max_cost, alphabet=alphabet, bound=bound)
    values, real = real_candidates(candidates, imag_tol=imag_tol)
    rng = np.random.default_rng(seed)
    hit_rows: list[dict[str, float | int | str]] = []
    null_rows: list[dict[str, float | int | str]] = []

    for target in targets():
        best = nearest(values, real, target.value)
        observed_ppm = ppm(best.value.real, target.value)
        random_values = rng.uniform(target.null_low, target.null_high, size=random_targets)
        random_ppm = np.array(
            [
                ppm(nearest(values, real, float(random_value)).value.real, float(random_value))
                for random_value in random_values
            ],
            dtype=float,
        )
        hit_rows.append(
            {
                "target": target.name,
                "value": f"{target.value:.17g}",
                "best_expression": best.expression,
                "best_value": f"{best.value.real:.17g}",
                "observed_ppm": observed_ppm,
                "cost": best.cost,
            }
        )
        null_rows.append(
            {
                "target": target.name,
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

    metadata = {
        "alphabet_size": len(atoms_for_alphabet(alphabet)),
        "total_candidates": len(candidates),
        "real_candidates": len(real),
    }
    return pd.DataFrame(hit_rows), pd.DataFrame(null_rows), metadata


def write_report(
    path: Path,
    hits: pd.DataFrame,
    nulls: pd.DataFrame,
    metadata: dict[str, int],
    *,
    alphabet: str,
    max_cost: int,
    random_targets: int,
    imag_tol: float,
    bound: float,
) -> None:
    lines = [
        "# Minimal Complex Alphabet Audit",
        "",
        f"Alphabet mode: `{alphabet}`.",
        "",
        "Expression cost is the number of atoms used. Binary operations are",
        "`+`, `-`, `*`, `/`, and tightly bounded exponentiation. Since targets are",
        "real, candidates with non-negligible imaginary parts are excluded from",
        "the nearest-hit table. The imaginary unit remains available through",
        "real reductions such as `i^2 = -1`.",
        "",
        "## Setup",
        "",
        f"- max atom cost: `{max_cost}`",
        f"- alphabet size: `{metadata['alphabet_size']}`",
        f"- random targets per range: `{random_targets}`",
        f"- magnitude bound: `{bound:g}`",
        f"- real-candidate imaginary tolerance: `{imag_tol:g}`",
        f"- total generated candidates: `{metadata['total_candidates']}`",
        f"- real generated candidates: `{metadata['real_candidates']}`",
        "",
        "## Best Real Hits",
        "",
        hits.to_markdown(index=False),
        "",
        "## Random-Target Null",
        "",
        nulls.to_markdown(index=False),
        "",
        "## Reading Rule",
        "",
        "- Low ppm error is only interesting if random targets are not hit as well.",
        "- High `empirical_p_random_as_good` means the grammar is dense in that",
        "  target range.",
        "- This audit intentionally restricts the alphabet to avoid importing",
        "  problem-specific constants in `minimal` mode. In `model` mode it",
        "  intentionally measures how much denser the grammar becomes after",
        "  adding the model-specific constants.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphabet",
        choices=["minimal", "model"],
        default="minimal",
        help=(
            "minimal = {0,1,i,pi,e}; model additionally includes 2,3,5,13,26,"
            "137, phi137, F26, I5, q, T5, cos(1/5), cos(2/5), delta_phi."
        ),
    )
    parser.add_argument("--max-cost", type=int, default=5)
    parser.add_argument("--random-targets", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--imag-tol", type=float, default=1e-10)
    parser.add_argument("--bound", type=float, default=1e6)
    parser.add_argument("--output-prefix", default="minimal_complex_alphabet_audit")
    args = parser.parse_args()

    if args.alphabet == "model" and args.max_cost > 4:
        raise SystemExit(
            "Exhaustive model-alphabet search above max-cost 4 is intentionally "
            "disabled: cost 4 already generates millions of candidates. Use "
            "--max-cost 4 for the reproducible audit."
        )

    hits, nulls, metadata = audit(
        max_cost=args.max_cost,
        random_targets=args.random_targets,
        seed=args.seed,
        alphabet=args.alphabet,
        imag_tol=args.imag_tol,
        bound=args.bound,
    )

    tables_dir = Path("outputs/tables")
    reports_dir = Path("outputs/reports")
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    hits_path = tables_dir / f"{args.output_prefix}_hits.csv"
    null_path = tables_dir / f"{args.output_prefix}_null.csv"
    report_path = reports_dir / f"{args.output_prefix}.md"
    hits.to_csv(hits_path, index=False)
    nulls.to_csv(null_path, index=False)
    write_report(
        report_path,
        hits,
        nulls,
        metadata,
        alphabet=args.alphabet,
        max_cost=args.max_cost,
        random_targets=args.random_targets,
        imag_tol=args.imag_tol,
        bound=args.bound,
    )

    print(hits_path)
    print(null_path)
    print(report_path)
    print(hits.to_string(index=False))
    print(nulls.to_string(index=False))


if __name__ == "__main__":
    main()
