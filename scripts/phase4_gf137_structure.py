"""Compute basic group-theoretic facts for GF(137)^*."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MOD = 137
PHI = MOD - 1
D = 26
N = 5


def multiplicative_order(value: int, mod: int = MOD) -> int:
    if value % mod == 0:
        raise ValueError("zero has no multiplicative order")
    acc = 1
    for order in range(1, mod):
        acc = (acc * value) % mod
        if acc == 1:
            return order
    raise RuntimeError("order not found")


def divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def subgroup_members(generator: int, size: int, mod: int = MOD) -> list[int]:
    return [pow(generator, power, mod) for power in range(size)]


def main() -> None:
    rows = [
        {"element": value, "order": multiplicative_order(value)}
        for value in range(1, MOD)
    ]
    orders = pd.DataFrame(rows)
    primitive_roots = orders.loc[orders["order"] == PHI, "element"].tolist()
    first_generator = int(primitive_roots[0])
    subgroup_rows = []
    for size in divisors(PHI):
        generator = pow(first_generator, PHI // size, MOD)
        members = subgroup_members(generator, size)
        subgroup_rows.append(
            {
                "subgroup_size": size,
                "generator": generator,
                "members_preview": " ".join(str(item) for item in members[:16]),
                "preview_truncated": len(members) > 16,
            }
        )
    subgroup_frame = pd.DataFrame(subgroup_rows)

    tables = Path("outputs/tables")
    reports = Path("outputs/reports")
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    orders.to_csv(tables / "phase4_gf137_orders.csv", index=False)
    subgroup_frame.to_csv(tables / "phase4_gf137_subgroups.csv", index=False)

    distribution = (
        orders.groupby("order")
        .size()
        .reset_index(name="n_elements")
        .sort_values("order")
    )
    distribution.to_csv(tables / "phase4_gf137_order_distribution.csv", index=False)

    lines = [
        "# GF(137) multiplicative-group structure",
        "",
        "`137` is prime, so the nonzero residue classes form a cyclic group",
        "`GF(137)^*` of order `phi(137)=136`.",
        "",
        "## Basic Facts",
        "",
        f"- `phi(137) = {PHI}`",
        f"- `D*N = {D*N}`",
        f"- residual `phi(137) - D*N = {PHI - D*N}`",
        f"- divisors of `136`: `{', '.join(str(item) for item in divisors(PHI))}`",
        f"- first primitive root found: `{first_generator}`",
        f"- number of primitive roots: `{len(primitive_roots)}`",
        "",
        "## Important Constraint",
        "",
        "`D*N = 130` is not a divisor of `136`. Therefore, it cannot be the",
        "order of a subgroup of `GF(137)^*`. A physically meaningful model may still",
        "use a `130 + 6` partition as a state-sector bookkeeping rule, but it is not",
        "a direct subgroup decomposition of the multiplicative group.",
        "",
        "## Order Distribution",
        "",
        distribution.to_markdown(index=False),
        "",
        "## Subgroups",
        "",
        subgroup_frame.to_markdown(index=False),
        "",
        "## Consequence For Model Building",
        "",
        "A rigorous finite-group version of the model needs an explicit representation",
        "or operator whose spectrum naturally splits into a 130-dimensional sector and",
        "a 6-dimensional sector. The bare multiplicative group does not provide that",
        "split by subgroup order.",
    ]
    report_path = reports / "phase4_gf137_structure_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(tables / "phase4_gf137_orders.csv")
    print(tables / "phase4_gf137_subgroups.csv")
    print(report_path)


if __name__ == "__main__":
    main()
