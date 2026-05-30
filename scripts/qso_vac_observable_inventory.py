"""Inventory QSO-suitable FastSpecFit observables after joining DESI LSS catalogs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = (
    "LYALPHA_EW",
    "LYALPHA_FLUX",
    "CIV_1549_EW",
    "CIV_1549_FLUX",
    "CIII_1908_EW",
    "CIII_1908_FLUX",
    "MGII_2796_EW",
    "MGII_2796_FLUX",
    "MGII_2803_EW",
    "MGII_2803_FLUX",
    "OII_3726_EW",
    "OII_3726_FLUX",
    "OII_3729_EW",
    "OII_3729_FLUX",
)


def _read_fits_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    import fitsio

    with fitsio.FITS(path) as fits:
        existing = set(fits[1].get_colnames())
    selected = [column for column in columns if column in existing]
    if not selected:
        raise ValueError(f"No requested columns are present in {path}")
    records = fitsio.read(path, ext=1, columns=selected)
    frame = pd.DataFrame({name: _native(records[name]) for name in records.dtype.names or []})
    return frame


def _native(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.byteorder in ("=", "|"):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _ivar_column(name: str) -> str:
    if name.endswith("_EW"):
        return f"{name}_IVAR"
    if name.endswith("_FLUX"):
        return f"{name}_IVAR"
    return f"{name}_IVAR"


def _candidate_columns(candidates: tuple[str, ...]) -> list[str]:
    columns = ["TARGETID", "Z", "RCHI2_LINE", "DELTA_LINECHI2"]
    for name in candidates:
        columns.append(name)
        columns.append(_ivar_column(name))
    return list(dict.fromkeys(columns))


def build_inventory(
    lss_paths: list[Path],
    vac_path: Path,
    candidates: tuple[str, ...],
    z_min: float,
    z_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lss_frames = []
    for path in lss_paths:
        frame = _read_fits_columns(path, ["TARGETID", "Z", "RA", "DEC", "WEIGHT"])
        frame["source_region"] = "NGC" if "_NGC_" in path.name else "SGC" if "_SGC_" in path.name else "unknown"
        lss_frames.append(frame)
    lss = pd.concat(lss_frames, ignore_index=True)
    lss = lss.loc[
        np.isfinite(lss["Z"]) & (lss["Z"] >= z_min) & (lss["Z"] <= z_max)
    ].copy()
    vac = _read_fits_columns(vac_path, _candidate_columns(candidates))
    joined = lss.merge(vac, on="TARGETID", how="inner", suffixes=("_LSS", "_VAC"))
    z_col = "Z_LSS" if "Z_LSS" in joined.columns else "Z"

    rows = []
    for name in candidates:
        if name not in joined.columns:
            continue
        ivar = _ivar_column(name)
        values = pd.to_numeric(joined[name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if ivar in joined.columns:
            ivar_values = pd.to_numeric(joined[ivar], errors="coerce").to_numpy(dtype=float)
            good_ivar = np.isfinite(ivar_values) & (ivar_values > 0.0)
        else:
            good_ivar = np.zeros(len(joined), dtype=bool)
        usable = finite & good_ivar
        low = joined[z_col].to_numpy(dtype=float) < 2.1
        high = ~low
        rows.append(
            {
                "observable": name,
                "ivar_column": ivar if ivar in joined.columns else "",
                "n_joined": len(joined),
                "n_finite": int(np.sum(finite)),
                "n_ivar_positive": int(np.sum(good_ivar)),
                "n_usable": int(np.sum(usable)),
                "n_usable_z1p5_2p1": int(np.sum(usable & low)),
                "n_usable_z2p1_3p5": int(np.sum(usable & high)),
                "usable_fraction": float(np.mean(usable)) if len(joined) else 0.0,
                "median": float(np.nanmedian(values[usable])) if np.any(usable) else np.nan,
                "p01": float(np.nanquantile(values[usable], 0.01)) if np.any(usable) else np.nan,
                "p99": float(np.nanquantile(values[usable], 0.99)) if np.any(usable) else np.nan,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["n_usable", "usable_fraction"],
        ascending=False,
    )
    return summary, joined[["TARGETID", z_col, "source_region"]].rename(columns={z_col: "z"})


def write_report(summary: pd.DataFrame, joined_index: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# QSO FastSpecFit observable inventory",
        "",
        "This report inventories emission-line observables that are more appropriate",
        "for QSO high-z work than `DN4000_MODEL`.",
        "",
        "## Joined Sample",
        "",
        f"- joined QSO rows: `{len(joined_index)}`",
        f"- NGC rows: `{int(np.sum(joined_index['source_region'] == 'NGC'))}`",
        f"- SGC rows: `{int(np.sum(joined_index['source_region'] == 'SGC'))}`",
        "",
        "## Candidate Observables",
        "",
        summary.to_markdown(index=False),
        "",
        "## Recommended Next Test",
        "",
        "Use the highest-coverage QSO emission-line observables with positive IVAR",
        "as population residuals, controlling for redshift and luminosity proxies.",
        "The first candidates are the rows with the largest `n_usable` in this table.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vac", default="data/raw/vac/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-dark.fits")
    parser.add_argument("--lss", action="append", default=["data/raw/QSO_NGC_clustering.dat.fits", "data/raw/QSO_SGC_clustering.dat.fits"])
    parser.add_argument("--z-min", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=3.5)
    parser.add_argument("--output-prefix", default="qso_vac_observable_inventory")
    args = parser.parse_args()

    summary, joined = build_inventory(
        lss_paths=[Path(path) for path in args.lss],
        vac_path=Path(args.vac),
        candidates=DEFAULT_CANDIDATES,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    table_path = Path("outputs/tables") / f"{args.output_prefix}.csv"
    report_path = Path("outputs/reports") / f"{args.output_prefix}.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    write_report(summary, joined, report_path)
    print(table_path)
    print(report_path)
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
