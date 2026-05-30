#!/usr/bin/env python3
"""Lightweight FITS integrity check for resumable catalog downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import fitsio


def validate(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with fitsio.FITS(path) as hdus:
        if len(hdus) < 2:
            raise ValueError(f"Missing table extension: {path}")
        table = hdus[1]
        nrows = int(table.get_nrows())
        columns = list(table.get_colnames())
        if nrows <= 0:
            raise ValueError(f"Empty table extension: {path}")
        if not columns:
            raise ValueError(f"No table columns: {path}")
        table.read(rows=[0, nrows - 1], columns=[columns[0]])


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: validate_fits.py PATH", file=sys.stderr)
        return 2
    try:
        validate(Path(argv[1]))
    except Exception as exc:
        print(f"invalid FITS: {argv[1]}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
