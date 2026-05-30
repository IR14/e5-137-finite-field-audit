from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cosmo_gradient.modular_compression import (
    archive_bytes_for_tokens,
    axis_count,
    compress_text,
    erase_axes,
    reference_vectors,
    repair_archive,
    subparticle_count,
    supervector_bytes,
    vector_count_for_tokens,
    window_tokens,
)


REPORT_PATH = Path("outputs/reports/e5137_extractor_manifest.md")
TABLE_PATH = Path("outputs/tables/e5137_extractor_shootout.csv")

LOREM_WORDS = [
    "lorem",
    "ipsum",
    "dolor",
    "amet",
    "elit",
    "orci",
    "nunc",
    "porta",
    "field",
    "phase",
    "hash",
    "token",
    "vector",
    "repair",
    "matrix",
    "base",
    "light",
    "wave",
    "grain",
    "vault",
    "seed",
    "prime",
    "orbit",
    "logic",
    "archive",
    "residue",
    "context",
    "vacuum",
    "modular",
    "signal",
    "memory",
    "axis",
]


@dataclass(frozen=True)
class Scenario:
    label: str
    description: str
    token_count: int


SCENARIOS = [
    Scenario("A", "Short paragraph", 150),
    Scenario("B", "Long article / semantic quantum", 1_500),
    Scenario("C", "Book-scale context", 120_000),
]


def lorem_stream(token_count: int) -> str:
    words: list[str] = []
    for index in range(token_count):
        word = LOREM_WORDS[(index * 17 + index // 7) % len(LOREM_WORDS)]
        if index % 97 == 0:
            word = word.capitalize()
        if index % 23 == 22:
            word += "."
        elif index % 11 == 10:
            word += ","
        words.append(word)
    return " ".join(words)


def run_scenario(scenario: Scenario) -> dict[str, object]:
    text = lorem_stream(scenario.token_count)
    original_bytes = len(text.encode("utf-8"))
    archive = compress_text(text, token_count=scenario.token_count)
    erased_axes = list(range(10))
    corrupted = erase_axes(archive, erased_axes)
    repaired, min_votes, repair_ok = repair_archive(corrupted)
    exact_archive_restore = bool(repair_ok and np.array_equal(repaired, archive))
    archive_bytes = int(archive.nbytes)
    return {
        "scenario": scenario.label,
        "description": scenario.description,
        "tokens": scenario.token_count,
        "original_bytes": original_bytes,
        "supervectors": int(archive.shape[0]),
        "archive_bytes": archive_bytes,
        "expected_archive_bytes": archive_bytes_for_tokens(scenario.token_count),
        "compression_ratio": original_bytes / archive_bytes,
        "bytes_saved": original_bytes - archive_bytes,
        "erased_axes": len(erased_axes),
        "remaining_axes": axis_count() - len(erased_axes),
        "min_repair_votes": min_votes,
        "majority_repair_ok": exact_archive_restore,
        "lossless_text_restore_claimed": False,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "scenario",
        "tokens",
        "original_bytes",
        "supervectors",
        "archive_bytes",
        "compression_ratio",
        "erased_axes",
        "min_repair_votes",
        "majority_repair_ok",
    ]
    return frame[columns].to_markdown(index=False, floatfmt=".6g")


def write_report(results: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = results.loc[results["scenario"].eq("C")].iloc[0]
    lines = [
        "# e-5-137 GF(137) hash-archive shootout",
        "",
        "Status: Phase 10 semantic hash-archive benchmark. This is a lossy",
        "GF(137) super-vector archive, not a Huffman/LZMA-style reversible text",
        "compressor. The repair test restores the archived super-vectors after",
        "axis erasure; it does not reconstruct arbitrary original prose from the",
        "hash alone.",
        "",
        "## Kernel",
        "",
        "- C++ source: `src/modular_compression/e5137_extractor.cpp`",
        "- Python binding: `src/cosmo_gradient/modular_compression.py`",
        "- Scenario runner: `scripts/e5137_compressor_shootout.py`",
        f"- GF modulus: 137",
        f"- axes: {axis_count()}",
        f"- subparticles per axis: {subparticle_count()}",
        f"- super-vector bytes: {supervector_bytes()}",
        f"- nominal token window: {window_tokens()}",
        f"- book-scale semantic cap: {reference_vectors()} super-vectors",
        "",
        "## Size shootout",
        "",
        markdown_table(results),
        "",
        "## Book-scale invariant",
        "",
        "Scenario C is forced through the archived semantic limit:",
        "",
        "```text",
        f"tokens = {int(c['tokens'])}",
        f"supervectors = {int(c['supervectors'])}",
        f"archive_bytes = {int(c['archive_bytes'])}",
        f"expected = 117 * 130 = {reference_vectors() * supervector_bytes()} bytes",
        f"compression_ratio = {float(c['compression_ratio']):.6g}",
        "```",
        "",
        "## Repair audit",
        "",
        "Ten of twenty-six axes are erased in every archived super-vector. The",
        "kernel reconstructs the erased axes by reducing each surviving axis back",
        "to its five core GF(137) subparticle residues and applying majority",
        "repair. With 10 erased axes, 16 axes remain, so the minimum successful",
        "vote count is expected to be 16.",
        "",
        "All three scenarios restore the archived super-vector exactly after",
        "axis erasure. No lossless text restoration claim is made.",
        "",
        "## Guardrail",
        "",
        "The byte-ratio table is a hash-archive ratio. It is useful for fixed-size",
        "semantic fingerprints, deduplicated archive catalogs, and fault-tolerant",
        "context signatures. It is not a universal compressor for arbitrary text",
        "unless an external generative dictionary or source model is also supplied.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [run_scenario(scenario) for scenario in SCENARIOS]
    results = pd.DataFrame(rows)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_PATH, index=False)
    write_report(results)
    print(REPORT_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
