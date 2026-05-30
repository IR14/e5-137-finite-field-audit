"""GF(137) semantic hash-archive bindings for the Phase 10 compression audit.

The archive is a deterministic lossy signature, not a lossless text codec. The
repair path restores the archived super-vector after axis erasure; it does not
reconstruct arbitrary original prose from the hash alone.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = PROJECT_ROOT / "src" / "modular_compression" / "e5137_extractor.cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "modular_compression"
COMPILE_LOG = BUILD_DIR / "compile.log"
LIBRARY_PATH = BUILD_DIR / (
    "libe5137compress.dylib" if sys.platform == "darwin" else "libe5137compress.so"
)
ERASED_AXIS_VALUE = 255


def compile_compression_kernel(*, force: bool = False) -> Path:
    """Compile the C++ GF(137) hash-archive kernel."""

    if not CPP_SOURCE.exists():
        raise FileNotFoundError(f"C++ source does not exist: {CPP_SOURCE}")
    if (
        LIBRARY_PATH.exists()
        and not force
        and LIBRARY_PATH.stat().st_mtime >= CPP_SOURCE.stat().st_mtime
    ):
        return LIBRARY_PATH

    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("No C++ compiler found. Install clang++ or g++.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    native_cpu_flag = "-mcpu=native" if sys.platform == "darwin" else "-march=native"
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        native_cpu_flag,
        shared_flag,
        str(CPP_SOURCE),
        "-o",
        str(LIBRARY_PATH),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    COMPILE_LOG.write_text(
        "\n".join(
            [
                "command: " + " ".join(command),
                f"returncode: {completed.returncode}",
                "",
                "[stdout]",
                completed.stdout,
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Compression kernel compilation failed. See {COMPILE_LOG}")
    _load_library.cache_clear()
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(compile_compression_kernel()))
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)
    int_ptr = ctypes.POINTER(ctypes.c_int)

    library.e5137_compression_axis_count.argtypes = []
    library.e5137_compression_axis_count.restype = ctypes.c_int
    library.e5137_compression_subparticle_count.argtypes = []
    library.e5137_compression_subparticle_count.restype = ctypes.c_int
    library.e5137_compression_supervector_bytes.argtypes = []
    library.e5137_compression_supervector_bytes.restype = ctypes.c_int
    library.e5137_compression_window_tokens.argtypes = []
    library.e5137_compression_window_tokens.restype = ctypes.c_int
    library.e5137_compression_reference_vectors.argtypes = []
    library.e5137_compression_reference_vectors.restype = ctypes.c_int
    library.e5137_compression_vector_count_for_tokens.argtypes = [ctypes.c_int]
    library.e5137_compression_vector_count_for_tokens.restype = ctypes.c_int
    library.e5137_compression_archive_bytes_for_tokens.argtypes = [ctypes.c_int]
    library.e5137_compression_archive_bytes_for_tokens.restype = ctypes.c_int
    library.e5137_compress_text_gf137.argtypes = [
        u8_ptr,
        ctypes.c_int,
        ctypes.c_int,
        u8_ptr,
    ]
    library.e5137_compress_text_gf137.restype = ctypes.c_int
    library.e5137_repair_archive_gf137.argtypes = [
        u8_ptr,
        ctypes.c_int,
        u8_ptr,
        int_ptr,
    ]
    library.e5137_repair_archive_gf137.restype = ctypes.c_int
    return library


def axis_count() -> int:
    return int(_load_library().e5137_compression_axis_count())


def subparticle_count() -> int:
    return int(_load_library().e5137_compression_subparticle_count())


def supervector_bytes() -> int:
    return int(_load_library().e5137_compression_supervector_bytes())


def window_tokens() -> int:
    return int(_load_library().e5137_compression_window_tokens())


def reference_vectors() -> int:
    return int(_load_library().e5137_compression_reference_vectors())


def vector_count_for_tokens(token_count: int) -> int:
    return int(_load_library().e5137_compression_vector_count_for_tokens(int(token_count)))


def archive_bytes_for_tokens(token_count: int) -> int:
    return int(_load_library().e5137_compression_archive_bytes_for_tokens(int(token_count)))


def _text_bytes(text: str | bytes | bytearray) -> bytes:
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    if not data:
        raise ValueError("text must not be empty.")
    return data


def _u8(array: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=np.uint8)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return result


def _ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def compress_text(text: str | bytes | bytearray, *, token_count: int) -> np.ndarray:
    """Project text bytes to GF(137) super-vectors of shape `(n, 26, 5)`."""

    data = _text_bytes(text)
    byte_array = np.frombuffer(data, dtype=np.uint8)
    vectors = vector_count_for_tokens(token_count)
    if vectors <= 0:
        raise ValueError("token_count must be positive.")
    out = np.empty((vectors, axis_count(), subparticle_count()), dtype=np.uint8)
    produced = int(
        _load_library().e5137_compress_text_gf137(
            _ptr(np.ascontiguousarray(byte_array)),
            int(byte_array.size),
            int(token_count),
            _ptr(out.reshape(-1)),
        )
    )
    if produced != vectors:
        raise RuntimeError(f"Compression produced {produced} vectors, expected {vectors}.")
    return out


def erase_axes(archive: np.ndarray, axes: list[int] | range | np.ndarray) -> np.ndarray:
    """Return a copy with selected axes erased by the sentinel value 255."""

    archived = _u8(archive, "archive")
    if archived.ndim != 3 or archived.shape[1:] != (axis_count(), subparticle_count()):
        raise ValueError(f"archive must have shape (n, {axis_count()}, {subparticle_count()}).")
    erased = np.array(archived, copy=True)
    axis_indices = [int(axis) for axis in axes]
    if any(axis < 0 or axis >= axis_count() for axis in axis_indices):
        raise ValueError(f"axes must be in [0, {axis_count()}).")
    erased[:, axis_indices, :] = ERASED_AXIS_VALUE
    return erased


def repair_archive(corrupted_archive: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Repair erased GF(137) archive axes and return `(archive, min_votes, ok)`."""

    corrupted = _u8(corrupted_archive, "corrupted_archive")
    if corrupted.ndim != 3 or corrupted.shape[1:] != (axis_count(), subparticle_count()):
        raise ValueError(
            f"corrupted_archive must have shape (n, {axis_count()}, {subparticle_count()})."
        )
    repaired = np.empty_like(corrupted)
    min_votes = np.zeros(1, dtype=np.int32)
    ok = int(
        _load_library().e5137_repair_archive_gf137(
            _ptr(np.ascontiguousarray(corrupted).reshape(-1)),
            int(corrupted.shape[0]),
            _ptr(repaired.reshape(-1)),
            min_votes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
    )
    return repaired, int(min_votes[0]), bool(ok)
