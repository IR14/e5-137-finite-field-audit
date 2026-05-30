#!/usr/bin/env python3
"""Parallel HTTP range downloader for large DESI files.

The script writes directly into the final output file using positioned writes,
so it does not need a second full-size temporary copy while assembling chunks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


class DownloadState:
    def __init__(self, manifest_path: Path, total_size: int, completed: set[int]) -> None:
        self.manifest_path = manifest_path
        self.total_size = total_size
        self.completed = completed
        self.completed_bytes = 0
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def mark_completed(self, chunk: Chunk) -> None:
        with self.lock:
            if chunk.index not in self.completed:
                self.completed.add(chunk.index)
                self.completed_bytes += chunk.size
                write_manifest(self.manifest_path, self.completed)

    def completed_count(self) -> int:
        with self.lock:
            return len(self.completed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel range downloader.")
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--engine", choices=["curl", "urllib"], default="curl")
    parser.add_argument("--low-speed-limit", type=int, default=32768)
    parser.add_argument("--low-speed-time", type=int, default=45)
    parser.add_argument(
        "--resolve-ip",
        help="Pin the URL hostname to this IP for curl, avoiding repeated DNS lookups.",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Known file size in bytes. Use this for WebDAV endpoints with broken HEAD metadata.",
    )
    args = parser.parse_args()

    head_size, accepts_ranges = head(args.url, timeout=args.timeout)
    size = args.size or head_size
    if size <= 0:
        raise RuntimeError("Could not determine file size; pass --size.")
    if not accepts_ranges and args.size is None:
        raise RuntimeError("Server did not advertise byte-range support.")

    chunk_size = args.chunk_mib * 1024 * 1024
    chunks = make_chunks(size, chunk_size)
    manifest_path = args.output.with_suffix(args.output.suffix + ".range-manifest.json")
    completed = read_manifest(manifest_path, args.url, size, chunk_size)
    state = DownloadState(manifest_path, size, completed)
    state.completed_bytes = sum(chunks[index].size for index in completed if index < len(chunks))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("ab") as handle:
        handle.truncate(size)
    part_dir = args.output.with_suffix(args.output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)

    pending = [chunk for chunk in chunks if chunk.index not in completed]
    print(
        f"url={args.url}\n"
        f"output={args.output}\n"
        f"size={size} bytes chunks={len(chunks)} pending={len(pending)} "
        f"workers={args.workers} chunk_mib={args.chunk_mib}",
        flush=True,
    )
    if not pending:
        print("already_complete=true", flush=True)
        return 0

    progress_stop = threading.Event()
    progress_thread = threading.Thread(
        target=print_progress,
        args=(state, len(chunks), progress_stop),
        daemon=True,
    )
    progress_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    download_chunk,
                    args.url,
                    args.output,
                    part_dir,
                    chunk,
                    args.retries,
                    args.timeout,
                    args.engine,
                    args.low_speed_limit,
                    args.low_speed_time,
                    args.resolve_ip,
                )
                for chunk in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                chunk = future.result()
                state.mark_completed(chunk)
    finally:
        progress_stop.set()
        progress_thread.join(timeout=2.0)

    if state.completed_count() != len(chunks):
        raise RuntimeError(f"Download incomplete: {state.completed_count()}/{len(chunks)} chunks")
    print_progress(state, len(chunks), threading.Event(), final=True)
    return 0


def head(url: str, timeout: float) -> tuple[int, bool]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        size = int(response.headers["Content-Length"])
        ranges = response.headers.get("Accept-Ranges", "").lower()
    return size, ranges == "bytes"


def make_chunks(total_size: int, chunk_size: int) -> list[Chunk]:
    count = math.ceil(total_size / chunk_size)
    return [
        Chunk(
            index=index,
            start=index * chunk_size,
            end=min(total_size - 1, (index + 1) * chunk_size - 1),
        )
        for index in range(count)
    ]


def read_manifest(path: Path, url: str, size: int, chunk_size: int) -> set[int]:
    if not path.exists():
        write_manifest(path, set(), url=url, size=size, chunk_size=chunk_size)
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("url") != url or payload.get("size") != size or payload.get("chunk_size") != chunk_size:
        raise RuntimeError(f"Existing manifest does not match this download: {path}")
    return {int(index) for index in payload.get("completed", [])}


def write_manifest(
    path: Path,
    completed: set[int],
    *,
    url: str | None = None,
    size: int | None = None,
    chunk_size: int | None = None,
) -> None:
    payload: dict[str, Any]
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    if url is not None:
        payload["url"] = url
    if size is not None:
        payload["size"] = size
    if chunk_size is not None:
        payload["chunk_size"] = chunk_size
    payload["completed"] = sorted(completed)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def download_chunk(
    url: str,
    output: Path,
    part_dir: Path,
    chunk: Chunk,
    retries: int,
    timeout: float,
    engine: str,
    low_speed_limit: int,
    low_speed_time: int,
    resolve_ip: str | None,
) -> Chunk:
    if engine == "curl":
        return download_chunk_curl(
            url,
            output,
            part_dir,
            chunk,
            retries,
            timeout,
            low_speed_limit,
            low_speed_time,
            resolve_ip,
        )
    return download_chunk_urllib(url, output, chunk, retries, timeout)


def download_chunk_curl(
    url: str,
    output: Path,
    part_dir: Path,
    chunk: Chunk,
    retries: int,
    timeout: float,
    low_speed_limit: int,
    low_speed_time: int,
    resolve_ip: str | None,
) -> Chunk:
    part_path = part_dir / f"chunk-{chunk.index:06d}.part"
    curl_timeout = int(timeout)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if part_path.exists() and part_path.stat().st_size != chunk.size:
                part_path.unlink()
            if not part_path.exists():
                command = [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--speed-limit",
                    str(low_speed_limit),
                    "--speed-time",
                    str(low_speed_time),
                    "--max-time",
                    str(curl_timeout),
                    "--range",
                    f"{chunk.start}-{chunk.end}",
                    "-o",
                    str(part_path),
                    url,
                ]
                if resolve_ip is not None:
                    host = urlparse(url).hostname
                    if host is None:
                        raise RuntimeError(f"Could not parse host from URL: {url}")
                    command[1:1] = ["--resolve", f"{host}:443:{resolve_ip}"]
                subprocess.run(command, check=True)
            if part_path.stat().st_size != chunk.size:
                raise RuntimeError(
                    f"Short curl chunk {chunk.index}: {part_path.stat().st_size} != {chunk.size}"
                )
            fd = os.open(output, os.O_RDWR)
            try:
                offset = chunk.start
                with part_path.open("rb") as handle:
                    while True:
                        data = handle.read(1024 * 1024)
                        if not data:
                            break
                        os.pwrite(fd, data, offset)
                        offset += len(data)
            finally:
                os.close(fd)
            part_path.unlink(missing_ok=True)
            return chunk
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            last_error = exc
            time.sleep(min(2.0 * attempt, 30.0))
    raise RuntimeError(f"Chunk {chunk.index} failed after {retries} retries: {last_error}")


def download_chunk_urllib(
    url: str,
    output: Path,
    chunk: Chunk,
    retries: int,
    timeout: float,
) -> Chunk:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={chunk.start}-{chunk.end}"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None)
                if status != 206:
                    raise RuntimeError(f"Expected HTTP 206 for chunk {chunk.index}, got {status}")
                fd = os.open(output, os.O_RDWR)
                try:
                    offset = chunk.start
                    bytes_read = 0
                    while True:
                        data = response.read(1024 * 1024)
                        if not data:
                            break
                        os.pwrite(fd, data, offset)
                        offset += len(data)
                        bytes_read += len(data)
                    if bytes_read != chunk.size:
                        raise RuntimeError(
                            f"Short chunk {chunk.index}: {bytes_read} != {chunk.size}"
                        )
                finally:
                    os.close(fd)
            return chunk
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            time.sleep(min(2.0 * attempt, 30.0))
    raise RuntimeError(f"Chunk {chunk.index} failed after {retries} retries: {last_error}")


def print_progress(
    state: DownloadState,
    chunk_count: int,
    stop: threading.Event,
    final: bool = False,
) -> None:
    while final or not stop.wait(30.0):
        elapsed = max(time.monotonic() - state.started, 1e-9)
        with state.lock:
            completed = state.completed_count()
            bytes_done = state.completed_bytes
        speed = bytes_done / elapsed
        remaining = state.total_size - bytes_done
        eta = remaining / speed if speed > 0.0 else float("inf")
        print(
            "progress="
            f"{bytes_done / state.total_size:.2%} "
            f"chunks={completed}/{chunk_count} "
            f"done_gib={bytes_done / 1024**3:.2f} "
            f"speed_mib_s={speed / 1024**2:.2f} "
            f"eta_min={eta / 60:.1f}",
            flush=True,
        )
        if final:
            return


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted=true", file=sys.stderr)
        raise
