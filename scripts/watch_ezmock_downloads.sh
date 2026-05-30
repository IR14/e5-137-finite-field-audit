#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

repo_dir="/Users/i.mikhailov/Desktop/work/cosmo/cosmo_genesis_gradient"
cd "$repo_dir"

interval_seconds="${1:-300}"
stall_window_seconds="${2:-60}"
stall_threshold_mib="${3:-8}"
curl_pattern="curl .*ELG_LOP_ffa"

measure_growth_mib() {
  local window_seconds="$1"
  uv run python - "$window_seconds" <<'PY'
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

window_seconds = int(sys.argv[1])


def active_files() -> list[Path]:
    try:
        pids = subprocess.check_output(
            ["pgrep", "-f", "curl .*ELG_LOP_ffa"],
            text=True,
        ).split()
    except subprocess.CalledProcessError:
        return []

    paths: set[Path] = set()
    for pid in pids:
        try:
            out = subprocess.check_output(
                ["lsof", "-Fn", "-p", pid],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            continue

        for line in out.splitlines():
            if (
                line.startswith("n/Volumes/T7 Shield/")
                and "ELG_LOP_ffa" in line
                and line.endswith(".fits")
            ):
                paths.add(Path(line[1:]))

    return sorted(paths)


paths = active_files()
before = {p: p.stat().st_size if p.exists() else 0 for p in paths}
time.sleep(window_seconds)
after_paths = active_files()
for p in after_paths:
    before.setdefault(p, p.stat().st_size if p.exists() else 0)
after = {
    p: p.stat().st_size if p.exists() else 0
    for p in sorted(set(paths) | set(after_paths))
}
delta = sum(after[p] - before.get(p, 0) for p in after)
print(f"{delta / 1024 / 1024:.1f}")
PY
}

echo "[$(date)] watchdog started interval=${interval_seconds}s window=${stall_window_seconds}s threshold=${stall_threshold_mib}MiB"
while true; do
  active_count="$(pgrep -f "$curl_pattern" | wc -l | tr -d ' ')"
  if [[ "$active_count" -gt 0 ]]; then
    growth_mib="$(measure_growth_mib "$stall_window_seconds")"
    echo "[$(date)] active_curls=$active_count growth=${growth_mib}MiB/${stall_window_seconds}s"
    if uv run python - "$growth_mib" "$stall_threshold_mib" <<'PY'
import sys

growth = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if growth < threshold else 1)
PY
    then
      echo "[$(date)] stalled download detected; terminating curl processes for resume"
      pgrep -f "$curl_pattern" | xargs kill -TERM || true
      sleep 15
    fi
  else
    echo "[$(date)] no active curls"
  fi

  sleep "$interval_seconds"
done
