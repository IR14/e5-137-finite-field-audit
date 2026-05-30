#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

repo_dir="/Users/i.mikhailov/Desktop/work/cosmo/cosmo_genesis_gradient"
cd "$repo_dir"

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 QUEUE_TSV [PARALLEL_DOWNLOADS] [MAX_PASSES] [SLEEP_SECONDS]" >&2
  exit 2
fi

queue_path="$1"
parallel_downloads="${2:-8}"
max_passes="${3:-100}"
sleep_seconds="${4:-300}"

count_missing_1_370() {
  uv run python - <<'PY'
from pathlib import Path

roots = [
    Path("data/raw/mocks/EZmock/dark/v1"),
    Path("/Volumes/T7 Shield/cosmo_genesis_gradient/raw/mocks/EZmock/dark/v1"),
]
files = [
    "ELG_LOP_ffa_NGC_clustering.dat.fits",
    "ELG_LOP_ffa_NGC_0_clustering.ran.fits",
    "ELG_LOP_ffa_SGC_clustering.dat.fits",
    "ELG_LOP_ffa_SGC_0_clustering.ran.fits",
]

missing = 0
for realization in range(1, 371):
    for name in files:
        if not any(
            (root / f"mock{realization}" / name).exists()
            and (root / f"mock{realization}" / f"{name}.done").exists()
            for root in roots
        ):
            missing += 1

print(missing)
PY
}

echo "[$(date)] waiting for mock1..370 to become complete"
echo "[$(date)] next queue: $queue_path"
while true; do
  remaining="$(count_missing_1_370)"
  echo "[$(date)] mock1..370 missing_files=$remaining"
  if [[ "$remaining" == "0" ]]; then
    break
  fi
  sleep "$sleep_seconds"
done

echo "[$(date)] mock1..370 complete; starting queue $queue_path"
df -h "/Volumes/T7 Shield" || true
/bin/bash scripts/download_queue_until_complete.sh "$queue_path" "$parallel_downloads" "$max_passes"
