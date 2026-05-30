#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 QUEUE_TSV [PARALLEL_DOWNLOADS] [MAX_PASSES]" >&2
  exit 2
fi

queue_path="$1"
parallel_downloads="${2:-3}"
max_passes="${3:-20}"

count_missing() {
  local total=0
  local missing=0
  local url
  local path
  while IFS=$'\t' read -r url path; do
    [[ -z "${url:-}" || "${url:0:1}" == "#" ]] && continue
    total=$((total + 1))
    if [[ ! -f "${path}.done" || ! -s "$path" ]]; then
      missing=$((missing + 1))
      continue
    fi
    if ! uv run python scripts/validate_fits.py "$path" >/dev/null 2>&1; then
      rm -f "${path}.done" "$path"
      missing=$((missing + 1))
    fi
  done < "$queue_path"
  echo "$missing $total"
}

for pass in $(seq 1 "$max_passes"); do
  read -r missing total < <(count_missing)
  echo "queue status before pass $pass: complete=$((total - missing))/$total missing=$missing"
  if [[ "$missing" -eq 0 ]]; then
    exit 0
  fi
  scripts/download_queue.sh "$queue_path" "$parallel_downloads" || true
  read -r missing total < <(count_missing)
  echo "queue status after pass $pass: complete=$((total - missing))/$total missing=$missing"
  if [[ "$missing" -eq 0 ]]; then
    exit 0
  fi
  sleep 30
done

read -r missing total < <(count_missing)
echo "queue incomplete after $max_passes passes: complete=$((total - missing))/$total missing=$missing" >&2
exit 1
