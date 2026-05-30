#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 QUEUE_TSV [PARALLEL_DOWNLOADS]" >&2
  exit 2
fi

queue_path="$1"
parallel_downloads="${2:-2}"

if [[ ! -f "$queue_path" ]]; then
  echo "Missing queue file: $queue_path" >&2
  exit 1
fi

download_one() {
  local url="$1"
  local path="$2"
  local done_path="${path}.done"
  local expected_size=""

  mkdir -p "$(dirname "$path")"
  expected_size="$(
    curl -fLsI "$url" |
      awk 'BEGIN{IGNORECASE=1}/^content-length:/{gsub("\r", "", $2); print $2; exit}'
  )" || expected_size=""
  if [[ -n "$expected_size" && "$expected_size" =~ ^[0-9]+$ && "$expected_size" -lt 1048576 ]]; then
    expected_size=""
  fi
  if [[ -f "$done_path" && -s "$path" ]]; then
    if [[ -n "$expected_size" ]]; then
      local actual_size
      actual_size="$(stat -f '%z' "$path")"
      if [[ "$actual_size" != "$expected_size" ]]; then
        echo "size mismatch for done file, redownloading $path actual=$actual_size expected=$expected_size" >&2
        rm -f "$done_path" "$path"
      fi
    fi
  fi
  if [[ -f "$done_path" && -s "$path" ]]; then
    if uv run python scripts/validate_fits.py "$path" >/dev/null; then
      echo "skip done $path"
      return 0
    fi
    echo "invalid done marker, redownloading $path" >&2
    rm -f "$done_path" "$path"
  fi
  if [[ -s "$path" ]]; then
    if [[ -n "$expected_size" ]]; then
      local partial_size
      partial_size="$(stat -f '%z' "$path")"
      if [[ "$partial_size" -gt "$expected_size" ]]; then
        echo "partial file is larger than server object, redownloading $path actual=$partial_size expected=$expected_size" >&2
        rm -f "$done_path" "$path"
      fi
    fi
  fi
  if [[ -s "$path" ]]; then
    if uv run python scripts/validate_fits.py "$path" >/dev/null 2>&1; then
      echo "mark valid existing $path"
      touch "$done_path"
      return 0
    fi
  fi

  echo "download $url -> $path"
  local curl_args=(
    -fL
    -C -
    --silent
    --show-error
    --retry 8
    --retry-delay 10
    --retry-all-errors
  )
  if [[ -n "${COSMO_DOWNLOAD_LIMIT_RATE:-}" ]]; then
    curl_args+=(--limit-rate "$COSMO_DOWNLOAD_LIMIT_RATE")
  fi
  local curl_status=0
  curl \
    "${curl_args[@]}" \
    -o "$path" \
    "$url" || curl_status=$?
  if [[ "$curl_status" -ne 0 ]]; then
    echo "curl failed with status $curl_status, leaving partial for resume: $path" >&2
    return "$curl_status"
  fi
  if ! uv run python scripts/validate_fits.py "$path" >/dev/null; then
    echo "downloaded file failed FITS validation: $path" >&2
    rm -f "$done_path" "$path"
    return 1
  fi
  if [[ -n "$expected_size" ]]; then
    local downloaded_size
    downloaded_size="$(stat -f '%z' "$path")"
    if [[ "$downloaded_size" != "$expected_size" ]]; then
      echo "downloaded file size mismatch: $path actual=$downloaded_size expected=$expected_size" >&2
      rm -f "$done_path" "$path"
      return 1
    fi
  fi
  touch "$done_path"
}

export -f download_one

awk -F '\t' 'NF >= 2 && $1 !~ /^#/ { printf "%s%c%s%c", $1, 0, $2, 0 }' "$queue_path" |
  xargs -0 -P "$parallel_downloads" -n 2 bash -c 'download_one "$1" "$2"' _
