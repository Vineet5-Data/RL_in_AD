#!/usr/bin/env bash
# Automated Kaggle train-and-fetch cycle for Stage2 RSSM runs.
#
# Bakes in every infra fix hit during Run 1-3 (see stage2_correction.md):
#   - access-token auth check (legacy kaggle.json silently fails on
#     `kernels list --mine` / `datasets status` / paginated dataset list)
#   - forces GPU accelerator explicitly (Kaggle sometimes assigns a P100,
#     sm_60, incompatible with current torch image -> "no kernel image" crash)
#   - auto-heals the Windows kaggle-CLI bug where the local resumable-upload
#     cache dir isn't mkdir -p'd, causing "[Errno 2] No such file or directory"
#     on first dataset upload
#   - exports PYTHONIOENCODING/PYTHONUTF8 before any `kaggle kernels output`
#     call (Windows charmap crash silently truncates the log to 0 bytes otherwise)
#   - polls kernel status to a TERMINAL state before pulling output — Kaggle
#     has no live/streaming log API, `kernels output` returns empty while RUNNING
#
# Usage:
#   ./run_kaggle.sh push <kernel-dir> [accelerator]        # push + poll + pull
#   ./run_kaggle.sh sync-dataset <local-dir> <label>        # create-or-version a dataset
#   ./run_kaggle.sh pull <kernel-id> <dest-dir>             # pull output only
#
# kernel-dir must contain kernel-metadata.json + the notebook it references.
# accelerator defaults to NvidiaTeslaT4 (valid values: NvidiaTeslaT4,
# NvidiaTeslaP100, Tpu1VmV38 -- P100 is the one that crashes current torch).
set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

POLL_INTERVAL=60

check_auth() {
  if [ -n "${KAGGLE_API_TOKEN:-}" ] || [ -f "$HOME/.kaggle/access_token" ]; then
    return 0
  fi
  if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "WARNING: only legacy kaggle.json auth found." >&2
    echo "  Legacy key silently fails on 'kernels list --mine', 'datasets status'," >&2
    echo "  and paginated dataset listing. Prefer KAGGLE_API_TOKEN (access token)." >&2
    return 0
  fi
  echo "ERROR: no Kaggle auth found (KAGGLE_API_TOKEN / ~/.kaggle/access_token / ~/.kaggle/kaggle.json)" >&2
  exit 1
}

# Retries a `kaggle datasets create|version` call once, auto-creating the
# missing local upload-cache dir the Windows CLI fails to mkdir -p itself.
dataset_sync_retry() {
  local cmd_str="$1"
  local out
  if out=$(eval "$cmd_str" 2>&1); then
    echo "$out"
    return 0
  fi
  local missing
  missing=$(echo "$out" | grep -oE "No such file or directory: '[^']+'" | sed -E "s/No such file or directory: '(.*)'//;s/^'//;s/'\$//" || true)
  missing=$(echo "$out" | grep -oE "No such file or directory: '[^']+'" | sed -E "s/.*: '(.*)'/\1/" || true)
  if [ -n "$missing" ]; then
    echo "known Windows kaggle-CLI bug: upload cache dir missing -> creating $(dirname "$missing")"
    mkdir -p "$(dirname "$missing")"
    eval "$cmd_str"
    return 0
  fi
  echo "$out" >&2
  return 1
}

sync_dataset() {
  local dir="$1" label="$2"
  echo "=== sync dataset: $label ($dir) ==="
  if [ ! -f "$dir/dataset-metadata.json" ]; then
    echo "ERROR: $dir/dataset-metadata.json missing (required for datasets create/version)" >&2
    exit 1
  fi
  if dataset_sync_retry "kaggle datasets version -p '$dir' -m 'automated sync'"; then
    return 0
  fi
  echo "version failed, trying create (first upload)..."
  dataset_sync_retry "kaggle datasets create -p '$dir' --dir-mode zip"
}

push_kernel() {
  local kdir="$1" accel="${2:-NvidiaTeslaT4}"
  local meta="$kdir/kernel-metadata.json"
  [ -f "$meta" ] || { echo "ERROR: no kernel-metadata.json in $kdir" >&2; exit 1; }
  local kid
  kid=$(python -c "import json;print(json.load(open(r'$meta'))['id'])")

  echo "=== push $kid (accelerator=$accel) ==="
  if ! (cd "$kdir" && kaggle kernels push --accelerator "$accel"); then
    echo "ERROR: push failed. If this is a 409 Conflict on re-push to an existing" >&2
    echo "  kernel id, the fix that worked before was bumping to a new kernel id" >&2
    echo "  in kernel-metadata.json (root cause not fully diagnosed)." >&2
    exit 1
  fi
  echo "$kid"
}

poll_kernel() {
  local kid="$1"
  echo "=== polling $kid every ${POLL_INTERVAL}s (Kaggle has no live log API) ===" >&2
  while true; do
    local raw status
    raw=$(kaggle kernels status "$kid" 2>&1) || true
    status=$(echo "$raw" | grep -oiE '"(complete|error|cancelled|running|queued)"' | tr -d '"' | tr '[:upper:]' '[:lower:]' | head -1)
    echo "$(date '+%H:%M:%S') $raw" >&2
    case "$status" in
      complete) echo "complete"; return 0 ;;
      error|cancelled) echo "$status"; return 1 ;;
    esac
    sleep "$POLL_INTERVAL"
  done
}

pull_output() {
  local kid="$1" dest="$2"
  mkdir -p "$dest"
  echo "=== pulling output for $kid -> $dest ==="
  kaggle kernels output "$kid" -p "$dest"
  echo "done: $(ls "$dest" | wc -l) files in $dest"
}

cmd="${1:-}"
case "$cmd" in
  push)
    check_auth
    kdir="${2:?usage: run_kaggle.sh push <kernel-dir> [accelerator]}"
    accel="${3:-NvidiaTeslaT4}"
    kid=$(push_kernel "$kdir" "$accel")
    if poll_kernel "$kid"; then
      pull_output "$kid" "$kdir/../ckpt_out_$(basename "$kid")"
    else
      dest="$kdir/../ckpt_out_$(basename "$kid")_FAILED"
      pull_output "$kid" "$dest"
      echo "kernel ended in error/cancelled state -- log pulled to $dest for inspection" >&2
      exit 1
    fi
    ;;
  sync-dataset)
    check_auth
    dir="${2:?usage: run_kaggle.sh sync-dataset <local-dir> <label>}"
    label="${3:-dataset}"
    sync_dataset "$dir" "$label"
    ;;
  pull)
    check_auth
    kid="${2:?usage: run_kaggle.sh pull <kernel-id> <dest-dir>}"
    dest="${3:?usage: run_kaggle.sh pull <kernel-id> <dest-dir>}"
    pull_output "$kid" "$dest"
    ;;
  *)
    echo "usage: $0 {push <kernel-dir> [accelerator] | sync-dataset <dir> <label> | pull <kernel-id> <dest-dir>}" >&2
    exit 1
    ;;
esac
