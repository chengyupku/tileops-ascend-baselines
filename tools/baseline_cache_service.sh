#!/usr/bin/env bash
# CPU-only process supervisor; every device child still runs in npu-guard.
set -u
baseline_cache_workspace="$(cd "$(dirname "$0")/.." && pwd)"
while true; do
    /usr/bin/python3 "$baseline_cache_workspace/tools/run_baseline_cache.py" "$@"
    baseline_cache_exit=$?
    if [ "$baseline_cache_exit" = 0 ]; then exit 0; fi
    sleep 15
done
