#!/bin/bash
set -euo pipefail
ROOT=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration
if [[ $# != 2 ]]; then echo 'usage: run_case.sh OP CASE_ID' >&2; exit 2; fi
mkdir -p "$ROOT/runs"
export NATIVE9_RUN_ROOT
NATIVE9_RUN_ROOT=$(mktemp -d "$ROOT/runs/run-XXXXXXXX")
touch "$NATIVE9_RUN_ROOT/RUN_EXCLUSIVE"
export NATIVE9_GUARD_OWNER="native9-registration/$(basename "$NATIVE9_RUN_ROOT")"
exec npu-guard run --owner "$NATIVE9_GUARD_OWNER" --wait 600 --timeout 300 -- bash "$ROOT/run_environment.sh" "$1" "$2" >"$NATIVE9_RUN_ROOT/guard.log" 2>&1
