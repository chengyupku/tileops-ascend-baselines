#!/bin/bash
# Parent-only future execution. This task does not run this recipe.
set -euo pipefail
ROOT=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_variant_registration1
mkdir -p "$ROOT/runs"
export BMM_VARIANT_RUN_ROOT
BMM_VARIANT_RUN_ROOT=$(mktemp -d "$ROOT/runs/run-XXXXXXXX")
touch "$BMM_VARIANT_RUN_ROOT/RUN_EXCLUSIVE"
export BMM_VARIANT_GUARD_OWNER="baseline700-20261026/threehost_bmm_variant_registration1/$(basename "$BMM_VARIANT_RUN_ROOT")"
exec npu-guard run --owner "$BMM_VARIANT_GUARD_OWNER" --wait 120 --timeout 360 -- bash "$ROOT/run_environment.sh" >"$BMM_VARIANT_RUN_ROOT/guard.log" 2>&1
