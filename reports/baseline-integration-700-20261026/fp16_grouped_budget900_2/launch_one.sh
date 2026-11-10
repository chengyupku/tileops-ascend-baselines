#!/bin/bash
set -eu
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fp16_grouped_budget900_2
CASE="$1"
test -f "$P/PARENT_RELEASE.json"
export BF16_GEMM_GUARD_OWNER=baseline700-20261026/fp16_grouped_budget900_2
export BF16_GEMM_RUN_ROOT="$P/$CASE/runtime"
exec bash "$P/environment.sh" /root/dyq/workspace/.venv/bin/python -B "$P/capture.py" "$P/$CASE/result" GroupedGemmFwdOp "$CASE"
