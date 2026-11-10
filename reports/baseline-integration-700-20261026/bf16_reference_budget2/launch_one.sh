#!/bin/bash
set -eu
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_reference_budget2
OP="$1"
CASE="$2"
test -f "$P/PARENT_RELEASE.json"
export BF16_GEMM_GUARD_OWNER=baseline700-20261026/bf16_reference_budget2
export BF16_GEMM_RUN_ROOT="$P/$CASE/runtime"
exec bash "$P/environment.sh" /root/dyq/workspace/.venv/bin/python -B "$P/capture.py" "$P/$CASE/result" "$OP" "$CASE"
