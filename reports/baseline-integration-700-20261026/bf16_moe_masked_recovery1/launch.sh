#!/bin/bash
set -eu
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_moe_masked_recovery1
test -f "$P/PARENT_RELEASE.json"
export BF16_GEMM_GUARD_OWNER=baseline700-20261026/bf16_moe_masked_recovery1
export BF16_GEMM_RUN_ROOT="$P/runtime"
exec bash "$P/environment.sh" /root/dyq/workspace/.venv/bin/python -B "$P/capture.py" "$P/result" MoEGroupedGemmFwdOp ds-v3-ep8-masked-bfloat16
