#!/bin/bash
set -eu
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fp16_local_remaining3
case "$1/$2" in
 BmmFwdOp/square-2k-float16|GroupedGemmFwdOp/moe-dgrad-float16|GroupedGemmFwdOp/moe-fwd-float16) ;;
 *) echo "Outside exact scope" >&2; exit 2;;
esac
export BF16_GEMM_GUARD_OWNER=baseline700-20261026/fp16_local_remaining3
export BF16_GEMM_RUN_ROOT="$P/$2/runtime"
exec bash "$P/environment.sh" /root/dyq/workspace/.venv/bin/python -B "$P/capture.py" "$P/$2/result" "$1" "$2"
