#!/usr/bin/env bash
set -euo pipefail
source /root/dyq/workspace/env.sh
export FLA_NPU_DISABLE_PTH=1
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn
export OPS_NN_VENDOR="$ASCEND_CUSTOM_OPP_PATH"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
export ASCEND_CACHE_PATH=/root/dyq/workspace/.cache/all-pairs/ops_nn/kernel_cache
export ASCEND_WORK_PATH=/root/dyq/workspace/.cache/all-pairs/ops_nn/work
export TMPDIR=/root/dyq/workspace/.cache/all-pairs/ops_nn/tmp
mkdir -p "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" "$TMPDIR"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=/root/dyq/workspace/.cache/all-pairs/ops_nn/logs
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
exec python /root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_nn/check.py "$@"
