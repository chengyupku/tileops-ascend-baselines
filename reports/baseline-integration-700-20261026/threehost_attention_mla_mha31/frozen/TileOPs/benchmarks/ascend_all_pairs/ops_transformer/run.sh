#!/usr/bin/env bash
set -e
source /root/dyq/workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1
export FLA_NPU_DISABLE_PTH=1
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/allpairs_transformer:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/causalpairs_transformer:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/kdapairs_transformer:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/attentionpairs_transformer:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/mlapairs_transformer:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/selectionpairs_transformer:$ASCEND_CUSTOM_OPP_PATH"
export LD_LIBRARY_PATH="/root/dyq/workspace/.cache/all-pairs/ops_transformer/install/vendors/allpairs_transformer/op_api/lib:$LD_LIBRARY_PATH"
export ASCEND_WORK_PATH=/root/dyq/workspace/.cache/all-pairs/ops_transformer
export ASCEND_CACHE_PATH=/root/dyq/workspace/.cache/all-pairs/ops_transformer/cann
export ASCEND_PROCESS_LOG_PATH=/root/dyq/workspace/.cache/all-pairs/ops_transformer/logs
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_EXTENSIONS_DIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/torch_extensions
export TRITON_CACHE_DIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/triton
export TMPDIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/tmp
mkdir -p "$TMPDIR" "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH"
cd /root/dyq/workspace/.cache/all-pairs/ops_transformer
mkdir -p /root/dyq/workspace/.cache/all-pairs/ops_transformer/console
exec python /root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/check.py "$@" > "/root/dyq/workspace/.cache/all-pairs/ops_transformer/console/check-$$.log" 2>&1
