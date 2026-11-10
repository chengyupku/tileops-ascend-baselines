#!/usr/bin/env bash
set -e
source /root/dyq/workspace/env.sh
export PYTHONDONTWRITEBYTECODE=1
export FLA_NPU_DISABLE_PTH=1
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH"
export ASCEND_WORK_PATH=/root/dyq/workspace/.cache/all-pairs/ops_transformer
export ASCEND_CACHE_PATH=/root/dyq/workspace/.cache/all-pairs/ops_transformer/cann
export TORCH_EXTENSIONS_DIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/torch_extensions
export TRITON_CACHE_DIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/triton
export TMPDIR=/root/dyq/workspace/.cache/all-pairs/ops_transformer/tmp
mkdir -p "$TMPDIR" "$ASCEND_CACHE_PATH"
cd /root/dyq/workspace/.cache/all-pairs/ops_transformer
exec python /root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/check.py "$@"
