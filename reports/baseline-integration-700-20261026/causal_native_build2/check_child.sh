#!/usr/bin/env bash
set -euo pipefail
D=/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2
CASE=$1
export GDN_RUN_DIR="$D/$CASE" GDN_CASE_OWNER='baseline700-20261026/causal_native_build2'
date -u +%FT%TZ > "$GDN_RUN_DIR/child_started.txt"
cp /run/lock/ascend-npu/device-0.lock "$GDN_RUN_DIR/guard_running.json"
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
export ASCEND_CUSTOM_OPP_PATH="$D/install/vendors/causalpairs_transformer"
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$ASCEND_CUSTOM_OPP_PATH/op_api/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib"
export PATH="/opt/tileops-ci/runtime/bin:$ASCEND_HOME_PATH/bin:/usr/bin:/bin"
export TMPDIR="$GDN_RUN_DIR/tmp" XDG_CACHE_HOME="$GDN_RUN_DIR/xdg" ASCEND_WORK_PATH="$GDN_RUN_DIR/work" ASCEND_CACHE_PATH="$GDN_RUN_DIR/cache" ASCEND_PROCESS_LOG_PATH="$GDN_RUN_DIR/logs" TORCH_EXTENSIONS_DIR="$GDN_RUN_DIR/extensions" TRITON_CACHE_DIR="$GDN_RUN_DIR/triton"
export ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1
cd "$D"
exec /opt/tileops-ci/runtime/bin/python -S -B "$D/adapter/entry.py" --op CausalConv1dUpdateFwdOp --case-id "$CASE" --library ops-transformer --output "$GDN_RUN_DIR/checker.json"
