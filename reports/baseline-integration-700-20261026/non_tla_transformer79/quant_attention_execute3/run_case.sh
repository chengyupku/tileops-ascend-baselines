#!/bin/bash
set -e
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/non_tla_transformer79/quant_attention_execute3
SDK=/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_tilekernels_adaptive_norm02/sdk_view_v10
OVERLAY=/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages
export ASCEND_HOME_PATH="$SDK" ASCEND_OPP_PATH="$SDK/opp" ASCEND_AICPU_PATH="$SDK"
export LD_LIBRARY_PATH="$OVERLAY/torch/lib:$OVERLAY/torch_npu/lib:$SDK/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
export PATH="$SDK/tools/bisheng_compiler/bin:$SDK/compiler/ccec_compiler/bin:/opt/tileops-ci/runtime/bin:/usr/local/bin:/usr/bin:/bin"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$P/cache/torch" XDG_CACHE_HOME="$P/cache/xdg" TMPDIR="$P/tmp" ASCEND_CACHE_PATH="$P/cache/ascend" ASCEND_WORK_PATH="$P/cache/work" ASCEND_PROCESS_LOG_PATH="$P/log"
cd "$P"
exec /opt/tileops-ci/runtime/bin/python -I -S -B "$P/runner.py" "$1"
