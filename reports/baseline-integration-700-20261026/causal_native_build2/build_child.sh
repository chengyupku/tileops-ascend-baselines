#!/usr/bin/env bash
set -euo pipefail
D=/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2
date -u +%FT%TZ > "$D/build_child_start.txt"
cp /run/lock/ascend-npu/device-0.lock "$D/build_guard_running.json"
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
export PATH="$D/bin:/opt/tileops-ci/runtime/bin:$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
export PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8
export TMPDIR="$D/runtime/tmp" XDG_CACHE_HOME="$D/runtime/xdg" TORCH_EXTENSIONS_DIR="$D/runtime/torch_extensions" TRITON_CACHE_DIR="$D/runtime/triton" ASCEND_PROCESS_LOG_PATH="$D/runtime/logs" ASCEND_CACHE_PATH="$D/runtime/cann" ASCEND_WORK_PATH="$D/runtime"
export PIP_FIND_LINKS="$D/build_deps" PIP_CACHE_DIR="$D/runtime/pip"
export PYTHONPATH="$D/build_deps/site-packages:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages:$ASCEND_HOME_PATH/python/site-packages"
export GIT_CEILING_DIRECTORIES="$D/source"
cd "$D/source"
exec bash build.sh --pkg --soc=ascend950 --ops=causal_conv1d --vendor_name=causalpairs -j8 --build-type=Release --ccache off
