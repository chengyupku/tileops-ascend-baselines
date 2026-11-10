#!/usr/bin/env bash
set -euo pipefail
cd /opt/tileops-all-pairs/workspace
sgl_root="$PWD"
export ASCEND_HOME_PATH="$sgl_root/.cache/all-pairs/cache-balance-sgl/sdk"
export ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
export LD_LIBRARY_PATH="$sgl_root/.cache/all-pairs/remote_host_builds/toolchain/lib64:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
cmake -S "$sgl_root/ascend_baselines/sgl-kernel-npu" \
  -B "$sgl_root/.cache/all-pairs/cache-balance-sgl/build" \
  -DBUILD_DEEPEP_MODULE=OFF -DBUILD_KERNELS_MODULE=ON -DBUILD_CATLASS_MODULE=OFF \
  -DBUILD_TESTS=OFF -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DASCEND_INCLUDE_DIR="$ASCEND_HOME_PATH/include" \
  -DPYTHON_EXECUTABLE="$sgl_root/reports/baseline-cache-20261027/balance-half-20261028/build-python" \
  -DCMAKE_CXX_COMPILER="$sgl_root/.cache/all-pairs/remote_host_builds/toolchain/bin/g++"
