#!/bin/bash
# Genuine dependency selection, matching successful pristine configure.
BUILD_OWN=/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/sgl_openmp_host_finish
SGL_SOURCE=/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source
SGL_OVERLAY=/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages
SGL_TOOLCHAIN=/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_host_builds/toolchain
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export PATH="/opt/tileops-ci/runtime/bin:$SGL_TOOLCHAIN/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
export PYTHONPATH="$SGL_SOURCE/python/sgl_kernel_npu:$SGL_OVERLAY"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$SGL_TOOLCHAIN/lib64:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_host_builds/deps/usr/lib/aarch64-linux-gnu:$SGL_OVERLAY/torch/lib:$SGL_OVERLAY/torch_npu/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$ASCEND_HOME_PATH/lib64"
export TMPDIR="$BUILD_OWN/tmp" XDG_CACHE_HOME="$BUILD_OWN/cache" ASCEND_PROCESS_LOG_PATH="$BUILD_OWN/ascend-logs" ASCEND_WORK_PATH="$BUILD_OWN/ascend-work"
export CPATH="$BUILD_OWN/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$BUILD_OWN/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
