#!/usr/bin/env bash
set -euo pipefail
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLA_NPU_DISABLE_PTH=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/runtime-cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/runtime-cache/torch-extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/runtime-cache/xdg}"
export TMPDIR="${TMPDIR:-/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/tmp}"
export TRITON_BACKEND=torch_npu
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1
export LD_LIBRARY_PATH=/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib:/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:/usr/local/Ascend/cann-9.1.0/compiler/ccec_compiler/bin:/usr/bin:$PATH

cd /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1
export ASCEND_CUSTOM_OPP_PATH=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1/private_provider
export ASCEND_CLASSIC_TRANSFORMER_VENDOR=$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_GLOBAL_LOG_LEVEL=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
for name in ASCEND_CACHE_PATH ASCEND_WORK_PATH XDG_CACHE_HOME TORCH_EXTENSIONS_DIR ASCEND_PROCESS_LOG_PATH TRITON_CACHE_DIR TMPDIR TMP TEMP ACL_OP_COMPILER_CACHE_DIR; do
 export "$name=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1/square-4k-bfloat16/$name"
 mkdir -p "${!name}"
done
exec /opt/tileops-ci/runtime/bin/python -I -S -B /opt/tileops-all-pairs/control/private_python.py /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1/capture.py /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1/square-4k-bfloat16 BmmFwdOp square-4k-bfloat16
