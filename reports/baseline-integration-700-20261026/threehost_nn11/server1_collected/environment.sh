set -eu
export NN_WORKSPACE=/opt/tileops-all-pairs/workspace
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLA_NPU_DISABLE_PTH=1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
export PATH=/opt/tileops-ci/runtime/bin:/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:/usr/local/Ascend/cann-9.1.0/compiler/ccec_compiler/bin:$PATH
NN_PRIVATE_VENDOR=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/vendor/allpairs_nn
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/lib64:$NN_PRIVATE_VENDOR/op_api/lib:$NN_PRIVATE_VENDOR/op_proto/es/lib/linux/aarch64:$NN_PRIVATE_VENDOR/op_proto/lib/linux/aarch64:$NN_PRIVATE_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib:/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
exec "$@"
