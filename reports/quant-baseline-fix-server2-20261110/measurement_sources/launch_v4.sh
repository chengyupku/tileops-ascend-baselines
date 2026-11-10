#!/bin/bash
set -eo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
export LD_LIBRARY_PATH=/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/lib:/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/lib:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/ccec_compiler/bin:$PATH
export TORCHINDUCTOR_NPU_BACKEND=ascendc
export TORCHINDUCTOR_CACHE_DIR=/opt/quant-baseline-fix-server2-20261110/cache/inductor
export ASCEND_CACHE_PATH=/opt/quant-baseline-fix-server2-20261110/cache/ascend
unset ASCEND_CUSTOM_OPP_PATH
if [ "${3:-}" = "ops_math" ]; then
 export OPS_MATH_VENDOR_ROOT=/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_ops_math/graph-runtime/vendors/allpairs_math_math
 export ASCEND_CUSTOM_OPP_PATH=$OPS_MATH_VENDOR_ROOT
 export OPS_MATH_GRAPH_CALLS=/opt/quant-baseline-fix-server2-20261110/ops-math-calls-${2}-${4:-a1}.jsonl
fi
exec /opt/tileops-ci/runtime/bin/python -I -S "$(dirname "$0")/runner_v4.py" "$@"
