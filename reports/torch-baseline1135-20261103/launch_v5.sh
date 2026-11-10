#!/bin/bash
set -eo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
export LD_LIBRARY_PATH=/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/lib:/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/lib:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/ccec_compiler/bin:$PATH
exec /opt/tileops-ci/runtime/bin/python -I -S "$(dirname "$0")/worker_v5.py" "$@"
