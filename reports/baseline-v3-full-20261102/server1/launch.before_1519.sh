#!/bin/bash
set -e
exp_dir=$(cd "$(dirname "$0")" && pwd)
root=/opt/tileops-baseline-cache/workspace
lib=$1
shift
if [ "$lib" = ops-transformer ] && [[ "$*" == *295741e4f4e994e54294* ]]; then
 export OLD1076_TRANSFORMER_VENDOR="$root/.cache/old1076/linear_recovery_transformer"
 export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR" ASCEND_CLASSIC_TRANSFORMER_VENDOR="$OLD1076_TRANSFORMER_VENDOR"
 export LD_LIBRARY_PATH="$OLD1076_TRANSFORMER_VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$exp_dir/old_transformer_entry.py" "$@"
fi
if [ "$lib" = sgl-kernel-npu ] && [[ "$*" == *31a4af28f480c43cb360* ]]; then
 export LD_LIBRARY_PATH="$root/.cache/all-pairs/sgl-ascend-runtime:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$exp_dir/old_sgl_entry.py" "$@"
fi
if [ "$lib" = tilelang ]; then
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$exp_dir/tilelang_entry.py" "$@"
fi
if [ "$lib" = vllm-ascend ] || [ "$lib" = tilelang-ascend ] || [ "$lib" = catlass ] || [ "$lib" = ops-tensor ]; then
 if [ "$lib" = vllm-ascend ]; then
  export VLLM_ALL_PAIRS_OPP="$root/.cache/all-pairs/vllm/install/vendors/custom_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$VLLM_ALL_PAIRS_OPP"
  export LD_LIBRARY_PATH="$VLLM_ALL_PAIRS_OPP/op_api/lib:${LD_LIBRARY_PATH:-}"
  export VLLM_ALL_PAIRS_EXTENSION="$root/.cache/all-pairs/vllm/install/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
 fi
 if [ "$lib" = tilelang-ascend ]; then
  export INTEGRATION700_TLA_ROUTE=legacy
  export TILELANG_LIBRARY_PATH="$root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
  export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
  export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
 fi
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$exp_dir/provider_entry.py" "$@"
fi
if [ "$lib" = ops-nn ]; then
 export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
 export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp ASCEND_AICPU_PATH=$ASCEND_HOME_PATH
 export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1
 export OPS_NN_VENDOR="$exp_dir/deps/reports/baseline-retest32-20261030/server2/original_nn950_complete_v12/vendors/allpairs_nn"
 case_op=$(python3 -c 'import json,sys;a=sys.argv[1:];print(json.load(open(a[a.index("--task")+1]))["key"][1])' "$@")
 if [ "$case_op" = GemmFwdOp ]; then
  export OPS_NN_VENDOR="$exp_dir/original_nn950_historical_v17/allpairs_nn"
 fi
 export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR" ASCEND_CLASSIC_NN_VENDOR="$OPS_NN_VENDOR"
 export LD_LIBRARY_PATH="/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/lib:/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/lib:$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/es/lib/linux/aarch64:$ASCEND_HOME_PATH/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
 export PATH="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/compiler/ccec_compiler/bin:$PATH"
 entry=nn_conv_entry.py
 case_op=$(python3 -c 'import json,sys;a=sys.argv[1:];print(json.load(open(a[a.index("--task")+1]))["key"][1])' "$@")
 case "$case_op" in GemmW4A16FwdOp|GemmFP8FwdOp) entry=nn_w4_entry.py;; esac
 shift
 exec /opt/tileops-ci/runtime/bin/python -I -B "$exp_dir/$entry" "$@"
fi
exec bash "$root/tools/baseline_retest32_server1_python_v72.sh" "$lib" "$@"
