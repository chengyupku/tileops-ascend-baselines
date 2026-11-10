#!/bin/bash
set -e
experiment_root="$(cd "$(dirname "$0")/.." && pwd)"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH="/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export OPS_NN_VENDOR="$experiment_root/reports/baseline-old1076-20261029/arm_nn_complete_subset/packages/vendors/allpairs_nn"
experiment_op="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"][1])' "$3")"
if [ "$experiment_op" = BmmFwdOp ] || [ "$experiment_op" = GemmFwdOp ]; then
 if [ -d "$experiment_root/reports/baseline-bmm-rootcause-20261030/allpairs_nn" ]; then
  export OPS_NN_VENDOR="$experiment_root/reports/baseline-bmm-rootcause-20261030/allpairs_nn"
 else
  export OPS_NN_VENDOR=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/vendor/allpairs_nn
 fi
fi
if [ "$(basename "$OPS_NN_VENDOR")" != allpairs_nn ] || [ ! -d "$OPS_NN_VENDOR/op_impl/ai_core/tbe/allpairs_nn_impl" ]; then
 echo "Vendor directory must match allpairs_nn_impl; refusing ambiguous compilation route" >&2
 exit 1
fi
export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
export ASCEND_CLASSIC_NN_VENDOR="$OPS_NN_VENDOR"
export LD_LIBRARY_PATH="$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/es/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 FLA_NPU_DISABLE_PTH=1
exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
