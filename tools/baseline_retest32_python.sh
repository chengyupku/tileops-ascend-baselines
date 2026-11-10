#!/bin/bash
set -e
retest_root="$(cd "$(dirname "$0")/.." && pwd)"
retest_lib="$1"
if [ "$retest_lib" = ops-nn ]; then
 source "$retest_root/.cache/all-pairs/remote_ops_math/remote_env.sh"
 export OPS_NN_VENDOR="$retest_root/reports/baseline-old1076-20261029/arm_nn_complete_subset/packages/vendors/allpairs_nn"
 export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
 export LD_LIBRARY_PATH="$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/es/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
exec bash "$retest_root/tools/baseline_cache_python.sh" "$@"
