#!/bin/bash
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
lib="$1"
if [ "$lib" = tilelang-ascend ]; then
 export INTEGRATION700_TLA_ROUTE=legacy
 export TILELANG_LIBRARY_PATH="$root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
 export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
 export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
 exec /opt/tileops-all-pairs/bin/allpairs-python "$root/tools/baseline_retest32_server1_entry_v8.py" "${@:2}"
fi
if [ "$lib" = vllm-ascend ]; then
 exec /opt/tileops-all-pairs/bin/allpairs-python "$root/tools/baseline_retest32_server1_entry_v8.py" "${@:2}"
fi
if [ "$lib" = flash-linear-attention-npu ]; then
 export ASCEND_CLASSIC_FLA_PACKAGE="$root/.cache/old1076/fla/fla_npu"
 case_op=$(python3 -c 'import json,sys; a=sys.argv[1:]; print(json.load(open(a[a.index("--task")+1]))["key"][1])' "${@:2}")
 if [ "$case_op" = SSDChunkCumsumFwdOp ]; then
  export ASCEND_CLASSIC_FLA_PACKAGE=$root/reports/baseline-retest32-20261030/server1/deployments/fla_gate/fla_npu
 fi
 export ASCEND_CUSTOM_OPP_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer:$ASCEND_CLASSIC_FLA_PACKAGE/opp"
 export LD_LIBRARY_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
exec bash "$root/tools/baseline_cache_python.sh" "$lib" "$root/tools/baseline_retest32_server1_entry_v8.py" "${@:2}"
