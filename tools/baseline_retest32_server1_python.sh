#!/bin/bash
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
lib="$1"
if [ "$lib" = flash-linear-attention-npu ]; then
 export ASCEND_CLASSIC_FLA_PACKAGE="$root/.cache/old1076/fla/fla_npu"
 export ASCEND_CUSTOM_OPP_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer:$ASCEND_CLASSIC_FLA_PACKAGE/opp"
 export LD_LIBRARY_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
exec bash "$root/tools/baseline_cache_python.sh" "$lib" "$root/tools/baseline_retest32_server1_entry.py" "${@:2}"
