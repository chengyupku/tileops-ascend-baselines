#!/usr/bin/env bash
set -e
old1076_root="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$1" = ops-transformer ] && { [ "$(basename "$2")" = baseline_old1076_transformer.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_pad_v2.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_reset_v2.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_sequence_v3.py ]; }; then
    export OLD1076_TRANSFORMER_VENDOR="$old1076_root/.cache/old1076/linear_recovery_transformer"
    export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"
    export ASCEND_CLASSIC_TRANSFORMER_VENDOR="$OLD1076_TRANSFORMER_VENDOR"
    export LD_LIBRARY_PATH="$OLD1076_TRANSFORMER_VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"
    shift
    exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$1" = ops-nn ] && [ "$(basename "$2")" = baseline_old1076_nn_arm_v3.py ]; then
    export OPS_NN_VENDOR="$old1076_root/.cache/old1076/nn_original_arm"
    export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
    shift
    exec bash /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/environment.sh /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$1" = ops-nn ] && { [ "$(basename "$2")" = baseline_old1076_nn_arm_v1.py ] || [ "$(basename "$2")" = baseline_old1076_nn_arm_v2.py ]; }; then
    export OPS_NN_VENDOR=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/vendor/allpairs_nn
    export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
    shift
    exec bash /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/environment.sh /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$1" = flash-linear-attention-npu ]; then
    export ASCEND_CLASSIC_FLA_PACKAGE="$old1076_root/.cache/old1076/fla/fla_npu"
    export ASCEND_CUSTOM_OPP_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer:$ASCEND_CLASSIC_FLA_PACKAGE/opp"
    export LD_LIBRARY_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
    export FLA_ALLPAIRS_MODE=gdn
    shift
    exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$1" = tilelang-ascend ]; then
    if [ "$(basename "$2")" = baseline_old1076_pto_entry_v1.py ] || [ "$(basename "$2")" = baseline_old1076_pto_entry_v2.py ]; then
        shift
        exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
    fi
    export INTEGRATION700_TLA_ROUTE=legacy
    export TILELANG_LIBRARY_PATH="$old1076_root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
    export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
    export PATH="$old1076_root/.cache/all-pairs/remote_tla_compiler/handoff/bin:$PATH"
    export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
    if [ "$(basename "$2")" = measure_baseline_cache_pair.py ]; then
        exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_old1076_tla_native_remote_v1.py" "$@"
    fi
    if [ "$(basename "$2")" = baseline_old1076_tla_argmin_v1.py ]; then
        exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_old1076_tla_argmin_remote_v1.py" "$@"
    fi
    if [ "$(basename "$2")" = baseline_old1076_tla_argmax_v3.py ]; then
        exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_old1076_tla_argmax_remote_v3.py" "$@"
    fi
    if [ "$(basename "$2")" = baseline_old1076_tla_argmax_v2.py ]; then
        exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_old1076_tla_argmax_remote_v2.py" "$@"
    fi
    if [ "$(basename "$2")" = baseline_old1076_tla_block_v3.py ]; then
        exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_old1076_tla_remote_v3.py" "$@"
    fi
    if [ "$(basename "$2")" = baseline_old1076_tla_block_v2.py ]; then
        exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_old1076_tla_remote_v2.py" "$@"
    fi
    exec /opt/tileops-all-pairs/bin/allpairs-python "$old1076_root/tools/baseline_support_tla_remote_launch_v3.py" "$@"
fi
exec bash "$old1076_root/tools/baseline_cache_python.sh" "$@"
