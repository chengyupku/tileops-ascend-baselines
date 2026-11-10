#!/usr/bin/env bash
set -e
old1076_root="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$(basename "$2")" = baseline_old1076_sgl_decode_v2.py ]; then
 shift
 exec "$old1076_root/.tools/baselines/baseline-run" sgl-kernel-npu python "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_vllm_local_v2.py ] || [ "$(basename "$2")" = baseline_old1076_vllm_recurrent_v3.py ]; then
 shift
 exec "$old1076_root/.tools/baselines/baseline-run" vllm-ascend python "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_tilelang_original_v2.py ]; then
 shift
 exec "$old1076_root/.tools/baselines/baseline-run" tilelang python "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_transformer_graph_cleanup_v1.py ]; then
 export BASELINE_CACHE_EXACT_ENV_SELECTED=1
 exec bash "$old1076_root/tools/baseline_cache_python.sh" "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_tla_original_target_v1.py ] || [ "$(basename "$2")" = baseline_old1076_tla_local_provider_v2.py ] || [ "$(basename "$2")" = baseline_old1076_tla_bound_local_v3.py ]; then
 source "$old1076_root/env.sh"
 export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
 shift
 exec "$old1076_root/.venv/bin/python" "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_fp8_bmm_fp16_v6.py ] || [ "$(basename "$2")" = baseline_old1076_fp8_padded_v4.py ]; then
 source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
 export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
 export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
 export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
 export ASCEND_CUSTOM_OPP_PATH="$old1076_root/reports/baseline-old1076-20261029/fp8_fp16_deployment/ops-transformer-vendor"
 export ASCEND_CLASSIC_TRANSFORMER_VENDOR="$ASCEND_CUSTOM_OPP_PATH"
 export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
 shift
 exec "$old1076_root/.venv/bin/python" "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_fp8_moe_v5.py ] || [ "$(basename "$2")" = baseline_old1076_fp8_bmm_v3.py ] || [ "$(basename "$2")" = baseline_old1076_fp8_padded_v1.py ] || [ "$(basename "$2")" = baseline_old1076_fp8_padded_v2.py ]; then
 shift
 exec bash "$old1076_root/reports/baseline-single-agent-local-20261027/ops-transformer-python" "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_nn_acl_dequant_v8.py ] || [ "$(basename "$2")" = baseline_old1076_nn_arm_staging_v3.py ] || [ "$(basename "$2")" = baseline_old1076_nn_arm_prelu_v4.py ] || [ "$(basename "$2")" = baseline_old1076_nn_acl_list_v5.py ] || [ "$(basename "$2")" = baseline_old1076_nn_acl_list_v6.py ] || [ "$(basename "$2")" = baseline_old1076_nn_acl_list_v7.py ]; then
 export OPS_NN_VENDOR="$old1076_root/reports/baseline-old1076-20261029/arm_nn_complete_subset/packages/vendors/allpairs_nn"
 export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
 export LD_LIBRARY_PATH="$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/es/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_nn_arm_staging_v2.py ]; then
 export OPS_NN_VENDOR="$old1076_root/reports/baseline-old1076-20261029/arm_nn_fresh_subset/packages/vendors/allpairs_nn"
 export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
 export LD_LIBRARY_PATH="$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/es/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_nn_arm_staging_v1.py ]; then
 export OPS_NN_VENDOR="$old1076_root/reports/baseline-old1076-20261029/arm_nn_install/vendors/allpairs_nn"
 export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
 export LD_LIBRARY_PATH="$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/es/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$(basename "$2")" = baseline_old1076_transformer_arm_fp8_v2.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_arm_graph_v4.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_arm_fp8_v1.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_arm_graph_v3.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_arm_graph_v2.py ] || [ "$(basename "$2")" = baseline_old1076_transformer_arm_graph_v1.py ]; then
 export ASCEND_CUSTOM_OPP_PATH="$old1076_root/reports/baseline-old1076-20261029/arm_graph_root/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer"
 export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
 shift
 exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
if [ "$old1076_root" = /opt/tileops-baseline-cache/workspace ]; then
    exec bash "$old1076_root/tools/baseline_old1076_python.sh" "$@"
fi
exec bash "$old1076_root/tools/baseline_cache_python.sh" "$@"
