#!/bin/bash
set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 ASCEND_GLOBAL_LOG_LEVEL=0
V=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer
export ASCEND_CUSTOM_OPP_PATH=$V ASCEND_CLASSIC_TRANSFORMER_VENDOR=$V LD_LIBRARY_PATH=$V/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_PROCESS_LOG_PATH=$1/ASCEND_PROCESS_LOG_PATH
exec /root/dyq/workspace/.venv/bin/python /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_moe_hadamard_native10/capture.py "$1" "$2" "$3"
