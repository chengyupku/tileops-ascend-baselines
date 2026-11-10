#!/bin/bash
set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH ASCEND_AICPU_PATH=$ASCEND_HOME_PATH ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/bin:$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:/root/dyq/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin
export CMAKE_PREFIX_PATH=$ASCEND_HOME_PATH/lib64/cmake ASC_DIR=$ASCEND_HOME_PATH/lib64/cmake
export TMPDIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/tmp MAKE_AUDIT=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/make-audit.jsonl PYTHONDONTWRITEBYTECODE=1
export ASCEND_PROCESS_LOG_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/build-cache/ASCEND_PROCESS_LOG_PATH
export ASCEND_WORK_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/build-cache/ASCEND_WORK_PATH
export ASCEND_CACHE_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/build-cache/ASCEND_CACHE_PATH
export XDG_CACHE_HOME=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/build-cache/XDG_CACHE_HOME
export CCACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/build-cache/CCACHE_DIR


export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer
cd /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/device_fp32
/usr/local/Ascend/cann-9.2.0-beta.2/bin/asc_opc /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_impl/ai_core/tbe/bf16gmm_transformer_impl/dynamic/grouped_matmul_apt.py --main_func=grouped_matmul --input_param=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/remote_original/generated/GroupedMatmul_e84aaec0eccf7b8304da51ea5f5a2076_param.json --soc_version=Ascend950PR_9599 --output=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/device_fp32 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic
