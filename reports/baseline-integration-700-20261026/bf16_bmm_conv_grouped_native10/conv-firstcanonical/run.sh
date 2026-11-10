#!/bin/bash
set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH ASCEND_AICPU_PATH=$ASCEND_HOME_PATH ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer
export ASCEND_CLASSIC_TRANSFORMER_VENDOR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer
export LD_LIBRARY_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_GLOBAL_LOG_LEVEL=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export ASCEND_CACHE_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/ASCEND_CACHE_PATH
export ASCEND_WORK_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/ASCEND_WORK_PATH
export XDG_CACHE_HOME=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/XDG_CACHE_HOME
export TORCH_EXTENSIONS_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/TORCH_EXTENSIONS_DIR
export ASCEND_PROCESS_LOG_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/ASCEND_PROCESS_LOG_PATH
export TRITON_CACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/TRITON_CACHE_DIR
export TMPDIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/TMPDIR
export TMP=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/TMP
export TEMP=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/TEMP
export ACL_OP_COMPILER_CACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/ACL_OP_COMPILER_CACHE_DIR
export CCACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical/CCACHE_DIR
cd /root/dyq/workspace
exec /root/dyq/workspace/.venv/bin/python /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/capture_conv.py /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/conv-firstcanonical whisper-conv1-bias-bfloat16
