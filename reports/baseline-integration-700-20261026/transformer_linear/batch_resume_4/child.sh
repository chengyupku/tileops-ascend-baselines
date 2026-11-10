#!/bin/bash
set -eo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear"
export GDN_RUN_DIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16"
export ASCEND_HOME_PATH="/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_ops_math_continuation/sdk920-original-layout"
export ASCEND_TOOLKIT_HOME="/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_ops_math_continuation/sdk920-original-layout"
export ASCEND_OPP_PATH="/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/gdn-original-install-validate-group/opp"
export ASCEND_CUSTOM_OPP_PATH="/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/recovery0730-gdn-fixed-abi/chunk_binary/install/vendors/kdapairs_transformer"
export PYTHONDONTWRITEBYTECODE="1"
export TORCH_DEVICE_BACKEND_AUTOLOAD="0"
export GDN_CASE_OWNER="baseline700-20261026/transformer_linear"
export ASCEND_CUSTOM_OPP_PATH=""
export ASCEND_CUSTOM_OPP_PATH="/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/recovery0730-gdn-fixed-abi/chunk_binary/install/vendors/kdapairs_transformer"
export LD_LIBRARY_PATH="/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_ops_math_continuation/sdk920-original-layout/lib64:/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/gdn-opp-repair-cpu/opapi-lib:/usr/local/Ascend/cann/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/lib:/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/recovery0730-gdn-fixed-abi/chunk_binary/install/vendors/kdapairs_transformer/op_api/lib"
export TMPDIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/TMPDIR"
export PIP_CACHE_DIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/PIP_CACHE_DIR"
export XDG_CACHE_HOME="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/XDG_CACHE_HOME"
export XDG_CONFIG_HOME="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/XDG_CONFIG_HOME"
export XDG_DATA_HOME="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/XDG_DATA_HOME"
export ASCEND_PROCESS_LOG_PATH="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/ASCEND_PROCESS_LOG_PATH"
export ASCEND_WORK_PATH="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/ASCEND_WORK_PATH"
export ASCEND_CACHE_PATH="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/ASCEND_CACHE_PATH"
export TORCH_EXTENSIONS_DIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/TORCH_EXTENSIONS_DIR"
export TRITON_CACHE_DIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/TRITON_CACHE_DIR"
export CCACHE_DIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp4-8x1024-bfloat16/CCACHE_DIR"
export FLA_NPU_DISABLE_PTH=1 PYTHONNOUSERSITE=1
exec /opt/tileops-ci/runtime/bin/python -S -B "/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/batch_entry.py" q35-397b-tp4-8x1024-bfloat16 q35-397b-tp8-32768-bfloat16 q35-397b-tp4-16384-bfloat16
