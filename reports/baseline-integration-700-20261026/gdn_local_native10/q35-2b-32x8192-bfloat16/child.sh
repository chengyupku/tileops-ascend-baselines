#!/bin/bash
set -eo pipefail
cd /root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export GDN_RUN_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.2.0-beta.2/opp
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export GDN_CASE_OWNER=baseline700-20261026/gdn_local_native10
export PYTHONPATH=/root/dyq/workspace/.venv/lib/python3.12/site-packages:/usr/local/Ascend/cann-9.2.0-beta.2/python/site-packages:/usr/local/Ascend/cann-9.2.0-beta.2/opp/built-in/op_impl/ai_core/tbe
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.0-beta.2/lib64:/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/install/vendors/kdapairs_transformer/op_api/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib:/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib
export FLA_NPU_DISABLE_PTH=1
export PYTHONNOUSERSITE=1
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TMPDIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/TMPDIR
export PIP_CACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/PIP_CACHE_DIR
export XDG_CACHE_HOME=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/XDG_CACHE_HOME
export XDG_CONFIG_HOME=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/XDG_CONFIG_HOME
export XDG_DATA_HOME=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/XDG_DATA_HOME
export ASCEND_PROCESS_LOG_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/ASCEND_PROCESS_LOG_PATH
export ASCEND_WORK_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/ASCEND_WORK_PATH
export ASCEND_CACHE_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/ASCEND_CACHE_PATH
export TORCH_EXTENSIONS_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/TRITON_CACHE_DIR
export CCACHE_DIR=/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/CCACHE_DIR
exec strace --seccomp-bpf -f -qq -e trace=openat -o /root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/native-open.trace /root/dyq/workspace/.venv/bin/python -S -B /root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/case_launcher.py --op GDNFwdOp --case-id q35-2b-32x8192-bfloat16 --library ops-transformer --output /root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_local_native10/q35-2b-32x8192-bfloat16/checker.json
