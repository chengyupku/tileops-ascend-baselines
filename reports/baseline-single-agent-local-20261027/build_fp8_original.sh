#!/bin/bash
set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export PATH="$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:/root/dyq/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-vendor
export ASCEND_PROCESS_LOG_PATH=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01/ASCEND_PROCESS_LOG_PATH
export ASCEND_CACHE_PATH=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01/ASCEND_CACHE_PATH
export ASCEND_WORK_PATH=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01/ASCEND_WORK_PATH
export XDG_CACHE_HOME=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01/XDG_CACHE_HOME
export TMPDIR=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01/tmp
cd /root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01
exec /usr/local/Ascend/cann-9.2.0-beta.2/bin/asc_opc /root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-vendor/op_impl/ai_core/tbe/bf16gmm_transformer_impl/dynamic/grouped_matmul_apt.py --main_func=grouped_matmul --input_param=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/fp8_original_param.json --soc_version=Ascend950PR_9599 --output=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/ops-transformer-fp8-build01/binary --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic
