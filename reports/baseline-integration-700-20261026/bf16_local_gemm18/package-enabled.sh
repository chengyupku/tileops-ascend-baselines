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
/usr/local/Ascend/cann-9.2.0-beta.2/tools/bisheng_compiler/bin/bisheng --version > /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/COMPILER_VERSION.txt
cd /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/source
bash build.sh --pkg --jit --soc=ascend950 --ops=grouped_matmul --vendor_name=bf16gmm -j8 --build-type=Release --ccache off
