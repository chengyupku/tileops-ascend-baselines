#!/bin/bash
set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:/root/dyq/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin"
export TMPDIR=/root/dyq/workspace/reports/baseline-single-agent-local-20261027/sum-original-build03/tmp
export ASCEND_CUSTOM_OPP_PATH=
cd /root/dyq/workspace/reports/baseline-single-agent-local-20261027/sum-original-source
exec bash build.sh --pkg --jit --ops=reduce_sum --soc=ascend950 --vendor_name=single_sum --ccache=off -j8
