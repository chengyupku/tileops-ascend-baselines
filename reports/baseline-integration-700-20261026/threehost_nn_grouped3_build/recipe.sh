#!/usr/bin/env bash
source /root/dyq/workspace/env.sh
export FLA_NPU_DISABLE_PTH=1
export PYTHONDONTWRITEBYTECODE=1
export OPS_NN_VENDOR=/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn
export ASCEND_CLASSIC_NN_VENDOR="$OPS_NN_VENDOR"
export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
