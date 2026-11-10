#!/bin/bash
set -e
cd "/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn_grouped3_build"
source ./env.sh
exec cmake -S source -B build -DASC_DIR="/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn_grouped3_build/asc_cmake" -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=OFF -DENABLE_CUSTOM=ON -DENABLE_PACKAGE=ON -DENABLE_BINARY=ON -DASCEND_COMPUTE_UNIT=ascend950 "-DASCEND_OP_NAME=mat_mul_v3;batch_mat_mul_v3" -DVENDOR_NAME=allpairs -DENABLE_GEN_ACLNN=OFF "-DASCEND_COMPILE_OPS=mat_mul_v3;batch_mat_mul_v3;fused_mat_mul;matmul.common" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@"
