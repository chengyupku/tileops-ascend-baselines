#!/bin/bash
set -eu
cd /opt/tileops-all-pairs/workspace
P=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider
export ASCEND_CLASSIC_TRANSFORMER_VENDOR="$ASCEND_CUSTOM_OPP_PATH"
for name in ASCEND_CACHE_PATH ASCEND_WORK_PATH XDG_CACHE_HOME TORCH_EXTENSIONS_DIR ASCEND_PROCESS_LOG_PATH TRITON_CACHE_DIR TMPDIR TMP TEMP ACL_OP_COMPILER_CACHE_DIR; do
 mkdir -p "$P/$name"
 export "$name=$P/$name"
done
export ASCEND_GLOBAL_LOG_LEVEL=0
export OMP_NUM_THREADS=4
exec /opt/tileops-all-pairs/bin/allpairs-python "$P/capture.py"
