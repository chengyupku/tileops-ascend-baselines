#!/bin/bash
set -e
ROOT=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration
unset LD_LIBRARY_PATH PYTHONPATH LD_PRELOAD PYTHONHOME
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider
export ASCEND_CLASSIC_TRANSFORMER_VENDOR=$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1 ASCEND_GLOBAL_LOG_LEVEL=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for name in ASCEND_CACHE_PATH ASCEND_WORK_PATH XDG_CACHE_HOME TORCH_EXTENSIONS_DIR ASCEND_PROCESS_LOG_PATH TRITON_CACHE_DIR TMPDIR TMP TEMP ACL_OP_COMPILER_CACHE_DIR CCACHE_DIR; do
 export "$name=$NATIVE9_RUN_ROOT/$name"
 mkdir -p "${!name}"
done
# stdlib preflight before Python runtime startup hooks/backend imports.
python3 -S "$ROOT/preflight.py"
cd /opt/tileops-all-pairs/workspace
exec /opt/tileops-ci/runtime/bin/python -B "$ROOT/runtime_check.py" "$1" "$2"
