#!/bin/bash
set -e
P=$(cd "$(dirname "$0")" && pwd)
MODE=$1
OP=$2
CASE=$3
W=$(dirname "$(dirname "$(dirname "$P")")")
unset LD_LIBRARY_PATH PYTHONPATH LD_PRELOAD PYTHONHOME
if [ "$MODE" = bf16 ]; then
 source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
 V="$P/../bf16_local_gemm18/install/vendors/bf16gmm_transformer"
 PY="$W/.venv/bin/python"
else
 source /usr/local/Ascend/cann-9.1.0/set_env.sh
 V="$P/../transformer_dense/gemm/private_provider"
 PY=/opt/tileops-ci/runtime/bin/python
fi
export ASCEND_CUSTOM_OPP_PATH=$(realpath "$V")
export ASCEND_CLASSIC_TRANSFORMER_VENDOR=$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export PYTHONDONTWRITEBYTECODE=1 ASCEND_GLOBAL_LOG_LEVEL=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for name in ASCEND_CACHE_PATH ASCEND_WORK_PATH XDG_CACHE_HOME TORCH_EXTENSIONS_DIR ASCEND_PROCESS_LOG_PATH TRITON_CACHE_DIR TMPDIR TMP TEMP ACL_OP_COMPILER_CACHE_DIR CCACHE_DIR; do
 export "$name=$P/$CASE/runtime/$name"
 mkdir -p "${!name}"
done
mkdir -p "$P/$CASE/result"
cd "$W"
exec "$PY" -B "$P/capture_arm2.py" "$P/$CASE/result" "$OP" "$CASE" "$MODE"
