#!/bin/bash
# RECIPE ONLY: run this entire script as npu-guard's foreground command.
set -eu
: "${BF16_CONV_GUARD_OWNER:?Explicit dedicated guard owner required}"
: "${BF16_CONV_RUN_ROOT:?Fresh external run root required}"
# Run in a clean shell; the wrapper validates the exact expansions.
unset LD_LIBRARY_PATH PYTHONPATH LD_PRELOAD PYTHONHOME FLA_NPU_DISABLE_PTH TORCH_DEVICE_BACKEND_AUTOLOAD
set +u  # Original SDK shell expands optional unset loader variables.
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
set -u
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH ASCEND_AICPU_PATH=$ASCEND_HOME_PATH ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export ASCEND_CUSTOM_OPP_PATH=/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer
export ASCEND_CLASSIC_TRANSFORMER_VENDOR=$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_GLOBAL_LOG_LEVEL=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
# Reject evidence caches and reused roots before any directory creation.
python3 -S - "$BF16_CONV_RUN_ROOT" <<'CPU_ROOT_CHECK'
import pathlib,sys
p=pathlib.Path(sys.argv[1]).resolve()
e=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026')
if p==e or e in p.parents or p.exists():
    raise SystemExit('RUN_ROOT must be fresh and outside historical evidence tree')
p.mkdir(parents=True)
CPU_ROOT_CHECK
for variable in ASCEND_CACHE_PATH ASCEND_WORK_PATH XDG_CACHE_HOME TORCH_EXTENSIONS_DIR ASCEND_PROCESS_LOG_PATH TRITON_CACHE_DIR TMPDIR TMP TEMP ACL_OP_COMPILER_CACHE_DIR CCACHE_DIR; do
  export "$variable=$BF16_CONV_RUN_ROOT/$variable"
  mkdir -p "${!variable}"
done
cd /root/dyq/workspace
exec "$@"
