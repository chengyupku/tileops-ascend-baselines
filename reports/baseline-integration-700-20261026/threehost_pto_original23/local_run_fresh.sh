#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /usr/bin/python3 -I -S -B /root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_pto_original23/local_evidence/run_case_fresh.py "$@"
