#!/bin/bash
set -eo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /opt/tileops-ci/runtime/bin/python -I -S /opt/quant-baseline-fix-server2-20261110/prepare_all.py
