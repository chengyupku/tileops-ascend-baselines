#!/bin/bash
set -e
root=/opt/tileops-all-pairs/workspace
export INTEGRATION700_TLA_ROUTE=legacy
export TILELANG_LIBRARY_PATH="$root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
exec /opt/tileops-all-pairs/bin/allpairs-python /opt/aicpu-baseline-fix-server2-20261108/probe_constants.py
