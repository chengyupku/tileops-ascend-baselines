#!/bin/bash
set -e
lib=$1
shift
root=/opt/tileops-all-pairs/workspace
export INTEGRATION700_TLA_ROUTE=legacy
if [ "$lib" = "sgl-kernel-npu" ]; then
  export LD_LIBRARY_PATH="$root/.cache/all-pairs/remote_host_builds/toolchain/lib64:${LD_LIBRARY_PATH:-}"
fi
export TILELANG_LIBRARY_PATH="$root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
exec /opt/tileops-all-pairs/bin/allpairs-python /opt/baseline-flex-20261106/server1/tla_launch.py "$@"
