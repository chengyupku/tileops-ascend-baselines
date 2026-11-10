#!/bin/bash
set -uo pipefail
root=/opt/torch-v010-20261106
host=$1
shift
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64
for index in "$@"; do
  task=$root/representatives/$index.json
  out=$root/preflight/representative-$index-v2-a1
  if [ -e "$out" ]; then echo "Refusing existing attempt: $out"; exit 1; fi
  mkdir -p "$out"
  cp "$task" "$out/task.json"
  if bash "$root/launch_v2.sh" prepare "$task" "$out" > "$out/prepare.log" 2>&1; then
    /opt/tileops-ci/runtime/bin/python -I -S "$root/tools/npu_guard.py" run --owner "torch-v010/$host/representative-$index" --wait 600 --timeout 1800 --receipt "$out/guard.json" -- bash "$root/launch_v2.sh" device "$task" "$out" > "$out/device.log" 2>&1
    echo "$host representative $index device rc=$?"
  else
    echo "$host representative $index prepare failed"
  fi
done
