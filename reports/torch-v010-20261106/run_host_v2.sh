#!/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
host=$1
case "$host" in server1|server2) ;; *) exit 2;; esac
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64
exec /opt/tileops-ci/runtime/bin/python -I -S "$root/supervisor_v2.py" "$host"
