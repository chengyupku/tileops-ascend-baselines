#!/bin/bash
set -e
R=/root/dyq/workspace/reports/baseline-single-agent-server2-20261027/nsa_compressed
B=/root/dyq/workspace/reports/baseline-integration-700-20261026
KEY=$1
test ! -e "$R/$KEY/checker.json"
test ! -e "$R/$KEY/exit"
mkdir -p "$R/$KEY" "$R/tmp" "$R/runtime_logs/$KEY"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source "$B/attention_generated_recovery5/native_install/vendors/attention_native_build5_transformer/bin/set_env.bash"
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$B/attention_generated_recovery5/python:$B/attention_native_build5/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$R/cache" ASCEND_CACHE_PATH="$R/ascend_cache" ASCEND_WORK_PATH="$R/ascend_work"
export ASCEND_PROCESS_LOG_PATH="$R/runtime_logs/$KEY" ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0
cd "$R"
export TMPDIR=/proc/self/cwd/tmp
set +e
npu-guard run --owner 'baseline-single-server2/nsa-compressed' --wait 600 --timeout 300 -- /opt/tileops-ci/runtime/bin/python3 "$R/runner.py" "$KEY" > "$R/$KEY/console.log" 2>&1
rc=$?
cat /run/lock/ascend-npu/device-0.lock > "$R/$KEY/guard.json"
printf '%s\n' "$rc" > "$R/$KEY/exit"
tail -35 "$R/$KEY/console.log"
exit "$rc"
